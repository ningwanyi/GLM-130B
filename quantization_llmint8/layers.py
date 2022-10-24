import torch
from torch.nn.parameter import Parameter

from SwissArmyTransformer.mpu import copy_to_model_parallel_region
from SwissArmyTransformer.mpu import gather_from_model_parallel_region
from SwissArmyTransformer.mpu import reduce_from_model_parallel_region
from SwissArmyTransformer.mpu import scatter_to_model_parallel_region
from SwissArmyTransformer.mpu import ColumnParallelLinear, RowParallelLinear

from .functional import W8A16Linear
from kernels import compress_int4_weight
import bitsandbytes as bnb


class QuantizedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, weight_bit_width: int, weight=None, has_fp16_weights=False, threshold=6.0, index=None, *args, **kwargs):
        super(QuantizedColumnParallelLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        shape = self.weight.shape
        del self.weight

        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = bnb.nn.Int8Params(weight.data, has_fp16_weights=has_fp16_weights)
    
    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x):
        self.state.is_training = self.training

        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(x)

        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()

        output_parallel = bnb.matmul(input_parallel, self.weight, bias=self.bias, state=self.state)

        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        if not self.state.has_fp16_weights and self.state.CB is not None:
            # we converted 8-bit row major to turing/ampere format in the first inference pass
            # we no longer need the row-major weight
            del self.state.CB
            self.weight.data = self.state.CxB

        return output


class QuantizedRowParallelLinear(RowParallelLinear):
    def __init__(self, weight_bit_width: int, weight=None, has_fp16_weights=False, threshold=6.0, index=None, *args, **kwargs):
        super(QuantizedRowParallelLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        shape = self.weight.shape
        del self.weight

        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = bnb.nn.Int8Params(weight.data, has_fp16_weights=has_fp16_weights)
    
    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x):
        self.state.is_training = self.training
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = x
        else:
            input_parallel = scatter_to_model_parallel_region(x)
        
        if self.weight.CB is not None:
            self.init_8bit_state()
        
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()

        # Matrix multiply.
        output_parallel = bnb.matmul(input_parallel, self.weight, bias=self.bias, state=self.state)
        
        # All-reduce across all the partitions.
        output = reduce_from_model_parallel_region(output_parallel)

        return output
