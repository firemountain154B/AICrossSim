import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..quant import scale_integer_quantizer
import sys
sys.path.append("/home/cx922/AICrossSim/acxsearch")

from ano.tools import get_logger, set_logging_verbosity

logger = get_logger(__name__)
set_logging_verbosity("debug")

# ToDo: ADD Drift Noise
# ToDo: add scaling factor, from weight to gt

def reram_programming_noise(weight, reram_weight_magnitude):
    """
    Implements weight programming noise model:
    g_prog = g_T + N(0, σ_prog)
    σ_prog = max(-1.1731g_T^2 + 1.9650g_T + 0.2635, 0)
    """
    weight_max = torch.max(torch.abs(weight))
    noise = torch.randn_like(weight) * weight_max * reram_weight_magnitude
    return weight + noise


def reram_mm(x, weight, config):
    activation_noise = config.get("activation_noise", 0.0)
    weight_noise = config.get("weight_noise", 0.0)

    x = x + torch.randn_like(x) * activation_noise
    weight = weight + torch.randn_like(weight) * weight_noise
    
    return x @ weight

def _analog_mm(x: Tensor, weight: Tensor, config: dict):
    if config.get("pcm_simulation", False):
        return pcm_mm(x, weight, config)
    elif config.get("reram_simulation", False):
        return reram_mm(x, weight, config)
    else:
        return x @ weight
    
class AnalogMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, config):
        # Save inputs for backward pass
        ctx.save_for_backward(x, weight)
        # Compute the noisy matrix multiplication
        result = _analog_mm(x, weight, config)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        x, weight = ctx.saved_tensors
        
        # Compute gradients using standard matrix multiplication
        # Ignoring the noise function during backpropagation
        grad_input = grad_output @ weight.t()
        
        grad_weight = x.transpose(-2, -1) @ grad_output
        
        # Return gradients for each input (None for config since it doesn't need gradients)
        return grad_input, grad_weight, None

def _runtime_rescale(
    x: Tensor, exponent_bits: int = 1, mantissa_bits: int = 1, rescale_dim: str = "element"
):
    """
    The rescaling in side the digital mm
    """

    if rescale_dim == "element":
        max_exponent = torch.log2(x).ceil()
    elif rescale_dim == "vector":
        max_exponent = (torch.abs(x) + 1e-8).max(dim=-1, keepdim=True).values.log2().ceil()
    else:
        raise ValueError(f"Invalid rescale_dim: {rescale_dim}")
    
    exponent_min = 0
    exponent_max = 2**exponent_bits - 1
    max_exponent = torch.clamp(max_exponent, exponent_min, exponent_max)

    mantissa = x / 2**max_exponent
    mantissa_max = 2**mantissa_bits - 1
    mantissa_min = -2**mantissa_bits

    # recast mantissa
    mantissa = torch.clamp(mantissa * 2**mantissa_bits, mantissa_min, mantissa_max)
    mantissa = mantissa.round()
    mantissa = mantissa / 2**mantissa_bits

    return mantissa * (2**max_exponent)


def _digital_mm(x: Tensor, weight: Tensor, config: dict):
    '''
    The digital mm is conducted in the following way:
    1. Reshape the x and weight to the vector-wise
    2. Conduct the digital mm
    3. Sum the result after blocking
    4. Reshape the result to the original shape

    the config should contain the following:
    - x_quant_type
    - weight_quant_type
    - rescale_dim
    - approximate_mode
    '''
    x_shape = x.shape
    weight_shape = weight.shape
    vector_size = config.get("vector_size", 1)
    
    # Pad x if not divisible by vector_size
    if x_shape[-1] % vector_size != 0:
        padding_size = vector_size - (x_shape[-1] % vector_size)
        padding_shape = list(x_shape)
        padding_shape[-1] = padding_size
        padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding], dim=-1)
        x_shape = x.shape
    
    # Pad weight if not divisible by vector_size
    if weight_shape[0] % vector_size != 0:
        padding_size = vector_size - (weight_shape[0] % vector_size)
        padding_shape = [padding_size] + list(weight_shape[1:])
        padding = torch.zeros(padding_shape, dtype=weight.dtype, device=weight.device)
        weight = torch.cat([weight, padding], dim=0)
        weight_shape = weight.shape
    assert (x_shape[-1] % vector_size == 0) and (weight.shape[0] % vector_size == 0), f"x.shape[-1] = {x_shape[-1]} and weight.shape[0] = {weight.shape[0]} must be divisible by vector_size = {vector_size}"

    px = x.reshape(-1, x_shape[-1]//vector_size, vector_size)
    px = px.permute(1, 0, 2)
    pw = weight.reshape(weight_shape[0]//vector_size, vector_size, weight_shape[1])

    out = _digital_mm_core(px, pw, config)

    out = out.sum(dim=0)
    out = out.reshape(x_shape[0:-1] + torch.Size([weight_shape[1]]))

    return out
    
class DigitalMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, config):
        ctx.save_for_backward(x, weight)
        ctx.config = config
        result = _digital_mm(x, weight, config)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_input = grad_output @ weight.t()
        grad_weight = x.transpose(-2, -1) @ grad_output
        # grad_input = _digital_mm(grad_output, weight.t(), ctx.config)
        # grad_weight = _digital_mm(x.transpose(-2, -1), grad_output, ctx.config)
        return grad_input, grad_weight, None
    
def cim_mm(x, weight, config):
    if config.get("digital_mm", False):
        return DigitalMM.apply(x, weight, config)
    else:
        return AnalogMM.apply(x, weight, config)