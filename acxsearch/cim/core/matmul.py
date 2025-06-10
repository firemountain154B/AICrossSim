import torch
from torch import Tensor

from .simulation_tile import sram_tile, reram_tile, pcm_tile

from ano.tools import get_logger, set_logging_verbosity
logger = get_logger(__name__)
set_logging_verbosity("debug")

# ToDo: ADD Drift Noise
# ToDo: add scaling factor, from weight to gt


def mm_tile(x: Tensor, weight: Tensor, config: dict):
    return x @ weight

def cim_tile(x, weight, config):
    if config.get("tile_type") == "digital":
        return sram_tile(x, weight, config)
    elif config.get("tile_type") == "reram":
        return reram_tile(x, weight, config)
    elif config.get("tile_type") == "pcm":
        return pcm_tile(x, weight, config)
    else:
        return mm_tile(x, weight, config)

def cim_mm(x: Tensor, weight: Tensor, config: dict):
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

    out = cim_tile(px, pw, config)

    out = out.sum(dim=0)
    out = out.reshape(x_shape[0:-1] + torch.Size([weight_shape[1]]))

    return out

class CIMCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, config):
        ctx.save_for_backward(x, weight)
        ctx.config = config
        return cim_mm(x, weight, config)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_input = grad_output @ weight.t()
        grad_weight = x.t() @ grad_output
        return grad_input, grad_weight, None

def cim_core(x, weight, config):
    return CIMCore.apply(x, weight, config)