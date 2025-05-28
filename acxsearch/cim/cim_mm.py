import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .quant import scale_integer_quantizer
import sys
sys.path.append("/home/cx922/AICrossSim/acxsearch")
from .utils import _get_similarity

from chop.tools import get_logger, set_logging_verbosity

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

def programming_noise(weight):
    """
    Implements PCM programming noise model:
    g_prog = g_T + N(0, σ_prog)
    σ_prog = max(-1.1731g_T^2 + 1.9650g_T + 0.2635, 0)
    
    Args:
        weight (torch.Tensor): Target weight values (g_T)
    
    Returns:
        torch.Tensor: Noisy weight values (g_prog)
    """
    # Calculate σ_prog using the quadratic equation
    sigma_prog = -1.1731 * weight**2 + 1.9650 * weight + 0.2635
    # Ensure σ_prog is non-negative
    sigma_prog = torch.maximum(sigma_prog, torch.zeros_like(sigma_prog))
    
    # Add noise from normal distribution N(0, σ_prog)
    noise = torch.randn_like(weight) * torch.sqrt(sigma_prog)
    g_prog = weight + noise
    
    return g_prog

def read_noise(analog_weight, analog_x, result):
    """
    Implements short-term PCM read noise model:
    Calculate weight-dependent noise standard deviation
    σ_i^W = σ_0^W * sqrt(sum_j |w_ij| |x_j|^2)
    σ_0^W = 0.0175
    Args:
        analog_weight (torch.Tensor): Weight values
        analog_x (torch.Tensor): Input values
        result (torch.Tensor): Result of analog matrix multiplication
    Returns:
        torch.Tensor: Result with read noise
    """
    sigma_0 = 0.0175
    
    # Calculate the weight-dependent term
    # Using absolute values of weights and squared inputs
    noise_term = torch.sqrt((analog_x ** 2)@torch.abs(analog_weight))
    
    # Generate noise with the calculated standard deviation
    noise = torch.randn_like(result) * sigma_0 * noise_term
    
    return result + noise

def pcm_mm_core(analog_x, analog_weight, config):
    """
    Implements the core analog matrix multiplication with noise components.
    
    The equation for PCM-based computation can be represented as:
    y_i = σ^out·ξ_i + (Δy_i^IR-drop + Σ_j((w_ij + σ^W·ξ_ij)·x_j)
    
    Where:
    - σ^out·ξ_i: Read noise component
    - Δy_i^IR-drop: IR drop noise
    - σ^W·ξ_ij: Weight noise component
    - x_j: Input values
    
    Args:
        analog_x (torch.Tensor): Quantized input tensor
        analog_weight (torch.Tensor): Quantized weight tensor
        config (dict): Configuration parameters for noise simulation
        
    Returns:
        torch.Tensor: Result of analog matrix multiplication with noise effects
    """
    # Notice the normalized weight here is just the normalized conductance
    # First we need to transform it to real conductance
    # Assume the gmax is 5us(the result is from the original paper)

    if config.get("reram_programming_noise", False):
        reram_weight_magnitude = config.get("reram_weight_magnitude", 0.10)
        analog_weight = reram_programming_noise(analog_weight, reram_weight_magnitude)
    
    if config.get("programming_noise", False):
        analog_weight = programming_noise(analog_weight)
    
    result = analog_x @ analog_weight
    # TODO: add irdrop noise currently negelect it
    if config.get("read_noise", False):
        result = read_noise(analog_weight, analog_x, result)

    return result

def pcm_mm(x, weight, config):
    """
    Implements noisy matrix multiplication for PCM-based computation.

    The general equation for PCM-based computation can be represented as:
    y_i = α·γ_i·quant_out(F_i(quant_in(x/α)))
    
    Where:
    - quant_in: Input quantization for the DAC
    - F_i: Real Computation with noise (programming noise, read noise, etc.)
    - quant_out: Output quantization for ADC
    - α, γ: Scaling factors for x and weight
    - β: Bias term
    Args:
        x (torch.Tensor): Input tensor
        weight (torch.Tensor): Weight tensor
        bias (torch.Tensor): Bias tensor
        config (dict): Configuration parameters for noise and quantization
        
    Returns:
        torch.Tensor: Result of noisy matrix multiplication
    """
    quantile = config.get("quantile", 1.0)
    width = config.get("width", 8)
    is_signed = config.get("is_signed", True)
    gmax = config.get("gmax", 5)
    
    x_quant, analog_x, scale_x = scale_integer_quantizer(x, width, is_signed, quantile)
    weight_quant, analog_weight, scale_weight = scale_integer_quantizer(weight, width, is_signed, quantile)

    analog_weight = analog_weight.mul(gmax)
    scale_weight = scale_weight.mul(gmax)
    analog_out = pcm_mm_core(analog_x, analog_weight, config)

    adc_out, _, _ = scale_integer_quantizer(analog_out, width, is_signed, quantile)

    result = adc_out.div(scale_x).div(scale_weight)

    return result

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

    
def _digital_mm_core(x: Tensor, weight: Tensor, config: dict):
    '''
    There is two mode to conducting the digital mm, 
    For the first accurate mode, 
    we only need to rescale the x and weight, 
    and then quantize output to simulate the behaviour

    For the second mode, 
    we need to simulate the lossy digital multiplication 
    and the lossy digital accumulation
    '''
    
    x_quant_type = config.get("x_quant_type")
    weight_quant_type = config.get("weight_quant_type")
    

    if x_quant_type == "e4m3":
        qx = _runtime_rescale(x, 4, 3, config.get("rescale_dim", "vector"))
    elif x_quant_type == "e5m2":
        qx = _runtime_rescale(x, 5, 2, config.get("rescale_dim", "vector"))
    elif x_quant_type == "e8m7":
        qx = _runtime_rescale(x, 8, 7, config.get("rescale_dim", "vector"))
    elif x_quant_type == "int4":
        qx = scale_integer_quantizer(x, 4, True, 1.0)
    elif x_quant_type == "int8":
        qx = scale_integer_quantizer(x, 8, True, 1.0)
    else:
        qx = x

    weight = weight.transpose(-1, -2) # the rescale dimension should be in the -2 dimension 
    if weight_quant_type == "e4m3":
        qweight = _runtime_rescale(weight, 4, 3, config.get("rescale_dim", "vector"))
    elif weight_quant_type == "e5m2":
        qweight = _runtime_rescale(weight, 5, 2, config.get("rescale_dim", "vector"))
    elif weight_quant_type == "e8m7":
        qweight = _runtime_rescale(weight, 8, 7, config.get("rescale_dim", "vector"))
    elif weight_quant_type == "int4":
        qweight = scale_integer_quantizer(weight, 4, True, 1.0)
    elif weight_quant_type == "int8":
        qweight = scale_integer_quantizer(weight, 8, True, 1.0)
    else:
        qweight = weight

    # similarity = _get_similarity(qx, x, metric="cosine")

    qweight = qweight.transpose(-1, -2) # permute back
    if config.get("approximate_mode", False):
        raise NotImplementedError("Approximate mode is not implemented")
    else:
        return qx @ qweight # Considering in the flow of the paper there is no cast while sending back to AHB, so no cast in the end


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