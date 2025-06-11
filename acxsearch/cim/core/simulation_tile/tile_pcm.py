import torch
from torch import Tensor
from .quantization import scale_integer_quantizer

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

    analog_weight = programming_noise(analog_weight)
    
    result = analog_x @ analog_weight
        
    result = read_noise(analog_weight, analog_x, result)

    return result

def pcm_tile(x, weight, config):
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
