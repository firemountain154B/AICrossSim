import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization import scale_integer_quantizer

def reram_tile(x, weight, config):
    x = scale_integer_quantizer(
        x, 
        config.get("num_bits", 8), 
        True, 
        config.get("quantile", 1.0)
    )
    weight = weight + torch.randn_like(weight) * config.get("weight_noise", 0.0)
    return x @ weight

# class ReRAMTile(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight, config):
#         ctx.save_for_backward(x, weight)
#         ctx.config = config
#         return reram_mm(x, weight, config)
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, weight = ctx.saved_tensors
#         grad_input = grad_output @ weight.t()
#         grad_weight = x.transpose(-2, -1) @ grad_output
#         return grad_input, grad_weight, None