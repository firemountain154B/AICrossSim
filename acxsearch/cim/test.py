
import torch, sys
from cim_mm import _digital_mm
sys.path.append("/home/cx922/AICrossSim/acxsearch")
from chop.tools import set_excepthook
set_excepthook()

x = torch.rand(1, 4)
weight = torch.rand(4, 8)
config = {
    "x_quant_type": "e4m3",
    "weight_quant_type": "e4m3",
    "rescale_dim": "vector",
    "approximate_mode": False,
}
out = _digital_mm(x, weight, config)
real_out = x @ weight
print(out, real_out)
print(out - real_out)
