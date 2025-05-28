
import torch, sys
from cim.cim_mm import _digital_mm
from cim.utils import _get_similarity
sys.path.append("/home/cx922/AICrossSim/acxsearch")
from chop.tools import set_excepthook
set_excepthook()

torch.manual_seed(0)
x = torch.rand(23, 256)
weight = torch.rand(256, 10)
config = {
    "x_quant_type": "e8m7",
    "weight_quant_type": "e8m7",
    "rescale_dim": "vector",
    "vector_size": 4,
    "approximate_mode": False,
}
print(x, weight)
out = _digital_mm(x, weight, config)
real_out = x @ weight
print(out, real_out)
print((out - real_out).abs().max())
