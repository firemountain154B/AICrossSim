import torch
from acxsearch.cim.core.matmul import cim_mm

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 4, 4)
    weight = torch.randn(4, 4)
    config = {
        "tile_type": "digital",
        "core_size": 2,
    }
    out = cim_mm(x, weight, config)
    print(out)
    print(x @ weight)