import torch
from frengression import Frengression

def test_forward_pass():
    model = Frengression(x_dim=3, y_dim=1, z_dim=2)
    x = torch.randn(5, 3)
    y = torch.randn(5, 1)
    z = torch.randn(5, 2)
    out = model(x, y, z)
    assert out is not None
