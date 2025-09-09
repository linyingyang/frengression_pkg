# frengression

A lightweight Python package that wraps the `Frengression` and `FrengressionSeq` models.

## Install (from source)
Please fork the repository and run the following:
```bash
pip install -U build
python -m build
pip install dist/*.whl
```

## Usage Example
```python
import torch
from frengression import Frengression

# Suppose x_dim=3, y_dim=1, z_dim=2 for toy data
model = Frengression(x_dim=1, y_dim=1, z_dim=3, device="cpu")

# Dummy data
x = torch.randn(10, 1)
y = torch.randn(10, 1)
z = torch.randn(10, 3)

model.train_y(x, z, y, num_iters=100, lr=1e-4, print_every_iter=50,tol=0.000)
model.train_xz(x, z, num_iters=100, lr=1e-4, print_every_iter=50)

```
