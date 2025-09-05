# frengression

A lightweight Python package that wraps the `Frengression` model.

## Install (from source)
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
model = Frengression(x_dim=3, y_dim=1, z_dim=2)

# Dummy data
x = torch.randn(10, 3)
y = torch.randn(10, 1)
z = torch.randn(10, 2)

# Forward call (adjust depending on your class signature)
out = model(x, y, z)
print(out)
```
