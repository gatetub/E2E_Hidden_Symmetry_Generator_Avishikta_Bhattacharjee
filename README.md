#  Symmetry Generator Learning: Install & Quickstart

### Install
- Requirements: Python 3.11, PyTorch, Matplotlib, tqdm, optional CUDA GPU support for faster training.
- Install:
```
pip install torch matplotlib tqdm
```

### Usage
- Open generator-functions.ipynb and run cells top-to-bottom to train and visualize generators.
- SO(4) example (Euclidean metric M=I) :
```python
import torch
M = torch.eye(4, device="cuda:0")
model = train_generic_lie_group(n=4, M=M)
visualize_generators(model, filename="so4.png", group_name="SO(4)")
```
- Lorentz SO(1,3) example (Minkowski metric diag(-1,1,1,1)) :
```python
import torch

def get_metric(n, device='cpu'):
    M = torch.eye(n, device=device)
    if n > 0:
        M[0, 0] = -1.0
    return M

M = get_metric(4, device="cuda:0")
model1 = train_generic_lie_group(n=4, M=M)
visualize_generators(model1, filename="Lorentz4.png", group_name="L(4)")
```
