#  Symmetry Generator Learning: Install & Quickstart

### Summary
This tutorial demonstrates how to learn latent symmetry generators that preserve a quadratic invariant $$ \psi(x) = x^\top M x $$ using PyTorch, with a simple training loop and visualization utilities. It includes ready-to-run examples for SO(4) (Euclidean metric) and SO(1,3) Lorentz symmetry (Minkowski metric), and a plotting function to save generator heatmaps. The notebook is designed for quick experimentation, with configurable dimension, number of generators, data size, epochs, device, and series order for the transform.

### Install
- Requirements: Python 3.11, PyTorch, Matplotlib, tqdm, optional CUDA GPU support for faster training.

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
