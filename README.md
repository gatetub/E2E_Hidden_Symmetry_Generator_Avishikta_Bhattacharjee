#  Symmetry Generator Learning: Install & Quickstart
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/gatetub/E2E_Hidden_Symmetry_Generator_Avishikta_Bhattacharjee/blob/main/generator-notebook.ipynb)

### Summary
This tutorial demonstrates how to learn latent symmetry generators that preserve a quadratic invariant $$\psi(x) = x^\top M x$$ using PyTorch, with a simple training loop and visualization utilities. It includes ready-to-run examples for SO(4) (Euclidean metric) and SO(1,3) Lorentz symmetry (Minkowski metric), and a plotting function to save generator heatmaps. The notebook is designed for quick experimentation, with configurable dimension, number of generators, data size, epochs, device, and series order for the transform.

### Manifold method

This tutorial treats the target symmetry group as the isometry group of a manifold defined by a metric $$M$$, and enforces invariance of the quadratic form $$\psi(x)=x^\top M x$$ under learned transformations to align generators with the manifold’s geometry.  

- Geometry via metric: choose $$M=I$$ for Euclidean manifolds (SO(n)) or $$M=\mathrm{diag}(-1,1,\dots,1)$$ for Minkowski space (SO(1,n−1)), which fixes the notion of distance preserved by the group.  
- Lie algebra constraint: learn generator matrices $$A$$ intended to satisfy the infinitesimal isometry condition $$A^\top M + M A = 0$$, implemented implicitly by minimizing the closure loss on $$\psi$$.  
- Group action: approximate the exponential map with a truncated series $$g(\theta)$$ to transform data while preserving $$\psi$$, and use an orthogonality penalty to encourage distinct generators.

### Install
- Requirements: Python 3.11, PyTorch, Matplotlib, tqdm, optional CUDA GPU support for faster training.


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
