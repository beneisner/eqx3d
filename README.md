# eqx3d

JAX+Equinox implementations of common neural network architectures for 3D data (mostly point clouds).

Current implementations:
- PointNet (Qi et al. 2017)
    - `src/eqx3d/pointnet.py`: The original PointNet architecture. Architecture is based on the Pytorch implementation found [here](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_cls.py).
    - `scripts/demo.py`: A simple demo script that trains a PointNet model on the ShapeNet dataset.

## Installation

### Requirements

Mostly this should be pure `jax` and `equinox`. If you want to run the demos for training you'll also need some `torch` and `torch_geometric` dependencies.

### Install

First, install jaq somehow: https://jax.readthedocs.io/en/latest/installation.html

Then, install the rest of the dependencies:

```bash
pip install -e ".[develop,notebooks,examples]"
```
