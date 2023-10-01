import equinox as eqx
import jax

from eqx3d.pointnet import PointNet


def test_pointnet_forward():
    key = jax.random.PRNGKey(1234)
    k1, k2, k3 = jax.random.split(key, 3)

    model = PointNet(10, k1)
    state = eqx.nn.State(model)
    model = eqx.tree_inference(model, value=True)

    x = jax.random.normal(k2, (3, 1024))
    res, _ = model(x, state, k3)

    assert res.shape == (10,)
