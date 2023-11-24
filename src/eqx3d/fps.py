import functools

import jax
import jax.numpy as jnp
import jax.typing as jxt


@functools.partial(jax.jit, static_argnames=("n_samples",))
def fps(points: jxt.ArrayLike, n_samples: int, key: jax.random.PRNGKey) -> jax.Array:
    """Furthest point sampling.

    Reimplemented in JAX, see https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py#L63C23-L63C23

    # It might be possible to use jax.lax.scan instead of a for loop to speed up
    # compilation. See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#using-python-control-flow

    Args:
        points (jxt.ArrayLike): Points to be sampled.
        n_samples (int): Number of unique samples.
        key (jax.random.PRNGKey): Random key..

    Returns:
        jax.Array: Indices of the sampled points.
    """
    N, C = points.shape
    centroids = jnp.zeros((n_samples,), dtype=jnp.int32)
    distance = jnp.ones((N,)) * 1e10
    farthest = jax.random.randint(key, tuple(), 0, N)
    for i in range(n_samples):
        centroids = centroids.at[i].set(farthest)
        centroid = points[farthest, :].reshape(1, C)
        dist = jnp.sum((points - centroid) ** 2, -1)
        distance = jnp.where(dist < distance, dist, distance)
        farthest = jnp.argmax(distance, -1)
    return centroids
