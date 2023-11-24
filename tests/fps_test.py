import jax
import jax.numpy as jnp

from eqx3d import fps


def test_fps():
    # Create a random point cloud.
    rng = jax.random.PRNGKey(0)

    num_points = 1000
    num_samples = 100
    points = jax.random.uniform(rng, (num_points, 3))

    # Compute FPS.
    samples = fps.fps(points, num_samples, jax.random.PRNGKey(1))

    # Check that the samples are unique.
    assert len(jnp.unique(samples)) == num_samples

    # Check how it works with vmap with 10 point clouds (each different).
    points = jax.random.uniform(rng, (10, num_points, 3))
    rngs = jax.random.split(rng, 10)
    samples = jax.vmap(fps.fps, in_axes=(0, None, 0))(points, num_samples, rngs)

    # Check that the samples are unique.
    for i in range(10):
        assert len(jnp.unique(samples[i])) == num_samples
