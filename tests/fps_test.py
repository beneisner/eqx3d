import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch_cluster import fps as torch_fps

from eqx3d import fps


@pytest.mark.skipif(not jax.local_devices(), reason="No JAX devices.")
def test_benchmark_fps(benchmark):
    # Create a random point cloud.
    rng = jax.random.PRNGKey(0)

    # Do a batch of 10 point clouds.

    num_points = 10000
    num_samples = 1000
    points = jax.random.uniform(rng, (10, num_points, 3))
    rngs = jax.random.split(rng, 10)

    # Do it once to compile.
    samples = jax.vmap(fps.fps, in_axes=(0, None, 0))(points, num_samples, rngs)

    print("Done with the first one!!!")

    def _benchmark():
        # Compute FPS.
        # Vmap.
        rng = jax.random.PRNGKey(time.time_ns())
        rngs_all = jax.random.split(rng, 11)
        rng, rngs = rngs_all[0], rngs_all[1:]
        samples = jax.vmap(fps.fps, in_axes=(0, None, 0))(points, num_samples, rngs)

    # Compute FPS.
    benchmark(_benchmark)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
def test_benchmark_torch_fps(benchmark):
    # Benchmark torch version.
    # Create a random point cloud.
    rng = jax.random.PRNGKey(0)

    # Do a batch of 10 point clouds.

    num_points = 10000
    num_samples = 1000
    points = jax.random.uniform(rng, (10, num_points, 3))
    # rngs = jax.random.split(rng, 10)

    # Convert to torch.
    points = np.asarray(points)
    # Make a copy
    points = points.copy()
    points = torch.from_numpy(points).reshape(-1, 3).cuda()

    batch = torch.arange(10).repeat_interleave(num_points).cuda()

    def _benchmark():
        # Compute FPS.
        # Vmap.
        with torch.no_grad():
            samples = torch_fps(points, batch, ratio=0.1)

    benchmark(_benchmark)


def test_fps():
    # Create a random point cloud.
    rng = jax.random.PRNGKey(0)

    num_points = 1000
    num_samples = 100
    points = jax.random.uniform(rng, (1, num_points, 3))

    # Compute FPS. Make sure to compile.
    rngs = jax.random.split(rng, 1)
    vmap_fps = jax.vmap(fps.fps, in_axes=(0, None, 0))

    samples = vmap_fps(points, num_samples, rngs)

    # Check that the samples are unique.
    assert len(jnp.unique(samples[0])) == num_samples
