import jax
import jax.numpy as jnp

from eqx3d.ball_query import query_ball_point


def test_ball_query():
    # Create a random point cloud.
    rng = jax.random.PRNGKey(0)

    # Do a batch of 10 point clouds.

    num_points = 1000
    num_queries = 100
    max_num_neighbors = 10

    points = jax.random.uniform(rng, (num_points, 3))

    _, rng = jax.random.split(rng, 2)
    queries = jax.random.uniform(rng, (num_queries, 3))

    results, mask = query_ball_point(0.1, max_num_neighbors, points, queries)

    assert results.shape == (num_queries, max_num_neighbors)

    # Check that all points are within the radius.
    assert jnp.all(
        jnp.linalg.norm(points[results] - queries[:, None], axis=-1) * mask <= 0.1
    )

    # Let's see if vmapping it works too.
    # Randomly sample a batch of 10 point clouds.
    rng = jax.random.PRNGKey(0)
    num_point_clouds = 10
    num_points = 1000
    num_queries = 100
    max_num_neighbors = 10

    points = jax.random.uniform(rng, (num_point_clouds, num_points, 3))

    _, rng = jax.random.split(rng, 2)

    rngs = jax.random.split(rng, num_point_clouds)

    # make the queries a strict subset of the points via a random permutation.
    # Each point cloud will have a different permutation.

    query_ixs = jax.vmap(
        lambda rng: jax.random.permutation(rng, num_points)[:num_queries]
    )(rngs)

    # index into the points to get the queries.
    queries = jax.vmap(lambda points, query_ix: points[query_ix])(points, query_ixs)

    results, mask = jax.vmap(
        lambda ps, qs: query_ball_point(0.1, max_num_neighbors, ps, qs)
    )(points, queries)
