from typing import Tuple

import jax
import jax.numpy as jnp
import jax.typing as jxt


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * src @ dst.T
    dist = dist + jnp.sum(src**2, -1).reshape(N, 1)
    dist = dist + jnp.sum(dst**2, -1).reshape(1, M)
    return dist


def query_ball_point(
    radius: float, nsample: int, xyz: jxt.ArrayLike, new_xyz: jxt.ArrayLike
) -> Tuple[jax.Array, jax.Array]:
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [N, 3]
        new_xyz: query points, [S, 3]
    Return:
        group_idx: grouped points index, [S, nsample]
    """

    # JAX implementation.
    N, C = xyz.shape
    S, _ = new_xyz.shape

    group_idx = jnp.arange(N, dtype=jnp.int32).reshape(1, N).repeat(S, axis=0)

    # Get the square distances between each point in the new_xyz and the xyz.
    sqrdists = square_distance(new_xyz, xyz)

    # Exclude points that are outside the radius. Setting these to N will ensure
    # that their values are larger.
    group_idx = jnp.where(sqrdists > radius**2, N, group_idx)

    # Now, sort the indices along the last axis. This will push the N values to
    # the end of each row.
    group_idx = jnp.sort(group_idx, axis=-1)

    # Now, we want to select the first nsample indices from each row. We can do
    # this by using the following slice.
    group_idx = group_idx[:, :nsample]
    group_mask = group_idx == N

    # Now, we want to replace the N values with the first index in each row.
    group_first = group_idx[:, 0].reshape(S, 1).repeat(nsample, axis=1)
    group_idx = jnp.where(group_mask, group_first, group_idx)

    # Return the group indices and the group mask.
    # A downstream application may want to use the group mask to mask out
    # points that are outside the radius if there are fewer than nsample points.
    return group_idx, ~group_mask
