import math
import jax.numpy as jnp
from jax import Array


def orthogonality_fro(V: Array) -> Array:
    m = V.shape[1]
    I = jnp.eye(m, dtype=V.dtype)
    return jnp.linalg.norm(0.5 * (V.T @ V + (V.T @ V).T) - I, ord="fro") / math.sqrt(m)


def orthogonality_max(V: Array) -> Array:
    m = V.shape[1]
    I = jnp.eye(m, dtype=V.dtype)
    return jnp.max(jnp.abs(0.5 * (V.T @ V + (V.T @ V).T) - I))


def subspace_residual(M: Array, V: Array) -> Array:
    """
    Relative invariant-subspace residual

        ||G V - V(V^T G V)||_F / ||G||_F,  G = M^T M.

    This does not need an SVD reference and is the main accuracy metric here.
    """
    G = M.T @ M
    T = V.T @ G @ V
    R = G @ V - V @ T
    return jnp.linalg.norm(R, ord="fro") / (jnp.linalg.norm(G, ord="fro") + 1e-30)


def principal_angle_cosines(M: Array, V: Array):
    """
    Reference metric using exact SVD for moderate test sizes.
    Returns min/mean/max singular values of V^T V_ref.
    """
    _, _, Vh = jnp.linalg.svd(M, full_matrices=False)
    V_ref = Vh.T
    s = jnp.linalg.svd(V.T @ V_ref, compute_uv=False)
    return s.min(), s.mean(), s.max()


def condition_estimate(M: Array) -> Array:
    s = jnp.linalg.svd(M, compute_uv=False)
    smin = jnp.maximum(s[-1], jnp.asarray(1e-30, M.dtype))
    return s[0] / smin


def finite_ok(X: Array) -> Array:
    return jnp.all(jnp.isfinite(X))
