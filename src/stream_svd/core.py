import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from .config import StreamingPowerConfig

Array = jax.Array

def make_streaming_power_step(m: int, cfg: StreamingPowerConfig = StreamingPowerConfig()):
    """
    Returns a jitted fp32 step

        V_t = orth2((M^T M + lambda I) V_{t-1})

    implemented in the modified Gram form:

        G = M^T M
        W = (G + lambda I) V_prev
        H = V_prev^T W
        Z = scqr(W, H)
        S = Z^T Z
        V = scqr(Z, S)

    The SCQR routine is preconditioned for fp32 Cholesky as:

        1) symmetrize the Gram
        2) floor the diagonal
        3) scale to unit diagonal
        4) add a small jitter in scaled coordinates
        5) Cholesky + triangular solve
    """
    dtype = jnp.float32
    I = jnp.eye(m, dtype=dtype)
    diag_idx = jnp.diag_indices(m)

    def _sym(A: Array) -> Array:
        return 0.5 * (A + A.T)

    def _fro(A: Array) -> Array:
        return jnp.sqrt(jnp.sum(A * A))

    def _chol_qr(A: Array, Gram: Array, jitter: float) -> Array:
        Gram = _sym(Gram)

        diag = jnp.diag(Gram)
        eps = jnp.finfo(dtype).eps
        tau = cfg.diag_floor_mult * eps * jnp.maximum(1.0, jnp.max(jnp.abs(diag)))
        d = jnp.maximum(diag, tau)
        inv_sqrt_d = jax.lax.rsqrt(d)

        C = Gram * inv_sqrt_d[:, None] * inv_sqrt_d[None, :]
        C = _sym(C)
        C = C.at[diag_idx].set(1.0)

        L = jnp.linalg.cholesky(C + jnp.asarray(jitter, dtype) * I)
        A_scaled = A * inv_sqrt_d[None, :]
        Q = solve_triangular(L, A_scaled.T, lower=True).T
        return Q

    @jax.jit
    def step(M: Array, V_prev: Array) -> Array:
        M = jnp.asarray(M, dtype)
        V_prev = jnp.asarray(V_prev, dtype)

        G = M.T @ M
        lam = jnp.asarray(cfg.power_shift, dtype) * _fro(G)
        W = (G + lam * I) @ V_prev
        H = V_prev.T @ W
        Z = _chol_qr(W, H, cfg.jitter1)
        S = Z.T @ Z
        V = _chol_qr(Z, S, cfg.jitter2)
        return V

    @jax.jit
    def uv(M: Array, V_prev: Array):
        V = step(M, V_prev)
        Y = M @ V
        # Avoid division by zero with small eps
        U = Y / jnp.sqrt(jnp.sum(Y * Y, axis=0, keepdims=True) + jnp.asarray(1e-12, dtype))
        return U, V

    return step, uv
