import jax.numpy as jnp
from jax import random


def rand_orth(key, rows: int, cols: int, dtype=jnp.float32):
    A = random.normal(key, (rows, cols), dtype)
    Q, _ = jnp.linalg.qr(A, mode="reduced")
    return Q


def make_gaussian(key, n, m, dtype=jnp.float32):
    return random.normal(key, (n, m), dtype)


def make_scaled_columns(key, n, m, spread=1e4, dtype=jnp.float32):
    k1, k2 = random.split(key)
    M = random.normal(k1, (n, m), dtype)
    scales = jnp.geomspace(jnp.asarray(1.0, dtype), jnp.asarray(spread, dtype), m)
    perm = random.permutation(k2, m)
    return M * scales[perm][None, :]


def make_controlled_spectrum(key, n, m, cond=1e6, dtype=jnp.float32):
    k1, k2 = random.split(key)
    U = rand_orth(k1, n, m, dtype)
    V = rand_orth(k2, m, m, dtype)
    s = jnp.geomspace(jnp.asarray(1.0, dtype), jnp.asarray(1.0 / cond, dtype), m)
    return (U * s[None, :]) @ V.T


def make_near_rank_deficient(key, n, m, rank=None, noise=1e-5, dtype=jnp.float32):
    if rank is None:
        rank = max(1, m // 8)
    k1, k2, k3 = random.split(key, 3)
    A = random.normal(k1, (n, rank), dtype)
    B = random.normal(k2, (rank, m), dtype)
    return A @ B + noise * random.normal(k3, (n, m), dtype)


def make_repeated_singular_values(key, n, m, dtype=jnp.float32):
    k1, k2 = random.split(key)
    U = rand_orth(k1, n, m, dtype)
    V = rand_orth(k2, m, m, dtype)
    a = m // 3
    b = m // 3
    c = m - a - b
    s = jnp.concatenate(
        [
            jnp.full((a,), 1.0, dtype),
            jnp.full((b,), 1e-1, dtype),
            jnp.full((c,), 1e-3, dtype),
        ]
    )
    return (U * s[None, :]) @ V.T


def make_correlated_columns(key, n, m, rho=0.999, dtype=jnp.float32):
    Z = random.normal(key, (n, m), dtype)
    idx = jnp.arange(m, dtype=dtype)
    T = rho ** jnp.abs(idx[:, None] - idx[None, :])
    L = jnp.linalg.cholesky(T + 1e-6 * jnp.eye(m, dtype=dtype))
    return Z @ L.T


def make_low_rank_plus_spikes(
    key, n, m, rank=8, spike_scale=100.0, noise=1e-2, dtype=jnp.float32
):
    k1, k2, k3, k4, k5 = random.split(key, 5)
    A = random.normal(k1, (n, rank), dtype)
    B = random.normal(k2, (rank, m), dtype)
    M = A @ B + noise * random.normal(k3, (n, m), dtype)
    cols = random.randint(k4, (max(1, m // 16),), 0, m)
    rows = random.randint(k5, (cols.shape[0],), 0, n)
    spikes = jnp.zeros((n, m), dtype)
    spikes = spikes.at[rows, cols].set(jnp.asarray(spike_scale, dtype))
    return M + spikes


def make_sparseish(key, n, m, density=0.02, dtype=jnp.float32):
    k1, k2 = random.split(key)
    mask = random.bernoulli(k1, p=density, shape=(n, m))
    vals = random.normal(k2, (n, m), dtype)
    return jnp.where(mask, vals, jnp.zeros((n, m), dtype))


def make_row_scale_imbalance(key, n, m, spread=1e5, dtype=jnp.float32):
    k1, k2 = random.split(key)
    M = random.normal(k1, (n, m), dtype)
    scales = jnp.geomspace(jnp.asarray(1.0, dtype), jnp.asarray(spread, dtype), n)
    perm = random.permutation(k2, n)
    return M * scales[perm][:, None]


def make_almost_duplicate_columns(key, n, m, eps=1e-4, dtype=jnp.float32):
    k1, k2 = random.split(key)
    base = random.normal(k1, (n, 1), dtype)
    coeff = jnp.linspace(0.9, 1.1, m, dtype=dtype)[None, :]
    return base @ coeff + eps * random.normal(k2, (n, m), dtype)


def make_cancellation_structure(key, n, m, dtype=jnp.float32):
    k1, k2, k3 = random.split(key, 3)
    U = random.normal(k1, (n, 4), dtype)
    A = random.normal(k2, (4, m), dtype)
    B = random.normal(k3, (4, m), dtype)
    return (
        U @ A
        - jnp.asarray(0.999, dtype) * (U @ B)
        + 1e-3 * random.normal(k1, (n, m), dtype)
    )


def make_tall_skinny(key, n=16384, m=128, dtype=jnp.float32):
    return random.normal(key, (n, m), dtype)


def make_stream(key, n, m, steps=8, drift=1e-2, dtype=jnp.float32):
    keys = random.split(key, steps + 1)
    M = random.normal(keys[0], (n, m), dtype)
    yield M
    for i in range(steps):
        M = M + drift * random.normal(keys[i + 1], (n, m), dtype)
        yield M


def make_rotating_stream(key, n, m, steps=8, angle=0.02, dtype=jnp.float32):
    """
    Slowly rotates the right singular subspace while keeping the spectrum fixed.
    """
    k1, k2, k3 = random.split(key, 3)
    U = rand_orth(k1, n, m, dtype)
    V = rand_orth(k2, m, m, dtype)
    # Omega is a skew-symmetric matrix for rotation.
    Omega = random.normal(k3, (m, m), dtype)
    Omega = 0.5 * (Omega - Omega.T)
    s = jnp.geomspace(jnp.asarray(1.0, dtype), jnp.asarray(1e-3, dtype), m)

    Vt = V
    for _ in range(steps + 1):
        yield (U * s[None, :]) @ Vt.T
        # Update rotation: V' = V * exp(angle * Omega) approx V * (I + angle * Omega)
        Vt = Vt @ (jnp.eye(m, dtype=dtype) + angle * Omega)
        Vt, _ = jnp.linalg.qr(Vt, mode="reduced")


def make_bad_v0_random(key, m, dtype=jnp.float32):
    return random.normal(key, (m, m), dtype)


def make_bad_v0_collinear(key, m, dtype=jnp.float32):
    base = random.normal(key, (m, 1), dtype)
    return base @ jnp.ones((1, m), dtype=dtype) + 1e-3 * random.normal(
        key, (m, m), dtype
    )
