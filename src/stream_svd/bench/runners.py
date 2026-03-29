import time
import jax
import jax.numpy as jnp
from jax import random

from ..core import make_streaming_power_step
from ..metrics import (
    orthogonality_fro,
    orthogonality_max,
    subspace_residual,
    principal_angle_cosines,
    condition_estimate,
    finite_ok,
)
from . import generators as gen


def _time_call(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    return out, 1000.0 * (t1 - t0)


def _fmt(x):
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "yes" if x else "no"
    if x is None:
        return "-"
    if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.4f}"


def print_table(rows, columns):
    if not rows:
        return
    widths = {c: max(len(c), max(len(_fmt(r.get(c))) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(_fmt(r.get(c)).ljust(widths[c]) for c in columns))


def evaluate_case(step, name, M, V0, compute_svd_reference=True):
    # Warmup to compile.
    _ = step(M, V0)
    jax.block_until_ready(_)

    V, ms = _time_call(step, M, V0)
    n, m = M.shape
    # Approximate GFLOPS for the power step:
    # G = M.T @ M (2*n*m^2)
    # W = (G + lam*I) @ V_prev (2*m^3)
    # H = V_prev.T @ W (2*m^3)
    # S = Z.T @ Z (2*m^3)
    # Two _chol_qr calls: 2 * (1/3*m^3 [chol] + m^3 [solve]) = 2.66*m^3
    # Total ~ 2*n*m^2 + 8.66*m^3
    flops = 2.0 * n * m**2 + (26.0 / 3.0) * m**3
    gflops = (flops / 1e9) / (ms / 1000.0)

    row = {
        "case": name,
        "n": int(n),
        "m": int(m),
        "ms": ms,
        "gflops": gflops,
        "finite": bool(finite_ok(V)),
        "orth_fro": float(orthogonality_fro(V)),
        "orth_max": float(orthogonality_max(V)),
        "residual": float(subspace_residual(M, V)),
        "cond_est": float(condition_estimate(M)),
    }
    if compute_svd_reference:
        cmin, cmean, cmax = principal_angle_cosines(M, V)
        row["cos_min"] = float(cmin)
        row["cos_mean"] = float(cmean)
        row["cos_max"] = float(cmax)
    return row


def build_suite_factories(n: int = 16384, m: int = 4096, seed: int = 0):
    """
    Returns a list of (name, factory_fn) pairs to avoid OOM by generating data lazily.
    """
    k = random.PRNGKey(seed)
    keys = random.split(k, 17)

    return [
        (
            "gaussian",
            lambda: (gen.make_gaussian(keys[0], n, m), jnp.eye(m, dtype=jnp.float32)),
        ),
        (
            "scaled_columns_1e4",
            lambda: (
                gen.make_scaled_columns(keys[1], n, m, spread=1e4),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "controlled_spectrum_1e3",
            lambda: (
                gen.make_controlled_spectrum(keys[2], n, m, cond=1e3),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "controlled_spectrum_1e6",
            lambda: (
                gen.make_controlled_spectrum(keys[3], n, m, cond=1e6),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "near_rank_deficient",
            lambda: (
                gen.make_near_rank_deficient(
                    keys[4], n, m, rank=max(1, m // 8), noise=1e-5
                ),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "repeated_singular_values",
            lambda: (
                gen.make_repeated_singular_values(keys[5], n, m),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "correlated_columns_rho_0.999",
            lambda: (
                gen.make_correlated_columns(keys[6], n, m, rho=0.999),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "low_rank_plus_spikes",
            lambda: (
                gen.make_low_rank_plus_spikes(
                    keys[7], n, m, rank=8, spike_scale=100.0, noise=1e-2
                ),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "sparseish_2pct",
            lambda: (
                gen.make_sparseish(keys[8], n, m, density=0.02),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "row_scale_imbalance_1e5",
            lambda: (
                gen.make_row_scale_imbalance(keys[9], n, m, spread=1e5),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "almost_duplicate_columns",
            lambda: (
                gen.make_almost_duplicate_columns(keys[10], n, m, eps=1e-4),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "cancellation_structure",
            lambda: (
                gen.make_cancellation_structure(keys[11], n, m),
                jnp.eye(m, dtype=jnp.float32),
            ),
        ),
        (
            "tall_skinny_16384x128",
            lambda: (
                gen.make_tall_skinny(keys[12], 16384, 128),
                jnp.eye(128, dtype=jnp.float32),
            ),
        ),
        (
            "bad_v0_random_dense",
            lambda: (
                gen.make_controlled_spectrum(keys[13], n, m, cond=1e5),
                gen.make_bad_v0_random(keys[14], m),
            ),
        ),
        (
            "bad_v0_nearly_collinear",
            lambda: (
                gen.make_controlled_spectrum(keys[13], n, m, cond=1e5),
                gen.make_bad_v0_collinear(keys[15], m),
            ),
        ),
        (
            "bad_v0_random_orth",
            lambda: (
                gen.make_controlled_spectrum(keys[13], n, m, cond=1e5),
                gen.rand_orth(keys[16], m, m),
            ),
        ),
    ]


def run_suite(
    n: int = 2048,
    m: int = 128,
    seed: int = 0,
    compute_svd_reference: bool = True,
    **sp_kwargs,
):
    factories = build_suite_factories(n=n, m=m, seed=seed)
    rows = []
    for name, factory in factories:
        print(f"Evaluating {name}...")
        M, V0 = factory()
        m_case = M.shape[1]
        step_m, _ = make_streaming_power_step(m_case, **sp_kwargs)
        row = evaluate_case(step_m, name, M, V0, compute_svd_reference)
        rows.append(row)
        del M
        del V0
        jax.clear_caches()

    cols = [
        "case",
        "n",
        "m",
        "ms",
        "gflops",
        "finite",
        "orth_fro",
        "orth_max",
        "residual",
        "cond_est",
    ]
    if compute_svd_reference:
        cols += ["cos_min", "cos_mean", "cos_max"]
    print("\nSingle-step suite")
    print_table(rows, cols)

    summary = {
        "cases": len(rows),
        "failures": sum(0 if r["finite"] else 1 for r in rows),
        "mean_ms": sum(r["ms"] for r in rows) / len(rows),
        "worst_orth_fro": max(r["orth_fro"] for r in rows),
        "worst_residual": max(r["residual"] for r in rows),
        "worst_cond_est": max(r["cond_est"] for r in rows),
    }
    if compute_svd_reference:
        summary["worst_cos_min"] = min(r["cos_min"] for r in rows)
    print("\nSingle-step summary")
    print_table([summary], list(summary.keys()))
    return rows


def run_streaming_examples(
    n: int = 2048,
    m: int = 128,
    seed: int = 0,
    streaming_steps: int = 8,
    streaming_drift: float = 1e-2,
    compute_svd_reference: bool = True,
    **sp_kwargs,
):
    step, _ = make_streaming_power_step(m, **sp_kwargs)
    k = random.PRNGKey(seed + 123)
    k1, k2 = random.split(k)

    experiments = [
        (
            "drifting_gaussian",
            lambda: gen.make_stream(k1, n, m, streaming_steps, streaming_drift),
        ),
        (
            "rotating_subspace",
            lambda: gen.make_rotating_stream(k2, n, m, streaming_steps, angle=0.02),
        ),
    ]

    all_rows = []
    for name, stream_factory in experiments:
        print(f"\nStreaming example: {name}")
        V = jnp.eye(m, dtype=jnp.float32)
        rows = []
        for t, M in enumerate(stream_factory()):
            # Warm state update.
            _ = step(M, V)
            jax.block_until_ready(_)
            V, ms = _time_call(step, M, V)

            n_val, m_val = M.shape
            # Total ~ 2*n*m^2 + 8.66*m^3 (see evaluate_case for breakdown)
            flops = 2.0 * n_val * m_val**2 + (26.0 / 3.0) * m_val**3
            gflops = (flops / 1e9) / (ms / 1000.0)

            row = {
                "experiment": name,
                "t": t,
                "ms": ms,
                "gflops": gflops,
                "finite": bool(finite_ok(V)),
                "orth_fro": float(orthogonality_fro(V)),
                "residual": float(subspace_residual(M, V)),
            }
            if compute_svd_reference:
                cmin, cmean, cmax = principal_angle_cosines(M, V)
                row["cos_min"] = float(cmin)
                row["cos_mean"] = float(cmean)
                row["cos_max"] = float(cmax)
            rows.append(row)
            all_rows.append(row)
            # Explicit delete of large matrix M and clear cache each step.
            del M
            jax.clear_caches()

        cols = ["experiment", "t", "ms", "gflops", "finite", "orth_fro", "residual"]
        if compute_svd_reference:
            cols += ["cos_min", "cos_mean", "cos_max"]
        print_table(rows, cols)

    summary_rows = []
    for name, _ in experiments:
        rows = [r for r in all_rows if r["experiment"] == name]
        summary = {
            "experiment": name,
            "steps": len(rows),
            "mean_ms": sum(r["ms"] for r in rows) / len(rows),
            "mean_gflops": sum(r["gflops"] for r in rows) / len(rows),
            "worst_orth_fro": max(r["orth_fro"] for r in rows),
            "worst_residual": max(r["residual"] for r in rows),
        }
        if compute_svd_reference:
            summary["worst_cos_min"] = min(r["cos_min"] for r in rows)
        summary_rows.append(summary)

    print("\nStreaming summary")
    print_table(summary_rows, list(summary_rows[0].keys()))
    return all_rows


def run_bad_init_recovery(
    n: int = 2048,
    m: int = 128,
    seed: int = 0,
    streaming_steps: int = 8,
    compute_svd_reference: bool = True,
    **sp_kwargs,
):
    step, _ = make_streaming_power_step(m, **sp_kwargs)
    k = random.PRNGKey(seed + 999)
    k1, k2 = random.split(k)

    M = gen.make_controlled_spectrum(k1, n, m, cond=1e5)
    V = gen.make_bad_v0_collinear(k2, m)

    rows = []
    for t in range(streaming_steps + 1):
        _ = step(M, V)
        jax.block_until_ready(_)
        V, ms = _time_call(step, M, V)
        row = {
            "t": t,
            "ms": ms,
            "finite": bool(finite_ok(V)),
            "orth_fro": float(orthogonality_fro(V)),
            "residual": float(subspace_residual(M, V)),
        }
        if compute_svd_reference:
            cmin, cmean, cmax = principal_angle_cosines(M, V)
            row["cos_min"] = float(cmin)
            row["cos_mean"] = float(cmean)
            row["cos_max"] = float(cmax)
        rows.append(row)

    cols = ["t", "ms", "finite", "orth_fro", "residual"]
    if compute_svd_reference:
        cols += ["cos_min", "cos_mean", "cos_max"]
    print("\nRecovery from nearly-collinear V0")
    print_table(rows, cols)
    return rows


def run_all(
    n: int = 2048,
    m: int = 128,
    seed: int = 0,
    streaming_steps: int = 8,
    streaming_drift: float = 1e-2,
    compute_svd_reference: bool = True,
    **sp_kwargs,
):
    print("Streaming power iteration, fp32 only")
    for k, v in sp_kwargs.items():
        print(f"{k:<12} = {v}")

    run_suite(
        n=n, m=m, seed=seed, compute_svd_reference=compute_svd_reference, **sp_kwargs
    )
    run_streaming_examples(
        n=n,
        m=m,
        seed=seed,
        streaming_steps=streaming_steps,
        streaming_drift=streaming_drift,
        compute_svd_reference=compute_svd_reference,
        **sp_kwargs,
    )
    run_bad_init_recovery(
        n=n,
        m=m,
        seed=seed,
        streaming_steps=streaming_steps,
        compute_svd_reference=compute_svd_reference,
        **sp_kwargs,
    )
