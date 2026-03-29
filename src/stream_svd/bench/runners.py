import time
import jax
import jax.numpy as jnp
from jax import random

from ..core import make_streaming_power_step
from ..metrics import (
    orthogonality_fro, orthogonality_max, subspace_residual,
    principal_angle_cosines, condition_estimate, finite_ok
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
        return 'yes' if x else 'no'
    if x is None:
        return '-'
    if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
        return f'{x:.2e}'
    return f'{x:.4f}'


def print_table(rows, columns):
    if not rows:
        return
    widths = {c: max(len(c), max(len(_fmt(r.get(c))) for r in rows)) for c in columns}
    header = ' | '.join(c.ljust(widths[c]) for c in columns)
    sep = '-+-'.join('-' * widths[c] for c in columns)
    print(header)
    print(sep)
    for r in rows:
        print(' | '.join(_fmt(r.get(c)).ljust(widths[c]) for c in columns))


def evaluate_case(step, name, M, V0, compute_svd_reference=True):
    # Warmup to compile.
    _ = step(M, V0)
    jax.block_until_ready(_)

    V, ms = _time_call(step, M, V0)
    row = {
        'case': name,
        'n': int(M.shape[0]),
        'm': int(M.shape[1]),
        'ms': ms,
        'finite': bool(finite_ok(V)),
        'orth_fro': float(orthogonality_fro(V)),
        'orth_max': float(orthogonality_max(V)),
        'residual': float(subspace_residual(M, V)),
        'cond_est': float(condition_estimate(M)),
    }
    if compute_svd_reference:
        cmin, cmean, cmax = principal_angle_cosines(M, V)
        row['cos_min'] = float(cmin)
        row['cos_mean'] = float(cmean)
        row['cos_max'] = float(cmax)
    return row


def build_suite(n: int = 2048, m: int = 128, seed: int = 0):
    k = random.PRNGKey(seed)
    keys = random.split(k, 17)

    base = [
        {'name': 'gaussian', 'M': gen.make_gaussian(keys[0], n, m), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'scaled_columns_1e4', 'M': gen.make_scaled_columns(keys[1], n, m, spread=1e4), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'controlled_spectrum_1e3', 'M': gen.make_controlled_spectrum(keys[2], n, m, cond=1e3), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'controlled_spectrum_1e6', 'M': gen.make_controlled_spectrum(keys[3], n, m, cond=1e6), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'near_rank_deficient', 'M': gen.make_near_rank_deficient(keys[4], n, m, rank=max(1, m // 8), noise=1e-5), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'repeated_singular_values', 'M': gen.make_repeated_singular_values(keys[5], n, m), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'correlated_columns_rho_0.999', 'M': gen.make_correlated_columns(keys[6], n, m, rho=0.999), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'low_rank_plus_spikes', 'M': gen.make_low_rank_plus_spikes(keys[7], n, m, rank=8, spike_scale=100.0, noise=1e-2), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'sparseish_2pct', 'M': gen.make_sparseish(keys[8], n, m, density=0.02), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'row_scale_imbalance_1e5', 'M': gen.make_row_scale_imbalance(keys[9], n, m, spread=1e5), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'almost_duplicate_columns', 'M': gen.make_almost_duplicate_columns(keys[10], n, m, eps=1e-4), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'cancellation_structure', 'M': gen.make_cancellation_structure(keys[11], n, m), 'V0': jnp.eye(m, dtype=jnp.float32)},
        {'name': 'tall_skinny_16384x128', 'M': gen.make_tall_skinny(keys[12], 16384, 128), 'V0': jnp.eye(128, dtype=jnp.float32)},
    ]

    bad_M = gen.make_controlled_spectrum(keys[13], n, m, cond=1e5)
    base.extend([
        {'name': 'bad_v0_random_dense', 'M': bad_M, 'V0': gen.make_bad_v0_random(keys[14], m)},
        {'name': 'bad_v0_nearly_collinear', 'M': bad_M, 'V0': gen.make_bad_v0_collinear(keys[15], m)},
        {'name': 'bad_v0_random_orth', 'M': bad_M, 'V0': gen.rand_orth(keys[16], m, m)},
    ])
    return base


def run_suite(
    n: int = 2048,
    m: int = 128,
    seed: int = 0,
    compute_svd_reference: bool = True,
    **sp_kwargs
):
    cases = build_suite(n=n, m=m, seed=seed)
    max_m = max(case['M'].shape[1] for case in cases)
    step, _ = make_streaming_power_step(max_m, **sp_kwargs)

    rows = []
    for case in cases:
        name, M, V0 = case['name'], case['M'], case['V0']
        m_case = M.shape[1]
        # Rebuild only if width changes.
        step_m, _ = (step, None) if m_case == max_m else make_streaming_power_step(m_case, **sp_kwargs)
        row = evaluate_case(step_m, name, M, V0, compute_svd_reference)
        rows.append(row)

    cols = ['case', 'n', 'm', 'ms', 'finite', 'orth_fro', 'orth_max', 'residual', 'cond_est']
    if compute_svd_reference:
        cols += ['cos_min', 'cos_mean', 'cos_max']
    print('\nSingle-step suite')
    print_table(rows, cols)

    summary = {
        'cases': len(rows),
        'failures': sum(0 if r['finite'] else 1 for r in rows),
        'mean_ms': sum(r['ms'] for r in rows) / len(rows),
        'worst_orth_fro': max(r['orth_fro'] for r in rows),
        'worst_residual': max(r['residual'] for r in rows),
        'worst_cond_est': max(r['cond_est'] for r in rows),
    }
    if compute_svd_reference:
        summary['worst_cos_min'] = min(r['cos_min'] for r in rows)
    print('\nSingle-step summary')
    print_table([summary], list(summary.keys()))
    return rows


def run_streaming_examples(
    n: int = 2048,
    m: int = 128,
    seed: int = 0,
    streaming_steps: int = 8,
    streaming_drift: float = 1e-2,
    compute_svd_reference: bool = True,
    **sp_kwargs
):
    step, _ = make_streaming_power_step(m, **sp_kwargs)
    k = random.PRNGKey(seed + 123)
    k1, k2 = random.split(k)

    experiments = [
        ('drifting_gaussian', gen.make_stream(k1, n, m, streaming_steps, streaming_drift)),
        ('rotating_subspace', gen.make_rotating_stream(k2, n, m, streaming_steps, angle=0.02)),
    ]

    all_rows = []
    for name, Ms in experiments:
        V = jnp.eye(m, dtype=jnp.float32)
        rows = []
        for t, M in enumerate(Ms):
            # Warm state update.
            _ = step(M, V)
            jax.block_until_ready(_)
            V, ms = _time_call(step, M, V)
            row = {
                'experiment': name,
                't': t,
                'ms': ms,
                'finite': bool(finite_ok(V)),
                'orth_fro': float(orthogonality_fro(V)),
                'residual': float(subspace_residual(M, V)),
            }
            if compute_svd_reference:
                cmin, cmean, cmax = principal_angle_cosines(M, V)
                row['cos_min'] = float(cmin)
                row['cos_mean'] = float(cmean)
                row['cos_max'] = float(cmax)
            rows.append(row)
            all_rows.append(row)

        cols = ['experiment', 't', 'ms', 'finite', 'orth_fro', 'residual']
        if compute_svd_reference:
            cols += ['cos_min', 'cos_mean', 'cos_max']
        print(f'\nStreaming example: {name}')
        print_table(rows, cols)

    summary_rows = []
    for name, _ in experiments:
        rows = [r for r in all_rows if r['experiment'] == name]
        summary = {
            'experiment': name,
            'steps': len(rows),
            'mean_ms': sum(r['ms'] for r in rows) / len(rows),
            'worst_orth_fro': max(r['orth_fro'] for r in rows),
            'worst_residual': max(r['residual'] for r in rows),
        }
        if compute_svd_reference:
            summary['worst_cos_min'] = min(r['cos_min'] for r in rows)
        summary_rows.append(summary)

    print('\nStreaming summary')
    print_table(summary_rows, list(summary_rows[0].keys()))
    return all_rows


def run_bad_init_recovery(
    n: int = 2048,
    m: int = 128,
    seed: int = 0,
    streaming_steps: int = 8,
    compute_svd_reference: bool = True,
    **sp_kwargs
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
            't': t,
            'ms': ms,
            'finite': bool(finite_ok(V)),
            'orth_fro': float(orthogonality_fro(V)),
            'residual': float(subspace_residual(M, V)),
        }
        if compute_svd_reference:
            cmin, cmean, cmax = principal_angle_cosines(M, V)
            row['cos_min'] = float(cmin)
            row['cos_mean'] = float(cmean)
            row['cos_max'] = float(cmax)
        rows.append(row)

    cols = ['t', 'ms', 'finite', 'orth_fro', 'residual']
    if compute_svd_reference:
        cols += ['cos_min', 'cos_mean', 'cos_max']
    print('\nRecovery from nearly-collinear V0')
    print_table(rows, cols)
    return rows


def run_all(
    n: int = 2048,
    m: int = 128,
    seed: int = 0,
    streaming_steps: int = 8,
    streaming_drift: float = 1e-2,
    compute_svd_reference: bool = True,
    **sp_kwargs
):
    print('Streaming power iteration, fp32 only')
    for k, v in sp_kwargs.items():
        print(f'{k:<12} = {v}')

    run_suite(n=n, m=m, seed=seed, compute_svd_reference=compute_svd_reference, **sp_kwargs)
    run_streaming_examples(
        n=n, m=m, seed=seed,
        streaming_steps=streaming_steps,
        streaming_drift=streaming_drift,
        compute_svd_reference=compute_svd_reference,
        **sp_kwargs
    )
    run_bad_init_recovery(
        n=n, m=m, seed=seed,
        streaming_steps=streaming_steps,
        compute_svd_reference=compute_svd_reference,
        **sp_kwargs
    )
