import jax.numpy as jnp
from jax import random
from . import generators as gen

def build_hard_suite(n: int = 16384, m: int = 4096, seed: int = 42):
    k = random.PRNGKey(seed)
    keys = random.split(k, 10)
    
    suite = [
        {
            'name': f'large_gaussian_{n}x{m}',
            'M': gen.make_gaussian(keys[0], n, m),
            'V0': jnp.eye(m, dtype=jnp.float32)
        },
        {
            'name': f'large_ill_conditioned_1e7_{n}x{m}',
            'M': gen.make_controlled_spectrum(keys[1], n, m, cond=1e7),
            'V0': jnp.eye(m, dtype=jnp.float32)
        },
        {
            'name': f'large_staircase_{n}x{m}',
            'M': gen.make_staircase_spectrum(keys[2], n, m, blocks=8, gap=0.5),
            'V0': jnp.eye(m, dtype=jnp.float32)
        },
        {
            'name': f'large_cliff_128_{n}x{m}',
            'M': gen.make_cliff_spectrum(keys[3], n, m, cliff_rank=128, cliff_ratio=1e-5),
            'V0': jnp.eye(m, dtype=jnp.float32)
        },
        {
            'name': f'large_low_rank_noise_{n}x{m}',
            'M': gen.make_near_rank_deficient(keys[4], n, m, rank=128, noise=1e-3),
            'V0': jnp.eye(m, dtype=jnp.float32)
        },
        {
            'name': f'large_sparseish_1pct_{n}x{m}',
            'M': gen.make_sparseish(keys[5], n, m, density=0.01),
            'V0': jnp.eye(m, dtype=jnp.float32)
        }
    ]
    return suite
