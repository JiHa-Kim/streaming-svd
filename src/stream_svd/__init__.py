from .core import make_streaming_power_step

import jax

__all__ = ['make_streaming_power_step']

# Global JAX configuration
# "highest" here means higher-precision fp32 matmul behavior on supported hardware, not fp64.
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_default_matmul_precision', 'highest')
