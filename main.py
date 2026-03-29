import sys
import os

# Set JAX memory flags before any JAX imports
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Optional: helps with driver version mismatch warnings on some WSL versions
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# Ensure the src directory is in the path if running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from stream_svd.bench.runners import run_all


def main():
    run_all(
        power_shift=1e-4,
        jitter1=1e-5,
        jitter2=1e-6,
        diag_floor_mult=10.0,
        n=16384,
        m=4096,
        seed=0,
        streaming_steps=8,
        streaming_drift=1e-2,
        compute_svd_reference=True,
    )


if __name__ == "__main__":
    main()
