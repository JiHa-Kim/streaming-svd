import sys
import os

# Ensure the src directory is in the path if running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stream_svd.config import StreamingPowerConfig, SuiteConfig
from stream_svd.bench.runners import run_all

def main():
    sp_cfg = StreamingPowerConfig(
        power_shift=1e-4,
        jitter1=1e-5,
        jitter2=1e-6,
        diag_floor_mult=10.0,
    )
    suite_cfg = SuiteConfig(
        n=2048,
        m=128,
        seed=0,
        streaming_steps=8,
        streaming_drift=1e-2,
        compute_svd_reference=True,
    )

    run_all(sp_cfg, suite_cfg)


if __name__ == '__main__':
    main()
