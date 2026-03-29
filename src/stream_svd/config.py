from dataclasses import dataclass

@dataclass(frozen=True)
class StreamingPowerConfig:
    power_shift: float = 1e-4
    jitter1: float = 1e-5
    jitter2: float = 1e-6
    diag_floor_mult: float = 10.0


@dataclass(frozen=True)
class SuiteConfig:
    n: int = 2048
    m: int = 128
    seed: int = 0
    streaming_steps: int = 8
    streaming_drift: float = 1e-2
    compute_svd_reference: bool = True
