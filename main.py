import sys
import os
import argparse

# Ensure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stream_svd.bench.runners import run_hard_bench

def main():
    parser = argparse.ArgumentParser(description='Run Harder Test Suite for Stream SVD')
    parser.add_argument('--n', type=int, default=16384, help='Number of rows')
    parser.add_argument('--m', type=int, default=4096, help='Number of columns')
    parser.add_argument('--steps', type=int, default=10, help='Streaming steps')
    parser.add_argument('--power_shift', type=float, default=1e-4, help='Power shift lambda')
    parser.add_argument('--jitter1', type=float, default=1e-5, help='Jitter 1')
    parser.add_argument('--jitter2', type=float, default=1e-6, help='Jitter 2')
    parser.add_argument('--ref', action='store_true', help='Compute SVD reference (slow!)')
    
    args = parser.parse_args()
    
    run_hard_bench(
        n=args.n,
        m=args.m,
        streaming_steps=args.steps,
        compute_svd_reference=args.ref,
        power_shift=args.power_shift,
        jitter1=args.jitter1,
        jitter2=args.jitter2
    )

if __name__ == '__main__':
    main()
