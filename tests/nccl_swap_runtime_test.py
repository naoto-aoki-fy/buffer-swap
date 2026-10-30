#!/usr/bin/env python3
"""Requires two CUDA GPUs (on one node or two IB nodes) and a built benchmark."""
import argparse
import csv
import io
import math
import shlex
import subprocess

HEADER = ['iteration', 'buffer_a_bytes', 'buffer_b_bytes', 'copy_back', 'elapsed_ms']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary')
    parser.add_argument('--launcher', default='mpirun -np 2',
                        help='Include site-specific placement and environment forwarding flags')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--stress-iterations', type=int, default=128)
    parser.add_argument('--stress-only', action='store_true')
    args = parser.parse_args()
    if args.stress_iterations < 2:
        parser.error('--stress-iterations must be at least 2')
    cases = 0

    def validate_case(a_size, b_size, copy_back, iterations, pipeline):
        nonlocal cases
        command = shlex.split(args.launcher) + [args.binary, '-a', str(a_size),
                    '-b', str(b_size), '-n', str(iterations), '--validate']
        if not copy_back:
            command.append('--skip-copy-back')
        command.append('--pipeline-copy-back' if pipeline else '--no-pipeline-copy-back')
        result = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
        assert result.returncode == 0, (command, result.stdout, result.stderr)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        assert rows and rows[0] == HEADER and len(rows) == iterations + 1, result.stdout
        for iteration, row in enumerate(rows[1:]):
            assert len(row) == len(HEADER), row
            assert tuple(map(int, row[:-1])) == (iteration, a_size, b_size, int(copy_back)), row
            elapsed = float(row[-1])
            assert math.isfinite(elapsed) and elapsed >= 0, row
        slots = 2 if pipeline and copy_back and b_size >= 2 else 1
        for rank in [0, 1]:
            assert f'rank {rank}: warm-up and all timed swaps validated' in result.stderr, result.stderr
            assert (f'rank {rank}: staging_slots={slots} chunk_bytes={b_size // slots}'
                    in result.stderr), result.stderr
        cases += 1

    for copy_back in [True, False]:
        if not args.stress_only:
            for a_size, b_size in [
                (1, 1), (7, 1),              # Single-byte fallback, including reuse.
                (1, 2), (2, 2), (3, 2),      # One, two, three chunks (slot reuse).
                (3, 7), (4, 7), (7, 7),      # Odd B, B>A, partial final chunks.
                (1024, 256), (1003, 257),    # Exact/partial chunks, odd A and B.
                (1025, 2048),               # B>A but A needs both half-B slots.
            ]:
                for iterations in [2, 3]:
                    for pipeline in [True, False]:
                        validate_case(a_size, b_size, copy_back, iterations, pipeline)
        # Many slot reuses and alternating swap parities, exact and partial tails.
        for a_size, b_size in [(8192, 128), (8193, 129)]:
            for pipeline in [True, False]:
                validate_case(a_size, b_size, copy_back, args.stress_iterations, pipeline)
    print(f'{cases} NCCL validation/CSV cases passed')


if __name__ == '__main__':
    main()
