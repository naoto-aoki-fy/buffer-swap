#!/usr/bin/env python3
"""Requires a two-GPU P2P node or two CUDA/IB nodes and a built executable."""
import argparse
import csv
import io
import math
import os
import shlex
import subprocess

HEADER = ['iteration', 'buffer_a_bytes', 'buffer_b_bytes', 'element_bytes',
          'block_size', 'sm_multiplier', 'grid_blocks', 'chunk_bytes', 'elapsed_ms']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary')
    parser.add_argument('--launcher', default='mpirun -np 2',
                        help='Include site-specific placement and environment forwarding flags')
    parser.add_argument('--sm-multipliers', default='1', help='Use 1,2 if cooperative capacity allows')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault('NVSHMEM_SYMMETRIC_SIZE', '64M')

    def run(options, success=True):
        result = subprocess.run(shlex.split(args.launcher) + [args.binary] + options,
                                env=env, text=True, capture_output=True, timeout=args.timeout)
        assert (result.returncode == 0) == success, result.stderr
        if not success:
            assert result.stderr and not result.stdout, (result.stdout, result.stderr)
        return result

    help_result = run(['--help'])
    assert not help_result.stdout and 'Usage:' in help_result.stderr
    for options in [['-a', '0'], ['-e', '3'], ['-t', '0'], ['-m', '0'], ['-n', '0'],
                    ['--unknown'], ['-a'], ['positional'], ['--validate'],
                    ['-a', '18446744073709551615G'], ['-a', '64G'],
                    ['-a', '8', '-t', '1025'], ['-a', '8', '-m', '18446744073709551615']]:
        run(options, success=False)
    cases = 0
    for multiplier in map(int, args.sm_multipliers.split(',')):
        for width in [1, 2, 4, 8]:
            for threads in [32, 128, 256]:
                chunk = width * threads
                # Tiny, exact chunk, partial chunk, many rounds and partial tail.
                for size in sorted({1, 3, chunk, chunk + 1, 1048579}):
                    result = run(['-a', str(size), '-e', str(width), '-t', str(threads),
                                  '-m', str(multiplier), '-n', '3', '--validate'])
                    rows = list(csv.reader(io.StringIO(result.stdout)))
                    assert rows[0] == HEADER and len(rows) == 4, result.stdout
                    for i, row in enumerate(rows[1:]):
                        assert len(row) == len(HEADER), row
                        values = list(map(int, row[:-1]))
                        iteration, a, b, e, t, m, grid, c = values
                        assert (iteration, a, e, t, m, c) == (i, size, width, threads, multiplier, chunk)
                        assert grid > 0 and grid % multiplier == 0 and b == chunk * grid
                        elapsed = float(row[-1])
                        assert math.isfinite(elapsed) and elapsed >= 0
                    assert 'rank 0: warm-up and all timed swaps validated' in result.stderr
                    assert 'rank 1: warm-up and all timed swaps validated' in result.stderr
                    if 'NVSHMEM_IB_ENABLE_IBGDA' in env:
                        for rank in [0, 1]:
                            assert (f'rank {rank} NVSHMEM_IB_ENABLE_IBGDA='
                                    f'{env["NVSHMEM_IB_ENABLE_IBGDA"]} (verify path in runtime logs)'
                                    in result.stderr), result.stderr
                    cases += 1
    print(f'{cases} NVSHMEM validation/CSV cases passed')


if __name__ == '__main__':
    main()
