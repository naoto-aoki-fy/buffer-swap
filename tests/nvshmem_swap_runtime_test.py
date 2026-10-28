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
    parser.add_argument('--stress-iterations', type=int, default=128,
                        help='Swaps in each repeated staging-reuse case (at least 2)')
    parser.add_argument('--stress-only', action='store_true',
                        help='Run only the long staging-reuse cases')
    args = parser.parse_args()
    if args.stress_iterations < 2:
        parser.error('--stress-iterations must be at least 2')
    env = os.environ.copy()
    env.setdefault('NVSHMEM_SYMMETRIC_SIZE', '64M')

    def run(options, success=True):
        result = subprocess.run(shlex.split(args.launcher) + [args.binary] + options,
                                env=env, text=True, capture_output=True, timeout=args.timeout)
        assert (result.returncode == 0) == success, result.stderr
        if not success:
            assert result.stderr and not result.stdout, (result.stdout, result.stderr)
        return result

    if not args.stress_only:
        help_result = run(['--help'])
        assert not help_result.stdout and 'Usage:' in help_result.stderr
        for options in [['--method'], ['--method', '0'], ['--method', '4'],
                        ['-a', '0'], ['-c', '0'], ['-c'], ['-c', '3', '-e', '2'], ['-e', '3'],
                        ['-t', '0'], ['-m', '0'], ['-n', '0'],
                        ['--unknown'], ['-a'], ['positional'], ['--validate'],
                        ['-a', '18446744073709551615G'], ['-a', '64G'],
                        ['-a', '8', '-t', '1025'], ['-a', '8', '-m', '18446744073709551615']]:
            run(options, success=False)
    cases = 0

    def validate_case(size, width, threads, multiplier, chunk=None, method=None, iterations=3):
        nonlocal cases
        options = ['-a', str(size), '-e', str(width), '-t', str(threads),
                   '-m', str(multiplier), '-n', str(iterations), '--validate']
        if method is not None:
            options += ['--method', str(method)]
        if chunk is not None:
            options += ['--chunk-size', str(chunk)]
        expected_chunk = 262144 if chunk is None else chunk
        result = run(options)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        assert rows[0] == HEADER and len(rows) == iterations + 1, result.stdout
        for i, row in enumerate(rows[1:]):
            assert len(row) == len(HEADER), row
            values = list(map(int, row[:-1]))
            iteration, a, b, e, t, m, grid, c = values
            assert (iteration, a, e, t, m, c) == (i, size, width, threads, multiplier, expected_chunk)
            assert grid > 0 and grid % multiplier == 0 and b == expected_chunk * grid
            elapsed = float(row[-1])
            assert math.isfinite(elapsed) and elapsed >= 0
        assert 'rank 0: warm-up and all timed swaps validated' in result.stderr
        assert 'rank 1: warm-up and all timed swaps validated' in result.stderr
        if 'NVSHMEM_IB_ENABLE_IBGDA' in env:
            for rank in [0, 1]:
                assert (f'rank {rank} NVSHMEM_IB_ENABLE_IBGDA='
                        f'{env["NVSHMEM_IB_ENABLE_IBGDA"]} (verify path in runtime logs)'
                        in result.stderr), result.stderr
        assert result.stderr.count(f'method={method or 1} validate=1') == 2, result.stderr
        cases += 1
        return grid

    if not args.stress_only:
        validate_case(3, 8, 32, 1)  # Omitting --method retains the original path.
        for method in [1, 2, 3]:
            for multiplier in map(int, args.sm_multipliers.split(',')):
                for width in [1, 2, 4, 8]:
                    for threads in [32, 128, 256]:
                        # Default chunk is independent of element width and block size.
                        for size in [1, 3, 262144, 262145, 1048579]:
                            validate_case(size, width, threads, multiplier, method=method)
                        # Underfilled blocks, one pass, multiple passes with a partial
                        # last pass, and an independent larger chunk. A has a byte tail.
                        for chunk in sorted({width, width * threads,
                                             width * (3 * threads + 1), 32768}):
                            size = 1048579 if chunk == width * threads else 3 * chunk + 3
                            validate_case(size, width, threads, multiplier, chunk, method=method)
                    # A tail may need several passes when the block has only one thread.
                    validate_case(32771, width, 1, multiplier, 32768, method=method)
    for method in [1, 2, 3]:
        for multiplier in map(int, args.sm_multipliers.split(',')):
            # Discover the actual grid so every block reuses B many times, even
            # on large GPUs. Exercise an exact final round and a one-byte tail
            # with unequal round counts across blocks, over both swap parities.
            grid = validate_case(1, 8, 32, multiplier, 8, method)
            stride = grid * 8
            for size in [17 * stride, 17 * stride + 1]:
                assert size <= 16 * 1024 * 1024, 'stress case exceeds validation limit'
                validate_case(size, 8, 32, multiplier, 8, method,
                              iterations=args.stress_iterations)
    print(f'{cases} NVSHMEM validation/CSV cases passed')


if __name__ == '__main__':
    main()
