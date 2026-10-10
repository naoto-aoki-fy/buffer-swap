#!/usr/bin/env python3
"""Integration checks; requires a built benchmark and two CUDA P2P GPUs."""
import csv
import io
import subprocess
import sys

binary = sys.argv[1] if len(sys.argv) > 1 else "./bin/gpu_comm_bench"


def run(*args):
    return subprocess.run([binary, *map(str, args)], capture_output=True, text=True,
                          timeout=120)


help_result = run("--help")
assert help_result.returncode == 0, help_result.stderr
assert help_result.stdout == "" and "Usage:" in help_result.stderr
for args in [("-a", "1003"), ("-b", "0"), ("-g", "2147483648"), ("-n", "0"),
             ("--b-size", "8"), ("--skip-copy-back",)]:
    result = run(*args)
    assert result.returncode != 0 and not result.stdout and result.stderr, result

# Both final parities with one warm-up, including a grid-stride tail, a single
# thread processing the entire buffer, and more launched threads than elements.
for size, block, grid in [(8, 256, 2), (1000, 3, 7), (4096, 1, 1), (1048576, 256, 2)]:
    for iterations in [1, 2, 3, 4]:
        result = run("-a", size, "-b", block, "-g", grid,
                     "-n", iterations, "--validate")
        assert result.returncode == 0, result.stderr
        assert "validation passed on both GPUs" in result.stderr, result.stderr
        assert "NVLink UNVERIFIED" in result.stderr, result.stderr
        rows = list(csv.reader(io.StringIO(result.stdout)))
        assert rows[0] == ["iteration", "buffer_a_bytes", "block_size", "grid_size", "elapsed_ms"]
        assert len(rows) == iterations + 1, rows
        for iteration, row in enumerate(rows[1:]):
            assert len(row) == 5 and list(map(int, row[:4])) == [iteration, size, block, grid], row
            assert 0 <= float(row[4]) < float("inf"), row

# Exceed the device's block limit without a large allocation; should fail before CSV.
result = run("-a", "8", "-b", "2147483647")
assert result.returncode != 0 and not result.stdout and "limit" in result.stderr, result
print("GPU validation and CSV tests passed; verify NVLink topology separately")
