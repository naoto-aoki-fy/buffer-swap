#!/usr/bin/env python3
"""Integration checks; require the built binary and two P2P-capable CUDA GPUs.
Verify NVLink topology separately before interpreting timings as NVLink results.
"""
import csv
import io
import re
import subprocess
import sys

binary = sys.argv[1] if len(sys.argv) > 1 else "./bin/uva_swap_bench"


def run(args):
    return subprocess.run([binary, *args], text=True, capture_output=True, check=False)


help_result = run(["--help"])
assert help_result.returncode == 0 and not help_result.stdout
assert "Usage:" in help_result.stderr
for args in (["-e", "3"], ["-a", "7"], ["-t", "0"], ["-n", "0"],
             ["--unknown"], ["-a"], ["--validate"],
             ["-m", "0"], ["-m"], ["--grid-stride=1"]):
    result = run(args)
    assert result.returncode != 0 and not result.stdout and result.stderr, args


def check_swap(element, elements, grid_stride, multiplier=4, block=256,
               memcpy_async=False):
    size = element * elements
    args = ["-a", str(size), "-e", str(element), "-t", str(block),
            "-n", "3", "--validate", "-m", str(multiplier)]
    if grid_stride:
        args.append("--grid-stride")
    if memcpy_async:
        args.append("--memcpy-async")
    result = run(args)
    assert result.returncode == 0, result.stderr
    if memcpy_async:
        assert "copy=cudaMemcpyAsync" in result.stderr, result.stderr
        sms = 0
        expected_grid = 0
    else:
        mode = "grid-stride" if grid_stride else "reference"
        config = re.search(rf"kernel={mode}, SMs=(\d+), sm_multiplier=(\d+)", result.stderr)
        assert config, result.stderr
        sms = int(config[1])
        assert int(config[2]) == multiplier
        expected_grid = (min((elements + block - 1) // block, sms * multiplier)
                         if grid_stride else elements // block + 1)
    reader = csv.DictReader(io.StringIO(result.stdout))
    assert reader.fieldnames == ["iteration", "buffer_a_bytes", "element_bytes",
                                 "block_size", "grid_size", "elapsed_ms"]
    rows = list(reader)
    assert len(rows) == 3, result.stdout
    for iteration, row in enumerate(rows):
        assert int(row["iteration"]) == iteration
        assert int(row["buffer_a_bytes"]) == size
        assert int(row["element_bytes"]) == element
        assert int(row["block_size"]) == block
        assert int(row["grid_size"]) == expected_grid
        assert float(row["elapsed_ms"]) >= 0
    assert result.stderr.count("validation passed after") == 4
    return sms


# Compare every byte on both GPUs after warm-up and three timed swaps.
# Exercise sub-word RNG tails, exact-block grids, and partial final blocks.
for grid_stride in (False, True):
    for element in (1, 2, 4, 8):
        for elements in (1, 3, 256, 257):
            sms = check_swap(element, elements, grid_stride)

# Straddle the actual device's grid cap and require several stride iterations.
for multiplier in (1, 4):
    stride = sms * multiplier * 32
    for element in (1, 2, 4, 8):
        for elements in (stride - 1, stride, stride + 1, stride * 3 + 7):
            check_swap(element, elements, True, multiplier, block=32)
for element in (1, 2, 4, 8):
    for elements in (1, 3, 256, 257, 131073):
        check_swap(element, elements, False, memcpy_async=True)
# Kernel hardware limits and grid-stride settings are unused in copy mode.
check_swap(1, 1003, True, block=1048576, memcpy_async=True)
print("UVA P2P runtime checks passed")
