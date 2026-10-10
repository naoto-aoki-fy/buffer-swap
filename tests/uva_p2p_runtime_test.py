#!/usr/bin/env python3
"""Integration checks; require the built binary and two P2P-capable CUDA GPUs.
Verify NVLink topology separately before interpreting timings as NVLink results.
"""
import csv
import io
import subprocess
import sys

binary = sys.argv[1] if len(sys.argv) > 1 else "./bin/uva_swap_bench"


def run(args):
    return subprocess.run([binary, *args], text=True, capture_output=True, check=False)


help_result = run(["--help"])
assert help_result.returncode == 0 and not help_result.stdout
assert "Usage:" in help_result.stderr
for args in (["-e", "3"], ["-a", "7"], ["-t", "0"], ["-n", "0"],
             ["--unknown"], ["-a"], ["--validate"]):
    result = run(args)
    assert result.returncode != 0 and not result.stdout and result.stderr, args

# Compare every byte on both GPUs after warm-up and three timed swaps.
# Exercise sub-word RNG tails, exact-block grids, and partial final blocks.
for element in (1, 2, 4, 8):
    for elements in (1, 3, 256, 257):
        size = element * elements
        result = run(["-a", str(size), "-e", str(element), "-t", "256",
                      "-n", "3", "--validate"])
        assert result.returncode == 0, result.stderr
        reader = csv.DictReader(io.StringIO(result.stdout))
        assert reader.fieldnames == ["iteration", "buffer_a_bytes", "element_bytes",
                                    "block_size", "grid_size", "elapsed_ms"]
        rows = list(reader)
        assert len(rows) == 3, result.stdout
        for iteration, row in enumerate(rows):
            assert int(row["iteration"]) == iteration
            assert int(row["buffer_a_bytes"]) == size
            assert int(row["element_bytes"]) == element
            assert int(row["block_size"]) == 256
            assert int(row["grid_size"]) == elements // 256 + 1
            assert float(row["elapsed_ms"]) >= 0
        assert result.stderr.count("validation passed after") == 4
print("UVA P2P runtime checks passed")
