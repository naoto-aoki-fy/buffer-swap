# CUDA GPU buffer exchange benchmark

`bin/gpu_comm_bench` runs one process managing CUDA devices 0 and 1 on a
single computer. GPU 0 launches a kernel that exchanges its buffer A with
GPU 1's buffer A using direct CUDA P2P loads and stores. Each thread reads
both 64-bit values before writing either, then advances by the grid's total
thread count. Indices and strides are 64-bit. Each element has one owner;
no atomics or inter-thread barriers are needed. A system memory fence at the
end of each thread publishes peer writes before kernel completion.

One `cudaMalloc` buffer per GPU is initialized with cuRAND using independent
seeds. There is no receive buffer, host-staged payload, MPI, NCCL, or
InfiniBand communication. The default is **32 GiB (34,359,738,368 bytes) per
GPU**, requiring that much free memory plus CUDA/cuRAND overhead.

## Build

Keep the existing Makefile, `nvcc`,
[`atlc`](https://github.com/naoto-aoki-fy/atlc), and
[`nvccoptions`](https://github.com/naoto-aoki-fy/nvccoptions) workflow.
Requirements are 64-bit Linux, a CUDA toolkit supporting your GPUs, and the
`atlc` headers. The CUDA and cuRAND calls use `ATLC_CHECK_CUDA` and
`ATLC_CHECK_CURAND`. The application links CUDA runtime and cuRAND only.

Generate compiler options with your existing `nvccoptions` installation:

```bash
cd /path/to/nvccoptions
./get_nvccopts.sh
```

The helper discovers NVIDIA HPC SDK wrapper options and caches them in
`nvccoptions.txt`. Its discovery step uses `mpicxx`; that is a tooling
requirement of the existing helper, not a runtime or link dependency of this
benchmark. Review the generated options, retain the CUDA/platform compiler
and linker flags, and remove MPI/NCCL include paths, library paths, libraries
(including `-lmpi` and `-lnccl`), and wrapper-specific MPI flags. Put the
remaining options into this repository's `config.mk`, which the Makefile
continues to include automatically. For example, adapting paths and GPU
architecture to your installation:

```make
CFLAGS_VENDOR = -I/path/to/atlc/include
LDFLAGS_VENDOR = -L/path/to/cuda/lib64
GENCODE_FLAGS = -gencode=arch=compute_80,code=sm_80
```

Then, from this repository:

```bash
make
```

The executable remains `bin/gpu_comm_bench`. `NVCC`, `INCLUDE`, `LDLIBS`, and
`BENCH_BIN` remain overridable; default `LDLIBS` is `-lcurand`. Set CUDA runtime
library paths through your site modules or `LD_LIBRARY_PATH`.
`make clean` removes the executable. No additional build system is used.

## Hardware and NVLink verification

Expose two GPUs on the same machine. The program selects visible CUDA
ordinals 0 and 1, requires unified virtual addressing, checks
`cudaDeviceCanAccessPeer(0, 1)`, and enables GPU 0 access to GPU 1 with
`cudaDeviceEnablePeerAccess(1, 0)`. Only GPU 0 runs the exchange kernel, so
reverse peer access is unnecessary. Both GPUs' initialization streams
finish before any exchange starts. Launch sizes are checked against GPU 0's
device and kernel limits; insufficient free device memory fails with an
error on stderr.

**CUDA P2P accessibility does not prove NVLink transport.** Before collecting
NVLink results, run:

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv
nvidia-smi topo -m
```

Match the PCI bus IDs printed by the benchmark on stderr to the physical GPUs
in the topology matrix. The intersection for those GPUs should show `NV#`
(e.g. `NV2`), indicating a bonded set of NVLinks. PCIe topology labels such as
`PIX`, `PXB`, `PHB`, or `SYS` are not NVLink confirmation. See NVIDIA's
[`nvidia-smi` topology reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
`CUDA_VISIBLE_DEVICES` can reorder visible CUDA ordinals, so do not assume
CUDA device 0 is physical GPU index 0.

This program does not automatically detect the physical transport. Every
run explicitly logs **NVLink UNVERIFIED** and warns that P2P may use PCIe.
Treat results as generic CUDA P2P measurements until the selected pair's
NVLink topology is verified, and keep that topology output alongside the CSV
and diagnostics. A PCIe P2P run is not an NVLink result. No host fallback is
implemented; lack of CUDA P2P causes failure. CUDA's
[peer memory access documentation](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)
describes the hardware-dependent support.

## Options and execution

| Option | Meaning | Default |
|---|---|---|
| `-a`, `--a-size SIZE` | Buffer A bytes on each GPU; multiple of 8 | `32G` |
| `-b`, `--block-size N` | CUDA threads per block | `256` |
| `-g`, `--grid-size N` | CUDA number of blocks | `1024` |
| `-n`, `--iterations N` | Number of timed full exchanges | `128` |
| `--validate` | Compare every byte of both buffers after each exchange | Disabled |
| `-h`, `--help` | Print help on stderr and exit before CUDA setup | |

Sizes use positive decimal integers with optional case-insensitive `K`, `M`,
or `G` binary suffixes; bare numbers mean bytes. Signs, whitespace, fractions,
zero, overflow, unaligned sizes, missing arguments, and positional arguments
are rejected. Launch sizes and iteration count must be positive decimal
integers. Launch dimensions must fit CUDA's integer limits and the selected
GPU/kernel limits. Validation requires A <= 16 MiB on each GPU to bound host
reference memory. `-b` now means block size; `--b-size` and
`--skip-copy-back` are removed.

Run directly, without an MPI launcher:

```bash
./bin/gpu_comm_bench -a 32G -b 256 -g 1024 -n 128 > results.csv 2> diagnostics.log
# Smaller allocations, selecting two physical GPUs if necessary:
CUDA_VISIBLE_DEVICES=0,1 ./bin/gpu_comm_bench -a 1G -n 16 > small.csv 2> small.log
```

## Timing and CSV

Allocation, initialization, reference snapshots, and one complete untimed
warm-up exchange precede timing. Each timed iteration records a start CUDA
event on GPU 0's exchange stream, launches the entire swap kernel, records a
stop event on that same stream, synchronizes the stop event, and obtains
milliseconds using `cudaEventElapsedTime`. The measured execution includes
remote P2P memory access and the final system memory fence. Validation runs
after synchronization and outside the measured interval. Every iteration
performs a complete exchange, so contents alternate, including the warm-up.

Stdout contains only this header and one row per timed iteration:

```csv
iteration,buffer_a_bytes,block_size,grid_size,elapsed_ms
0,34359738368,256,1024,1234.567000
1,34359738368,256,1024,1231.234000
```

Times above are illustrative. Iterations start at zero. Help, configuration,
GPU identities, topology warnings, validation, and errors go to stderr.
Timings describe this load/store exchange kernel, not isolated NVLink latency
or peak link bandwidth.

## Tests and validation

Host-only parser tests require no CUDA installation:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude tests/bench_options_test.cpp -o /tmp/bench_options_test
/tmp/bench_options_test
```

They test defaults, short/long options, binary suffixes, integer/size overflow,
alignment, validation bounds, removed options, and device launch limit checks.
Help and invalid options can also run without GPU hardware once the CUDA
executable has been built (the shared CUDA libraries must still be present).

On a two-GPU CUDA P2P system, build and run the integration suite:

```bash
python3 tests/gpu_bench_test.py ./bin/gpu_comm_bench
```

It checks CLI behavior, stderr diagnostics, exact CSV fields and row counts,
nonnegative finite elapsed times, and full
correctness with one through four timed exchanges. Sizes include one 64-bit
element, a grid-stride tail, a single-thread loop, and a larger small buffer.
Validation snapshots both initial buffers, checks the warm-up against the
opposite snapshots, then checks both buffers after every timed exchange.
With one warm-up, an odd number of timed exchanges restores the initial
contents; an even number leaves them swapped. Validation failure returns a
nonzero status and reports the GPU and exchange on stderr. Host snapshots
are references only and never carry the actual exchange payload.

Individual parity checks:

```bash
./bin/gpu_comm_bench -a 1000 -b 3 -g 7 -n 3 --validate > odd.csv 2> odd.log
./bin/gpu_comm_bench -a 1000 -b 3 -g 7 -n 4 --validate > even.csv 2> even.log
```

The integration suite requires real GPUs; it does not prove NVLink transport.
Also verify NVLink topology as above and run the default 32 GiB configuration
on a pair with sufficient free memory before collecting production results.
GPU correctness, launch behavior, P2P visibility, and event timing cannot be
verified by the host-only parser suite.

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017,
commissioned by the New Energy and Industrial Technology Development
Organization (NEDO).
