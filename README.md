# GPU communication benchmark

`bin/gpu_comm_bench` is a two-rank GPU communication benchmark for separate
compute nodes connected by InfiniBand. Each rank owns a main GPU buffer A and a
smaller receive buffer B. Consecutive chunks of A are sent to the peer using
grouped `ncclSend`/`ncclRecv`; received chunks are copied from B into A on the
same CUDA stream. The final chunk can be smaller than B. Only these two data
buffers are allocated, using `cudaMalloc` and initialized with cuRAND.

## Build

Use the existing Makefile, `nvcc`,
[`atlc`](https://github.com/naoto-aoki-fy/atlc), and
[`nvccoptions`](https://github.com/naoto-aoki-fy/nvccoptions) workflow. Requirements
are a 64-bit Linux system, CUDA toolkit, MPI supporting MPI-3 shared-node
communicators, and NCCL **2.10 or newer** (for forced IB network selection).
Error-checking macros from `atlc` check CUDA, cuRAND, MPI and NCCL calls.

Load your site's CUDA, NCCL and MPI modules and set up `atlc`'s include path.
Obtain the vendor/compiler options with `nvccoptions` as in the original build,
and put them in the automatically included `config.mk`. For example, replacing
all paths and architecture values with those for your installation:

```make
CFLAGS_VENDOR = -I/path/to/atlc/include -I/path/to/mpi/include -I/path/to/nccl/include
LDFLAGS_VENDOR = -L/path/to/mpi/lib -lmpi -L/path/to/nccl/lib
GENCODE_FLAGS = -gencode=arch=compute_80,code=sm_80
```

Use the complete MPI wrapper compile/link flags for your MPI implementation;
for Open MPI these can be inspected with `mpicxx --showme:compile` and
`mpicxx --showme:link`. Include any extra platform libraries supplied by the
wrapper in the vendor flags. Ensure these flags select the same MPI that
`mpirun` uses. Set runtime library paths for MPI, NCCL and CUDA through your
site modules or `LD_LIBRARY_PATH`.

```bash
make
```

The default target builds `bin/gpu_comm_bench` directly with `nvcc`, linking
cuRAND and NCCL plus the MPI flags in `LDFLAGS_VENDOR`. `NVCC`, `INCLUDE`,
`LDLIBS` and `BENCH_BIN` remain overridable. No CMake or generated argument
parser is required. `make clean` removes the benchmark executable.

## Options

| Option | Meaning | Default |
|---|---|---|
| `-a`, `--a-size SIZE` | A size in bytes | `32G` (32 GiB) |
| `-b`, `--b-size SIZE` | B size in bytes | `1G` (1 GiB) |
| `-n`, `--iterations N` | Number of timed full-A exchanges | `128` |
| `--skip-copy-back` | Omit B-to-A device copies; A remains unchanged | Disabled |
| `--validate` | Compare every byte of A after each timed iteration | Disabled |
| `-h`, `--help` | Print help to stderr and exit | |

Sizes are positive decimal integers with optional case-insensitive `K`, `M`
or `G` suffixes, representing powers of 1024. Bare numbers mean bytes. Signs,
fractions, zero, overflow and extra positional arguments are rejected. Sizes
need not be multiples of four or of each other; B may be larger than A.
Iterations must be positive decimal integers. All ranks must use identical
options. `--validate` limits each buffer to 16 MiB to keep host reference
storage bounded. The default buffers require **33 GiB free GPU memory per
rank plus CUDA/cuRAND/NCCL overhead**; reduce the sizes for smaller GPUs.
Allocation errors and insufficient memory abort the MPI job.

## Launch on two nodes

Run exactly two MPI ranks, with **one rank per node**. The program rejects
same-node placement to avoid measuring NVLink or PCIe. Each rank selects CUDA
device ordinal 0 relative to its `CUDA_VISIBLE_DEVICES` setting; let your
scheduler expose the assigned GPU, or set the appropriate visibility per node.
MPI broadcasts the NCCL unique ID. All benchmark payload transfers use NCCL
on GPU buffers; MPI handles setup, barriers and time reductions.

For a scheduler allocation already placing one process on each node:

```bash
mpirun -np 2 ./bin/gpu_comm_bench \
    --a-size 32G \
    --b-size 1G \
    --iterations 128 \
    > results.csv
```

An Open MPI example explicitly placing processes on two allocated nodes is:

```bash
mpirun -np 2 --host node-a,node-b --map-by ppr:1:node \
    -x NCCL_IB_HCA -x NCCL_IB_DISABLE -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
    ./bin/gpu_comm_bench -a 32G -b 1G -n 128 > results.csv 2> diagnostics.log
```

Set the exported variables as described below before using this command.
For Slurm, request two nodes, one task and one GPU per node, for example
`#SBATCH --nodes=2`, `#SBATCH --ntasks-per-node=1`, and
`#SBATCH --gpus-per-node=1`, then use your site's supported MPI launch command
(e.g. `srun --mpi=pmix -n 2 --ntasks-per-node=1 ./bin/gpu_comm_bench ...`).
PMIx/PMI selection, GPU binding, CPU binding, HCA selection and module names
are site-specific; use the cluster's MPI and GPU allocation instructions.
Do not launch outside an allocation on systems that require one.

## Require and verify InfiniBand

The program sets `NCCL_NET=IB` before the first NCCL call, overriding any
inherited setting. This forces NCCL's built-in IB verbs network rather than
allowing TCP socket fallback. It also sets `NCCL_DEBUG_FILE=/dev/stderr`,
overriding inherited log destinations, so NCCL diagnostics cannot pollute CSV
stdout. External network plugins with other names are not supported by this
benchmark. These settings require NCCL 2.10+.

On **both** compute nodes, confirm that the intended HCA/port is active and
its link layer is InfiniBand (for example, with `ibstat` or `ibv_devinfo`).
IB verbs can also operate over RoCE; `NCCL_NET=IB` alone does not prove that
the physical link is InfiniBand. Configure matching, reachable HCAs on the
fabric, for example:

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA='=mlx5_0:1'   # Replace with the actual adapter and port
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
```

Forward these settings to both MPI processes using your launcher's environment
mechanism. HCA names may differ between nodes; use per-node settings where
necessary. IP interfaces selected by `NCCL_SOCKET_IFNAME` are used for
bootstrap connectivity and do not force the GPU data transport to IB.
NCCL and MPI can use bootstrap/control sockets even when payloads use IB.

Inspect `diagnostics.log` from both ranks for network selection and actual
channel connections using `NET/IB`, the intended HCA/port, and GPUDirect RDMA
indications such as `GDRDMA` (wording varies by NCCL version). Reject results
showing data channels using `NET/Socket`, `P2P`, or `SHM`. Seeing an HCA in an
initialization message alone is insufficient; check the actual channel paths.
A forced-IB run should fail if the required IB transport is unavailable;
verify the failure behavior on your installation before collecting results.

For direct GPU-to-GPU RDMA, the system must support GPUDirect RDMA, with the
site's supported `nvidia-peermem` or DMA-BUF setup, GPU/HCA topology and memory
registration configuration. Confirm GDR use in channel logs; IB without GDR
can involve NCCL-internal host staging and is not an intended direct-RDMA
measurement. The application itself never stages benchmark payloads through
host memory. Administrators may require topology-specific NCCL GDR settings;
use their validated configuration rather than assuming every GPU/HCA pair
supports direct RDMA.

See NVIDIA's [NCCL environment variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
and [GPUDirect RDMA troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#gpu-direct)
for network selection, logging, HCA filters and prerequisites.

## Timing and CSV

Allocation, cuRAND initialization, communicator setup and one untimed full-A
warm-up exchange run before timing. The warm-up follows the chosen copy mode;
in normal mode it swaps A once. Each timed iteration synchronizes ranks with
an MPI barrier, then records start and stop CUDA events on the single stream
around the entire chunk loop. The stop event is synchronized before reading
`cudaEventElapsedTime`. MPI reduces the two local event times with `MPI_MAX`
after the stop event; rank 0 reports that maximum in milliseconds.

Normal mode includes NCCL communication and all B-to-A device copies, leaving
A equal to the peer's A from the beginning of that iteration. Successive
iterations alternate the two contents. `--skip-copy-back` includes only NCCL
communication: each chunk is received into B and overwritten by the next,
while A remains unchanged. A full iteration sends A bytes and receives A bytes
per rank in both modes. Timings cover execution on the stream, including
waiting and any gaps in enqueueing; they are not isolated link latency or
MPI wall-clock timings. Diagnostic logging can affect measurements, so use
INFO to verify the transport, then lower verbosity for production runs.

Only rank 0 writes stdout, with one header and one row per timed iteration:

```csv
iteration,buffer_a_bytes,buffer_b_bytes,copy_back,elapsed_ms
0,34359738368,1073741824,1,1234.567
1,34359738368,1073741824,1,1231.234
```

These elapsed values are illustrative. Iteration numbering starts at zero;
`copy_back` is `1` normally and `0` in skip mode. Help, configuration, errors,
validation messages and NCCL logs go to stderr. Redirect stdout directly to
CSV and stderr to a diagnostics file.

## Validation

`--validate` snapshots each rank's post-warm-up A on the host and exchanges
bounded reference snapshots with MPI outside timing. After every timed
iteration it compares all bytes of A to the expected peer snapshot (odd
exchange count) or local snapshot (even exchange count). Skip mode always
compares against the local snapshot. Validation copies and comparisons are
outside the CUDA-event interval and allocate no extra GPU data buffer.
Any mismatch aborts the job; successful checks are reported on stderr.

On an allocated two-node GPU/InfiniBand system, run these examples with the
same placement and NCCL settings as above:

```bash
# Multiple chunks, an odd byte count, and a partial final chunk; test both parities.
mpirun -np 2 --host node-a,node-b --map-by ppr:1:node \
    ./bin/gpu_comm_bench -a 1003 -b 257 -n 3 --validate > validate-copy.csv
mpirun -np 2 --host node-a,node-b --map-by ppr:1:node \
    ./bin/gpu_comm_bench -a 1003 -b 257 -n 3 --validate --skip-copy-back > validate-skip.csv
# Exact chunks, B larger than A, and sub-word cuRAND initialization.
mpirun -np 2 --host node-a,node-b --map-by ppr:1:node \
    ./bin/gpu_comm_bench -a 1024 -b 256 -n 2 --validate > validate-exact.csv
mpirun -np 2 --host node-a,node-b --map-by ppr:1:node \
    ./bin/gpu_comm_bench -a 3 -b 7 -n 2 --validate > validate-tiny.csv
```

Check that each output has the exact CSV header, the requested number of
rows and five fields per row. Also check that one/three ranks, two ranks on
a single node, conflicting rank options and excessive allocation requests
fail with diagnostics on stderr. Test invalid options and `--help` without a
GPU; help should leave stdout empty.

Host-only option tests can run without CUDA or MPI:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude tests/bench_options_test.cpp -o /tmp/bench_options_test
/tmp/bench_options_test
```

They cover defaults, binary suffixes, arbitrary byte sizes, copy mode,
validation bounds, invalid options, missing arguments, unsigned integer
limits, size multiplication overflow and combined allocation overflow.
GPU correctness, timing and transport verification require the actual
CUDA/NCCL/MPI and two-node InfiniBand environment.

## Single-process UVA/P2P buffer swap benchmark

`bin/uva_p2p_bench` is a separate benchmark for **two GPUs in one host**, intended
for NVLink. One host process manages CUDA devices 0 and 1 (relative to
`CUDA_VISIBLE_DEVICES`). It uses neither MPI nor NCCL. The existing two-node
MPI/NCCL benchmark and its default build target retain their behavior.

### Prerequisites and build

Use the same Makefile, `nvcc`, `nvccoptions`, `config.mk`, compiler flags and
`atlc` CUDA/cuRAND checking headers described above. Only CUDA and cuRAND are
needed for this executable; MPI and NCCL installations are unnecessary.
For a CUDA-only installation, a minimal site-specific `config.mk` is:

```make
CFLAGS_VENDOR = -I/path/to/atlc/include
GENCODE_FLAGS = -gencode=arch=compute_80,code=sm_80
# Optional CUDA-only extra library search paths/link options:
UVA_LDFLAGS =
```

```bash
make bin/uva_p2p_bench
```

The new target reuses `NVCC_CFLAGS` but deliberately excludes `LDFLAGS_VENDOR`
and `LDLIBS`, which can contain MPI/NCCL libraries. Its separate `UVA_LDFLAGS`
and `UVA_LDLIBS` (default `-lcurand`) must contain only CUDA-related dependencies.
`UVA_BIN` can override the output path; `make clean` removes both executables.

Both GPUs must support UVA, and GPU 0 must support direct peer access to GPU 1.
The program checks `cudaDeviceCanAccessPeer(0, 1)` and enables that direction
using `cudaDeviceEnablePeerAccess`. Only GPU 0 executes the swap kernel, so
reverse peer access is unnecessary. Unsupported UVA/P2P fails with an error.
Each GPU needs **32 GiB free for A, plus CUDA/cuRAND overhead**, by default.
Initialization allocates no additional application GPU staging or communication
buffer; cuRAND and CUDA may require internal resources. Reduce A for smaller GPUs.

**A successful CUDA P2P check does not prove NVLink connectivity.** Before
collecting results, inspect:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p n
```

In the topology matrix the selected pair should have an `NV#` connection;
the `n` P2P query checks NVLink capability. Match the reported PCI bus IDs in
stderr to the physical GPUs, especially when visibility reorders device ordinals.
PCIe-only P2P can also pass the CUDA check. The executable explicitly reports
NVLink as unverified; do not label such results NVLink measurements without
verifying the pair's topology. These commands describe connectivity, not a
measurement of actual link utilization during a run.

### Execution and options

Launch directly, without a multi-process launcher:

```bash
./bin/uva_p2p_bench \
    --a-size 32G \
    --element-bytes 8 \
    --block-size 256 \
    --iterations 128 \
    > results.csv 2> diagnostics.log
```

| Option | Meaning | Default |
|---|---|---|
| `-a`, `--a-size SIZE` | A bytes **per GPU** | `32G` (32 GiB) |
| `-e`, `--element-bytes N` | Unsigned typed access width: 1, 2, 4 or 8 bytes | `8` |
| `-t`, `--block-size N` | CUDA threads per block | `256` |
| `-n`, `--iterations N` | Timed complete swaps | `128` |
| `--validate` | Compare both buffers after warm-up and every timed swap | Disabled |
| `-h`, `--help` | Print help to stderr without accessing CUDA | |

Sizes use the existing positive decimal and binary K/M/G suffix conventions.
A must be divisible by the element width. Other widths, zero, signs, overflow,
unknown options, missing arguments and positional arguments are rejected.
Block size is checked against GPU 0's thread and block-x limits. The grid is
`floor((A bytes / element bytes) / block size) + 1`, enforcing the strict
inequality `block_size * grid_size * element_bytes > A bytes`, even for exact
multiples. Grid limits and 64-bit coverage overflow are checked before launch.
All index arithmetic is 64-bit; surplus threads are bounds checked.

Each buffer is allocated with `cudaMalloc` and initialized with cuRAND using
different seeds. For 1/2-byte sizes, a cuRAND device kernel fills any trailing
bytes that cannot be generated as whole 32-bit words. GPU 0's kernel performs
one local load, one direct UVA peer load, and two opposite stores per element.
Each valid index belongs to exactly one thread. No second swap kernel runs on
GPU 1, and the exchange uses no copy API or extra GPU buffer.

### Timing, CSV and validation

Both initializations complete before one untimed warm-up swap. Every timed
iteration encloses one complete swap in CUDA start/stop events on GPU 0's
launch stream. The stop event is synchronized before `cudaEventElapsedTime`
and before the next iteration. Allocation, initialization, warm-up, validation
and cleanup are excluded. These times cover a full in-place exchange including
local memory operations and peer reads/writes; they are not isolated NVLink
latency or bandwidth measurements.

Stdout contains only this header and one row per timed iteration:

```csv
iteration,buffer_a_bytes,element_bytes,block_size,grid_size,elapsed_ms
0,34359738368,8,256,16777217,123.456
1,34359738368,8,256,16777217,122.987
```

Times are illustrative and in milliseconds. Diagnostics and validation results
go to stderr. `--validate` requires A <= 16 MiB per GPU, takes original host
snapshots before warm-up, and compares every byte of both buffers after each
swap. Warm-up exchanges the originals; timed iteration 0 restores them,
iteration 1 exchanges them again, and so on. Host copies serve only validation
outside timing. Mismatches cause a nonzero exit status.

Run both host-only test suites without CUDA, MPI or GPUs:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude tests/bench_options_test.cpp -o /tmp/bench_options_test
/tmp/bench_options_test
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude tests/uva_p2p_options_test.cpp -o /tmp/uva_p2p_options_test
/tmp/uva_p2p_options_test
```

The new suite checks defaults, all element widths, parsing errors, divisibility,
validation bounds, strict grid coverage, hardware limits and arithmetic overflow.
On the actual two-GPU system, after verifying NVLink topology, run:

```bash
python3 tests/uva_p2p_runtime_test.py ./bin/uva_p2p_bench
```

This integration test validates both GPUs across repeated swaps for every
supported width, sub-word initialization, exact and partial blocks, and tiny
buffers. It also checks help/errors and the CSV schema and row counts. GPU
correctness, event timing, allocation at 32 GiB, and NVLink transport require
the actual CUDA toolkit and two-GPU hardware; host tests cannot verify them.

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017,
commissioned by the New Energy and Industrial Technology Development
Organization (NEDO).
