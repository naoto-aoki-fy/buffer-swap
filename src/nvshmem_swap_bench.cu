#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <atlc/check_cuda.hpp>
#include <atlc/check_curand.hpp>
#include <atlc/check_mpi.hpp>
#include "nvshmem_swap_options.hpp"

static void check_nvshmem(int status, const char* operation) {
    if (status) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}
static void usage() {
    std::fprintf(stderr,
        "Usage: nvshmem_swap_bench [options] (two ranks on separate nodes)\n"
        "  -a, --a-size SIZE       A bytes per GPU (default 32G)\n"
        "  -e, --element-bytes N   1, 2, 4 or 8 (default 8)\n"
        "  -t, --block-size N      Threads per block (default 256)\n"
        "  -m, --sm-multiplier N   Blocks per SM (default 1)\n"
        "  -n, --iterations N      Timed full swaps (default 128)\n"
        "      --validate          Verify warm-up and every swap; A <= 16M\n"
        "  -h, --help              Show help on stderr\n"
        "SIZE: positive integer with binary K/M/G suffix. Set NVSHMEM_SYMMETRIC_SIZE\n"
        "large enough for A+B+signals+1 MiB; default A needs e.g. 33G.\n");
}

__global__ void random_tail(unsigned char* tail, size_t bytes, unsigned long long seed) {
    curandState state;
    curand_init(seed, 0, 0, &state);
    const unsigned int value = curand(&state);
    for (size_t i = 0; i < bytes; ++i) tail[i] = static_cast<unsigned char>(value >> (8 * i));
}

__global__ void swap_buffer(unsigned char* a, unsigned char* b, uint64_t* signals,
                            uint64_t a_bytes, uint64_t chunk_bytes,
                            uint64_t element_bytes, uint64_t generation, int peer) {
    const uint64_t stride = uint64_t(gridDim.x) * chunk_bytes;
    unsigned char* staging = b + uint64_t(blockIdx.x) * chunk_bytes;
    uint64_t round = 0;
    for (uint64_t offset = uint64_t(blockIdx.x) * chunk_bytes; offset < a_bytes; ++round) {
        const uint64_t remaining = a_bytes - offset;
        const uint64_t bytes = remaining < chunk_bytes ? remaining : chunk_bytes;
        // Blocking block GET completes into local B. Every thread calls it
        // with identical arguments, even for a sub-element final chunk.
        nvshmemx_getmem_block(staging, a + offset, bytes, peer);
        __syncthreads();
        if (threadIdx.x == 0) {
            nvshmem_quiet();
            nvshmemx_signal_op(signals + blockIdx.x, generation + round + 1,
                               NVSHMEM_SIGNAL_SET, peer);
            // Drain this SET before another generation can be issued: no
            // delayed old SET can replace a newer notification.
            nvshmem_quiet();
            nvshmem_signal_wait_until(signals + blockIdx.x, NVSHMEM_CMP_GE,
                                      generation + round + 1);
        }
        __syncthreads();
        const uint64_t first = uint64_t(threadIdx.x) * element_bytes;
        for (uint64_t j = 0; j < element_bytes && first + j < bytes; ++j)
            a[offset + first + j] = staging[first + j];
        __syncthreads(); // Complete the copy before reusing this block's B.
        if (stride >= remaining) break; // Avoid overflowing a 64-bit offset.
        offset += stride;
    }
    // Publish every thread's local A stores to the NIC before kernel completion
    // and the host MPI barrier allow the next iteration's remote GETs.
    __threadfence_system();
}

static void matching(const uint64_t* values, int count) {
    std::vector<uint64_t> lo(count), hi(count);
    ATLC_CHECK_MPI(MPI_Allreduce, values, lo.data(), count, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    ATLC_CHECK_MPI(MPI_Allreduce, values, hi.data(), count, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    if (lo != hi) throw std::runtime_error("ranks must have matching options, SM counts and block counts");
}
static void environment(const char* key, const char* value) {
    if (setenv(key, value, 1)) throw std::runtime_error(std::string("cannot set ") + key);
}

int main(int argc, char** argv) {
    bool mpi_active = false;
    int rank = -1;
    try {
        ATLC_CHECK_MPI(MPI_Init, &argc, &argv);
        mpi_active = true;
        ATLC_CHECK_MPI(MPI_Comm_set_errhandler, MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        ATLC_CHECK_MPI(MPI_Comm_rank, MPI_COMM_WORLD, &rank);
        const auto o = nvshmem_swap::parse_options(argc, argv);
        const uint64_t config[] = {o.a_size, o.element_bytes, o.block_size, o.sm_multiplier,
                                   o.iterations, uint64_t(o.validate), uint64_t(o.help)};
        matching(config, 7);
        if (o.help) {
            if (!rank) usage();
            ATLC_CHECK_MPI(MPI_Finalize);
            mpi_active = false;
            return 0;
        }
        int ranks;
        ATLC_CHECK_MPI(MPI_Comm_size, MPI_COMM_WORLD, &ranks);
        if (ranks != 2) throw std::runtime_error("exactly two MPI ranks are required");
        MPI_Comm local;
        ATLC_CHECK_MPI(MPI_Comm_split_type, MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                       rank, MPI_INFO_NULL, &local);
        int local_size;
        ATLC_CHECK_MPI(MPI_Comm_size, local, &local_size);
        ATLC_CHECK_MPI(MPI_Comm_free, &local);
        if (local_size != 1) throw std::runtime_error("place one rank on each of two separate nodes");
        ATLC_CHECK_CUDA(cudaSetDevice, 0);
        cudaDeviceProp prop;
        ATLC_CHECK_CUDA(cudaGetDeviceProperties, &prop, 0);
        if (!prop.cooperativeLaunch || prop.major < 7)
            throw std::runtime_error("NVSHMEM synchronization requires a Volta-or-newer GPU with cooperative launch");
        const auto l = nvshmem_swap::layout(o, prop.multiProcessorCount,
                                            prop.maxThreadsPerBlock, prop.maxThreadsDim[0], prop.maxGridSize[0]);
        const char* heap_setting = std::getenv("NVSHMEM_SYMMETRIC_SIZE");
        const uint64_t heap_bytes = bench::parse_size(heap_setting ? heap_setting : "1G");
        const uint64_t hardware[] = {uint64_t(prop.multiProcessorCount), l.grid_blocks, heap_bytes};
        matching(hardware, 3);
        size_t free_bytes, total_bytes;
        ATLC_CHECK_CUDA(cudaMemGetInfo, &free_bytes, &total_bytes);
        nvshmem_swap::check_capacity(l, heap_bytes, free_bytes);
        // A fixed cudaMalloc-backed symmetric heap makes the capacity check
        // meaningful and avoids VMM/registration differences between releases.
        environment("NVSHMEM_SYMMETRIC_SIZE", std::to_string(heap_bytes).c_str());
        environment("NVSHMEM_DISABLE_CUDA_VMM", "1");
        environment("NVSHMEM_REMOTE_TRANSPORT", "ibrc");
        environment("NVSHMEM_IB_ENABLE_IBGDA", "0");
        environment("NVSHMEM_DISABLE_P2P", "1");
        environment("NVSHMEM_DISABLE_MNNVL", "1");
        environment("NVSHMEM_DISABLE_NCCL", "1");
        environment("NVSHMEM_DEBUG_FILE", "/dev/stderr");
        MPI_Comm comm = MPI_COMM_WORLD;
#ifdef NVSHMEMX_INIT_ATTR_INITIALIZER
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
#else
        nvshmemx_init_attr_t attr{};
#endif
        attr.mpi_comm = &comm;
        check_nvshmem(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr), "nvshmemx_init_attr");
        if (nvshmem_n_pes() != 2 || nvshmem_my_pe() != rank)
            throw std::runtime_error("NVSHMEM PE mapping differs from MPI ranks");
        auto* a = static_cast<unsigned char*>(nvshmem_malloc(o.a_size));
        auto* b = static_cast<unsigned char*>(nvshmem_malloc(l.b_bytes));
        auto* signals = static_cast<uint64_t*>(nvshmem_malloc(l.signal_bytes));
        if (!a || !b || !signals)
            throw std::runtime_error("NVSHMEM symmetric allocation failed; increase heap capacity/free GPU memory");
        char hostname[MPI_MAX_PROCESSOR_NAME] = {};
        int hostname_length;
        ATLC_CHECK_MPI(MPI_Get_processor_name, hostname, &hostname_length);
        std::fprintf(stderr, "rank %d host %.*s GPU 0 %s SMs=%d blocks=%" PRIu64
            " A=%zu B=%" PRIu64 " chunk=%" PRIu64 " heap=%" PRIu64
            "; requested transport=ibrc (verify IB/GDR in runtime logs)\n",
            rank, hostname_length, hostname, prop.name, prop.multiProcessorCount,
            l.grid_blocks, o.a_size, l.b_bytes, l.chunk_bytes, heap_bytes);
        std::fprintf(stderr, "rank %d element=%" PRIu64 " block=%" PRIu64
            " multiplier=%" PRIu64 " iterations=%" PRIu64 " validate=%d\n",
            rank, o.element_bytes, o.block_size, o.sm_multiplier, o.iterations, int(o.validate));
        cudaStream_t stream;
        ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
        ATLC_CHECK_CUDA(cudaMemsetAsync, signals, 0, l.signal_bytes, stream);
        curandGenerator_t rng;
        ATLC_CHECK_CURAND(curandCreateGenerator, &rng, CURAND_RNG_PSEUDO_DEFAULT);
        ATLC_CHECK_CURAND(curandSetStream, rng, stream);
        const unsigned long long seed = 1234ULL + rank;
        ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng, seed);
        if (o.a_size / 4)
            ATLC_CHECK_CURAND(curandGenerate, rng, reinterpret_cast<unsigned int*>(a), o.a_size / 4);
        if (o.a_size % 4) {
            random_tail<<<1, 1, 0, stream>>>(a + o.a_size - o.a_size % 4, o.a_size % 4, seed);
            ATLC_CHECK_CUDA(cudaGetLastError);
        }
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        ATLC_CHECK_CURAND(curandDestroyGenerator, rng);
        uint64_t a_bytes = o.a_size, chunk = l.chunk_bytes, element = o.element_bytes, generation = 0;
        int peer = 1 - rank;
        void* args[] = {&a, &b, &signals, &a_bytes, &chunk, &element, &generation, &peer};
        const void* kernel = reinterpret_cast<const void*>(swap_buffer);
        int max_blocks = 0;
        check_nvshmem(nvshmemx_collective_launch_query_gridsize(kernel, dim3(o.block_size), args,
                                                               0, &max_blocks), "collective launch query");
        if (max_blocks <= 0 || l.grid_blocks > uint64_t(max_blocks))
            throw std::runtime_error("SM multiplier/block size exceeds NVSHMEM cooperative launch capacity (" +
                                     std::to_string(max_blocks) + " blocks)");
        auto exchange = [&] {
            check_nvshmem(nvshmemx_collective_launch(kernel, dim3(l.grid_blocks), dim3(o.block_size),
                                                    args, 0, stream), "collective launch");
            ATLC_CHECK_CUDA(cudaGetLastError);
        };
        std::vector<unsigned char> original, peer_original, actual;
        if (o.validate) {
            original.resize(o.a_size); peer_original.resize(o.a_size); actual.resize(o.a_size);
            ATLC_CHECK_CUDA(cudaMemcpy, original.data(), a, o.a_size, cudaMemcpyDeviceToHost);
            ATLC_CHECK_MPI(MPI_Sendrecv, original.data(), int(o.a_size), MPI_BYTE, peer, 0,
                           peer_original.data(), int(o.a_size), MPI_BYTE, peer, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        auto validate = [&](bool peer_expected) {
            if (!o.validate) return;
            ATLC_CHECK_CUDA(cudaMemcpy, actual.data(), a, o.a_size, cudaMemcpyDeviceToHost);
            if (actual != (peer_expected ? peer_original : original))
                throw std::runtime_error("byte-for-byte A validation failed at generation " + std::to_string(generation));
            ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
        };
        ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
        exchange(); // Exactly one untimed full swap.
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        validate(true);
        cudaEvent_t start, stop;
        ATLC_CHECK_CUDA(cudaEventCreate, &start);
        ATLC_CHECK_CUDA(cudaEventCreate, &stop);
        if (!rank) std::printf("iteration,buffer_a_bytes,buffer_b_bytes,element_bytes,block_size,sm_multiplier,grid_blocks,chunk_bytes,elapsed_ms\n");
        for (uint64_t i = 0; i < o.iterations; ++i) {
            generation = (i + 1) * l.rounds;
            ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
            ATLC_CHECK_CUDA(cudaEventRecord, start, stream);
            exchange();
            ATLC_CHECK_CUDA(cudaEventRecord, stop, stream);
            ATLC_CHECK_CUDA(cudaEventSynchronize, stop);
            float elapsed, maximum = 0;
            ATLC_CHECK_CUDA(cudaEventElapsedTime, &elapsed, start, stop);
            ATLC_CHECK_MPI(MPI_Reduce, &elapsed, &maximum, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
            validate(i % 2 != 0); // Warm-up swapped once; timed swap 0 restores originals.
            if (!rank && (std::printf("%" PRIu64 ",%zu,%" PRIu64 ",%" PRIu64 ",%" PRIu64
                                      ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.3f\n",
                i, o.a_size, l.b_bytes, o.element_bytes, o.block_size, o.sm_multiplier,
                l.grid_blocks, l.chunk_bytes, double(maximum)) < 0 || std::fflush(stdout)))
                throw std::runtime_error("failed to write CSV");
        }
        if (o.validate) std::fprintf(stderr, "rank %d: warm-up and all timed swaps validated\n", rank);
        ATLC_CHECK_CUDA(cudaEventDestroy, stop);
        ATLC_CHECK_CUDA(cudaEventDestroy, start);
        ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
        ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
        nvshmem_free(signals); nvshmem_free(b); nvshmem_free(a);
        nvshmem_finalize();
        ATLC_CHECK_MPI(MPI_Finalize);
        mpi_active = false;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rank %d: %s\n", rank, e.what());
        // The peer may be waiting in a GPU kernel or collective; abort rather
        // than attempt cleanup that could hang indefinitely.
        if (mpi_active) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
}
