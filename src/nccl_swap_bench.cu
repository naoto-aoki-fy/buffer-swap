#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <atlc/check_cuda.hpp>
#include <atlc/check_curand.hpp>
#include <atlc/check_mpi.hpp>
#include <atlc/check_nccl.hpp>
#include "bench_options.hpp"

static_assert(sizeof(size_t) >= 8, "This benchmark requires 64-bit size_t");

static void usage() {
    std::fprintf(stderr,
        "Usage: nccl_swap_bench [options] (exactly two ranks, one GPU per rank)\n"
        "  -a, --a-size SIZE       Main buffer bytes (default 32G)\n"
        "  -b, --b-size SIZE       Total staging bytes (default 1G)\n"
        "  -n, --iterations N      Timed exchanges (default 128)\n"
        "      --skip-copy-back    Leave A unchanged; omit B-to-A copies\n"
        "      --pipeline-copy-back     Use two slots and a separate copy stream (default)\n"
        "      --no-pipeline-copy-back  Use one slot; serialize communication and copies\n"
        "      --no-blocking-stream-sync  Use default stream synchronization\n"
        "      --no-blocking-event-sync   Use default event synchronization\n"
        "      --validate          Check warm-up and every iteration; A/B <= 16M\n"
        "  -h, --help              Show this help on stderr\n"
        "SIZE is a positive integer with optional K/M/G suffix (binary units).\n");
}

// cuRAND's host API writes whole unsigned ints. Initialize a 1-3 byte tail
// separately on the GPU so arbitrary byte sizes need no padded allocation.
__global__ void random_tail(unsigned char* tail, size_t count,
                            unsigned long long seed) {
    curandState state;
    curand_init(seed, 0, 0, &state);
    const unsigned int value = curand(&state);
    for (size_t i = 0; i < count; ++i)
        tail[i] = static_cast<unsigned char>(value >> (8 * i));
}

static void initialize(unsigned char* data, size_t bytes,
                       curandGenerator_t rng, cudaStream_t stream,
                       unsigned long long seed) {
    ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng, seed);
    const size_t words = bytes / sizeof(unsigned int);
    if (words)
        ATLC_CHECK_CURAND(curandGenerate, rng,
                         reinterpret_cast<unsigned int*>(data), words);
    const size_t tail = bytes % sizeof(unsigned int);
    if (tail) {
        random_tail<<<1, 1, 0, stream>>>(data + bytes - tail, tail, seed);
        ATLC_CHECK_CUDA(cudaGetLastError);
    }
}

struct ExchangePipeline {
    cudaStream_t copy_stream{};
    cudaEvent_t transfer_done[2]{};
    cudaEvent_t copy_done[2]{};
};

static void exchange(unsigned char* a, unsigned char* b,
                     const bench::Options& options, int rank,
                     ncclComm_t comm, cudaStream_t stream,
                     const ExchangePipeline& pipeline) {
    // Keep the total B allocation fixed. An odd B leaves one byte unused.
    // B=1 cannot hold two slots; skip-copy mode needs no copy pipeline.
    const bool pipelined = options.pipeline_copy_back && options.copy_back && options.b_size >= 2;
    const size_t chunk_size = pipelined ? options.b_size / 2 : options.b_size;
    bool used[2] = {};
    unsigned slot = 0;
    for (size_t offset = 0; offset < options.a_size;) {
        const size_t bytes = std::min(chunk_size, options.a_size - offset);
        unsigned char* receive = b + slot * chunk_size;
        if (pipelined && used[slot])
            ATLC_CHECK_CUDA(cudaStreamWaitEvent, stream, pipeline.copy_done[slot], 0);
        ATLC_CHECK_NCCL(ncclGroupStart);
        ATLC_CHECK_NCCL(ncclSend, a + offset, bytes, ncclUint8, 1 - rank, comm, stream);
        ATLC_CHECK_NCCL(ncclRecv, receive, bytes, ncclUint8, 1 - rank, comm, stream);
        ATLC_CHECK_NCCL(ncclGroupEnd);
        if (pipelined) {
            // Both send and receive must finish before overwriting this A chunk.
            ATLC_CHECK_CUDA(cudaEventRecord, pipeline.transfer_done[slot], stream);
            ATLC_CHECK_CUDA(cudaStreamWaitEvent, pipeline.copy_stream,
                            pipeline.transfer_done[slot], 0);
            ATLC_CHECK_CUDA(cudaMemcpyAsync, a + offset, receive, bytes,
                            cudaMemcpyDeviceToDevice, pipeline.copy_stream);
            ATLC_CHECK_CUDA(cudaEventRecord, pipeline.copy_done[slot], pipeline.copy_stream);
            used[slot] = true;
            slot ^= 1;
        } else if (options.copy_back) {
            ATLC_CHECK_CUDA(cudaMemcpyAsync, a + offset, receive, bytes,
                            cudaMemcpyDeviceToDevice, stream);
        }
        offset += bytes;
    }
    // Join the copy stream so warm-up synchronization, the timed stop event,
    // and any subsequent exchange include every copy. Stream order means the
    // last copy also covers all earlier copies, including the other slot.
    if (pipelined)
        ATLC_CHECK_CUDA(cudaStreamWaitEvent, stream, pipeline.copy_done[slot ^ 1], 0);
}

int main(int argc, char** argv) {
    bool mpi_active = false;
    int rank = -1;
    try {
        ATLC_CHECK_MPI(MPI_Init, &argc, &argv);
        mpi_active = true;
        ATLC_CHECK_MPI(MPI_Comm_set_errhandler, MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        ATLC_CHECK_MPI(MPI_Comm_rank, MPI_COMM_WORLD, &rank);
        const auto options = bench::parse_options(argc, argv);
        const uint64_t configuration[] = {options.a_size, options.b_size,
            options.iterations, uint64_t(options.copy_back),
            uint64_t(options.validate), uint64_t(options.help),
            uint64_t(options.blocking_stream_sync), uint64_t(options.blocking_event_sync),
            uint64_t(options.pipeline_copy_back)};
        uint64_t minimum[9], maximum[9];
        ATLC_CHECK_MPI(MPI_Allreduce, configuration, minimum, 9, MPI_UINT64_T,
                       MPI_MIN, MPI_COMM_WORLD);
        ATLC_CHECK_MPI(MPI_Allreduce, configuration, maximum, 9, MPI_UINT64_T,
                       MPI_MAX, MPI_COMM_WORLD);
        if (!std::equal(minimum, minimum + 9, maximum))
            throw std::runtime_error("all ranks must use identical command-line options");
        if (options.help) {
            if (rank == 0) usage();
            ATLC_CHECK_MPI(MPI_Finalize);
            mpi_active = false;
            return 0;
        }
        int ranks;
        ATLC_CHECK_MPI(MPI_Comm_size, MPI_COMM_WORLD, &ranks);
        if (ranks != 2) throw std::runtime_error("exactly two MPI ranks are required");

        // Assign GPUs using the rank within each shared-memory node.
        MPI_Comm local;
        ATLC_CHECK_MPI(MPI_Comm_split_type, MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                       rank, MPI_INFO_NULL, &local);
        int local_size, local_rank;
        ATLC_CHECK_MPI(MPI_Comm_size, local, &local_size);
        ATLC_CHECK_MPI(MPI_Comm_rank, local, &local_rank);
        int devices;
        ATLC_CHECK_CUDA(cudaGetDeviceCount, &devices);
        if (!devices) throw std::runtime_error("no visible CUDA device");
        // Per-process GPU binding may expose only the assigned device.
        // Otherwise local ranks must see the same ordered device list.
        const int device = devices == 1 ? 0 : local_rank;
        if (device >= devices)
            throw std::runtime_error("not enough visible CUDA devices for local MPI ranks");
        if (options.blocking_stream_sync)
            ATLC_CHECK_CUDA(cudaSetDeviceFlags, cudaDeviceScheduleBlockingSync);
        ATLC_CHECK_CUDA(cudaSetDevice, device);
        cudaDeviceProp properties;
        ATLC_CHECK_CUDA(cudaGetDeviceProperties, &properties, device);
        // UUIDs detect accidental sharing even with different visibility lists.
        std::vector<cudaUUID_t> local_uuids(local_size);
        ATLC_CHECK_MPI(MPI_Allgather, &properties.uuid, int(sizeof(cudaUUID_t)), MPI_BYTE,
                       local_uuids.data(), int(sizeof(cudaUUID_t)), MPI_BYTE, local);
        ATLC_CHECK_MPI(MPI_Comm_free, &local);
        if (local_size == 2 &&
            std::equal(local_uuids[0].bytes, local_uuids[0].bytes + sizeof(local_uuids[0].bytes),
                       local_uuids[1].bytes))
            throw std::runtime_error("local ranks must use distinct GPUs; check CUDA_VISIBLE_DEVICES");
        char hostname[MPI_MAX_PROCESSOR_NAME] = {};
        int hostname_length;
        ATLC_CHECK_MPI(MPI_Get_processor_name, hostname, &hostname_length);
        std::fprintf(stderr, "rank %d host %.*s CUDA device %d: %s; A=%zu B=%zu copy_back=%d; placement=%s\n",
                     rank, hostname_length, hostname, device, properties.name,
                     options.a_size, options.b_size, int(options.copy_back),
                     local_size == 2 ? "single-node" : "two-node");
        const bool pipelined = options.pipeline_copy_back && options.copy_back && options.b_size >= 2;
        std::fprintf(stderr, "rank %d: staging_slots=%d chunk_bytes=%zu\n", rank,
                     pipelined ? 2 : 1, pipelined ? options.b_size / 2 : options.b_size);

        // Cross-node runs force IB; local runs let NCCL select P2P/SHM
        // without requiring an IB adapter. Keep NCCL logs off CSV stdout.
        if ((local_size == 1 ? setenv("NCCL_NET", "IB", 1) : unsetenv("NCCL_NET")) ||
            setenv("NCCL_DEBUG_FILE", "/dev/stderr", 1))
            throw std::runtime_error("failed to configure NCCL environment");
        int nccl_version;
        ATLC_CHECK_NCCL(ncclGetVersion, &nccl_version);
        if (nccl_version < 21000)
            throw std::runtime_error("NCCL 2.10 or newer is required to force the IB network");
        ncclUniqueId id;
        if (rank == 0) ATLC_CHECK_NCCL(ncclGetUniqueId, &id);
        ATLC_CHECK_MPI(MPI_Bcast, &id, int(sizeof(id)), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclComm_t comm;
        ATLC_CHECK_NCCL(ncclCommInitRank, &comm, 2, id, rank);
        cudaStream_t stream;
        ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
        ExchangePipeline pipeline;
        if (pipelined) {
            ATLC_CHECK_CUDA(cudaStreamCreateWithFlags, &pipeline.copy_stream, cudaStreamNonBlocking);
            for (unsigned slot = 0; slot < 2; ++slot) {
                ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &pipeline.transfer_done[slot], cudaEventDisableTiming);
                ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &pipeline.copy_done[slot], cudaEventDisableTiming);
            }
        }
        size_t free_bytes, total_bytes;
        ATLC_CHECK_CUDA(cudaMemGetInfo, &free_bytes, &total_bytes);
        if (options.a_size + options.b_size > free_bytes)
            throw std::runtime_error("insufficient GPU memory for A+B (NCCL also needs memory)");
        unsigned char *a, *b;
        ATLC_CHECK_CUDA(cudaMalloc, reinterpret_cast<void**>(&a), options.a_size);
        ATLC_CHECK_CUDA(cudaMalloc, reinterpret_cast<void**>(&b), options.b_size);
        curandGenerator_t rng;
        ATLC_CHECK_CURAND(curandCreateGenerator, &rng, CURAND_RNG_PSEUDO_DEFAULT);
        ATLC_CHECK_CURAND(curandSetStream, rng, stream);
        initialize(a, options.a_size, rng, stream, 1234ULL + rank * 2);
        initialize(b, options.b_size, rng, stream, 1235ULL + rank * 2);
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        ATLC_CHECK_CURAND(curandDestroyGenerator, rng);

        // Validation uses bounded host snapshots only as references. The actual
        // exchange always uses NCCL GPU buffers, including in validation mode.
        std::vector<unsigned char> original, peer_original, actual;
        if (options.validate) {
            original.resize(options.a_size);
            peer_original.resize(options.a_size);
            actual.resize(options.a_size);
            ATLC_CHECK_CUDA(cudaMemcpy, original.data(), a, options.a_size,
                           cudaMemcpyDeviceToHost);
            ATLC_CHECK_MPI(MPI_Sendrecv, original.data(), int(options.a_size), MPI_BYTE,
                           1 - rank, 0, peer_original.data(), int(options.a_size), MPI_BYTE,
                           1 - rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Establish lazy NCCL connections outside the measured iterations.
        // Validate warm-up against the initial data, before it can mask errors.
        ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
        exchange(a, b, options, rank, comm, stream, pipeline);
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        if (options.validate) {
            ATLC_CHECK_CUDA(cudaMemcpy, actual.data(), a, options.a_size,
                           cudaMemcpyDeviceToHost);
            if (actual != (options.copy_back ? peer_original : original))
                throw std::runtime_error("buffer A validation failed during warm-up");
        }
        cudaEvent_t start, stop;
        ATLC_CHECK_CUDA(cudaEventCreate, &start);
        ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &stop,
                        options.blocking_event_sync ? cudaEventBlockingSync : cudaEventDefault);
        if (rank == 0)
            std::printf("iteration,buffer_a_bytes,buffer_b_bytes,copy_back,elapsed_ms\n");
        for (uint64_t iteration = 0; iteration < options.iterations; ++iteration) {
            ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
            ATLC_CHECK_CUDA(cudaEventRecord, start, stream);
            exchange(a, b, options, rank, comm, stream, pipeline);
            ATLC_CHECK_CUDA(cudaEventRecord, stop, stream);
            ATLC_CHECK_CUDA(cudaEventSynchronize, stop);
            float elapsed_ms, max_ms = 0;
            ATLC_CHECK_CUDA(cudaEventElapsedTime, &elapsed_ms, start, stop);
            ATLC_CHECK_MPI(MPI_Reduce, &elapsed_ms, &max_ms, 1, MPI_FLOAT,
                           MPI_MAX, 0, MPI_COMM_WORLD);
            if (options.validate) {
                ATLC_CHECK_CUDA(cudaMemcpy, actual.data(), a, options.a_size,
                               cudaMemcpyDeviceToHost);
                // Warm-up has already performed the first swap.
                const auto& expected = options.copy_back && iteration % 2 == 1
                                       ? peer_original : original;
                if (actual != expected)
                    throw std::runtime_error("buffer A validation failed at iteration " +
                                             std::to_string(iteration));
                ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
            }
            if (rank == 0) {
                if (std::printf("%" PRIu64 ",%zu,%zu,%d,%.3f\n", iteration,
                                options.a_size, options.b_size,
                                int(options.copy_back), double(max_ms)) < 0 ||
                    std::fflush(stdout) != 0)
                    throw std::runtime_error("failed to write CSV output");
            }
        }
        if (options.validate)
            std::fprintf(stderr, "rank %d: warm-up and all timed swaps validated\n", rank);
        ATLC_CHECK_CUDA(cudaEventDestroy, stop);
        ATLC_CHECK_CUDA(cudaEventDestroy, start);
        ATLC_CHECK_NCCL(ncclCommDestroy, comm);
        ATLC_CHECK_CUDA(cudaFree, b);
        ATLC_CHECK_CUDA(cudaFree, a);
        if (pipelined) {
            for (unsigned slot = 0; slot < 2; ++slot) {
                ATLC_CHECK_CUDA(cudaEventDestroy, pipeline.copy_done[slot]);
                ATLC_CHECK_CUDA(cudaEventDestroy, pipeline.transfer_done[slot]);
            }
            ATLC_CHECK_CUDA(cudaStreamDestroy, pipeline.copy_stream);
        }
        ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
        ATLC_CHECK_MPI(MPI_Finalize);
        mpi_active = false;
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
        // A peer may be blocked in NCCL or a collective. Abort the whole job;
        // attempting stream/resource cleanup here can itself deadlock.
        if (mpi_active) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
}
