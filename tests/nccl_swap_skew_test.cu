// Compile the production benchmark with rank submission and copy-stream skew.
// This binary is for correctness testing only; its timings are not benchmarks.
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <chrono>
#include <thread>

__global__ void copy_test_delay(unsigned long long cycles) {
    const unsigned long long start = clock64();
    while (clock64() - start < cycles) {}
}

static ncclResult_t skew_group_start() {
    static unsigned call = 0;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Alternate the slower submitter without synchronizing the ranks.
    if ((++call + unsigned(rank)) % 7 == 0)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    return ncclGroupStart();
}

static cudaError_t skew_copy(void* dest, const void* source, size_t bytes,
                             cudaMemcpyKind kind, cudaStream_t stream) {
    static unsigned call = 0;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Delay slot consumption on the GPU, exposing premature receive-slot reuse.
    copy_test_delay<<<1, 1, 0, stream>>>(4096ULL * (1 + (++call + rank * 3) % 17));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    return cudaMemcpyAsync(dest, source, bytes, kind, stream);
}

#define ncclGroupStart skew_group_start
#define cudaMemcpyAsync skew_copy
#include "../src/nccl_swap_bench.cu"
