// Compile the production benchmark with delays at communication boundaries.
// This binary is for correctness testing only; its timings are not benchmarks.
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__device__ void protocol_test_delay(int peer, unsigned phase) {
    // Vary delays across ranks, blocks, phases and calls without changing the
    // protocol, adding memory fences, or synchronizing otherwise independent CTAs.
    const unsigned long long start = clock64();
    const unsigned long long cycles = 1024 *
        (1 + ((start / 1024 + blockIdx.x * 3 + peer * 5 + phase) % 32));
    while (clock64() - start < cycles) {}
}

__device__ void skew_getmem_block(void* dest, const void* source, size_t bytes, int peer) {
    protocol_test_delay(peer, 0);
    nvshmemx_getmem_block(dest, source, bytes, peer);
}

__device__ void skew_putmem_signal_block(void* dest, const void* source, size_t bytes,
                                        uint64_t* signal, uint64_t value, int op, int peer) {
    protocol_test_delay(peer, 1);
    nvshmemx_putmem_signal_block(dest, source, bytes, signal, value, op, peer);
}

__device__ void skew_signal_op(uint64_t* signal, uint64_t value, int op, int peer) {
    protocol_test_delay(peer, 2);
    nvshmemx_signal_op(signal, value, op, peer);
}

#define nvshmemx_getmem_block skew_getmem_block
#define nvshmemx_putmem_signal_block skew_putmem_signal_block
#define nvshmemx_signal_op skew_signal_op
#include "../src/nvshmem_swap_bench.cu"
