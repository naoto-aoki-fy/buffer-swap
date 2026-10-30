#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <atlc/check_cuda.hpp>
#include <atlc/check_curand.hpp>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "uva_p2p_options.hpp"

static void usage() {
    std::fprintf(stderr,
        "Usage: uva_swap_bench [options] (one process, CUDA devices 0 and 1)\n"
        "  -a, --a-size SIZE       Buffer bytes per GPU (default 32G)\n"
        "  -e, --element-bytes N   Typed element size: 1, 2, 4, 8 (default 8)\n"
        "  -t, --block-size N      Threads per block (default 256)\n"
        "      --grid-stride       Use the bounded grid-stride kernel (default off)\n"
        "  -m, --sm-multiplier N   Grid-stride block cap per SM (default 4)\n"
        "  -n, --iterations N      Timed swaps (default 128)\n"
        "      --no-blocking-stream-sync  Use default stream synchronization\n"
        "      --no-blocking-event-sync   Use default event synchronization\n"
        "      --validate          Compare all bytes after every swap; A <= 16M\n"
        "  -h, --help              Show this help on stderr\n"
        "SIZE accepts positive integers with optional binary K/M/G suffixes.\n"
        "Verify NVLink for the selected pair using nvidia-smi topo -m or topo -p2p n.\n");
}

__global__ void random_tail(unsigned char* tail, size_t count,
                            unsigned long long seed) {
    curandState state;
    curand_init(seed, 0, 0, &state);
    const unsigned int value = curand(&state);
    for (size_t i = 0; i < count; ++i)
        tail[i] = static_cast<unsigned char>(value >> (8 * i));
}

static void initialize(unsigned char* data, size_t bytes, cudaStream_t stream,
                       unsigned long long seed) {
    curandGenerator_t rng;
    ATLC_CHECK_CURAND(curandCreateGenerator, &rng, CURAND_RNG_PSEUDO_DEFAULT);
    ATLC_CHECK_CURAND(curandSetStream, rng, stream);
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
    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
    ATLC_CHECK_CURAND(curandDestroyGenerator, rng);
}

template <typename T>
__global__ void swap_buffers(T* a0, T* a1, uint64_t elements) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) {
        const T local = a0[i];
        const T remote = a1[i];
        a0[i] = remote;
        a1[i] = local;
    }
}

template <typename T>
__global__ void swap_buffers_grid_stride(T* a0, T* a1, uint64_t elements) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += stride) {
        const T local = a0[i];
        const T remote = a1[i];
        a0[i] = remote;
        a1[i] = local;
        if (stride >= elements - i) break; // Avoid 64-bit index wraparound.
    }
}

template <typename T>
static void launch_swap(unsigned char* a0, unsigned char* a1, uint64_t elements,
                        unsigned grid, unsigned block, bool grid_stride,
                        cudaStream_t stream) {
    if (grid_stride)
        swap_buffers_grid_stride<<<grid, block, 0, stream>>>(
            reinterpret_cast<T*>(a0), reinterpret_cast<T*>(a1), elements);
    else
        swap_buffers<<<grid, block, 0, stream>>>(
            reinterpret_cast<T*>(a0), reinterpret_cast<T*>(a1), elements);
}

static void exchange(unsigned char* a0, unsigned char* a1,
                     const uva_p2p::Options& options, unsigned grid,
                     cudaStream_t stream) {
    const uint64_t elements = options.a_size / options.element_bytes;
    const unsigned block = static_cast<unsigned>(options.block_size);
    switch (options.element_bytes) {
        case 1: launch_swap<uint8_t>(a0, a1, elements, grid, block, options.grid_stride, stream); break;
        case 2: launch_swap<uint16_t>(a0, a1, elements, grid, block, options.grid_stride, stream); break;
        case 4: launch_swap<uint32_t>(a0, a1, elements, grid, block, options.grid_stride, stream); break;
        case 8: launch_swap<uint64_t>(a0, a1, elements, grid, block, options.grid_stride, stream); break;
    }
    ATLC_CHECK_CUDA(cudaGetLastError);
}

static void validate(unsigned char* const a[2],
                     const std::vector<unsigned char> (&original)[2],
                     std::vector<unsigned char>& actual, bool swapped,
                     const std::string& phase) {
    for (int device = 0; device < 2; ++device) {
        ATLC_CHECK_CUDA(cudaSetDevice, device);
        ATLC_CHECK_CUDA(cudaMemcpy, actual.data(), a[device], actual.size(),
                       cudaMemcpyDeviceToHost);
        if (actual != original[swapped ? 1 - device : device])
            throw std::runtime_error("GPU " + std::to_string(device) +
                                     " validation failed after " + phase);
    }
    ATLC_CHECK_CUDA(cudaSetDevice, 0);
    std::fprintf(stderr, "validation passed after %s (both GPUs)\n", phase.c_str());
}

int main(int argc, char** argv) {
    try {
        const auto options = uva_p2p::parse_options(argc, argv);
        if (options.help) { usage(); return 0; }
        int devices;
        ATLC_CHECK_CUDA(cudaGetDeviceCount, &devices);
        if (devices < 2) throw std::runtime_error("two visible CUDA GPUs are required");
        cudaDeviceProp properties[2];
        for (int device = 0; device < 2; ++device) {
            ATLC_CHECK_CUDA(cudaGetDeviceProperties, &properties[device], device);
            if (!properties[device].unifiedAddressing)
                throw std::runtime_error("both GPUs must support CUDA UVA");
            char bus_id[32];
            ATLC_CHECK_CUDA(cudaDeviceGetPCIBusId, bus_id, sizeof(bus_id), device);
            std::fprintf(stderr, "CUDA device %d: %s, PCI bus %s\n",
                         device, properties[device].name, bus_id);
        }
        const uint64_t grid = uva_p2p::grid_size(options,
            properties[0].maxThreadsPerBlock, properties[0].maxThreadsDim[0],
            properties[0].maxGridSize[0], properties[0].multiProcessorCount);
        // Only GPU 0 accesses peer memory. GPU 1 never accesses GPU 0's buffer.
        int accessible;
        ATLC_CHECK_CUDA(cudaDeviceCanAccessPeer, &accessible, 0, 1);
        if (!accessible)
            throw std::runtime_error("direct P2P access from GPU 0 to GPU 1 is unavailable");
        if (options.blocking_stream_sync)
            ATLC_CHECK_CUDA(cudaSetDeviceFlags, cudaDeviceScheduleBlockingSync);
        ATLC_CHECK_CUDA(cudaSetDevice, 0);
        ATLC_CHECK_CUDA(cudaDeviceEnablePeerAccess, 1, 0);
        std::fprintf(stderr,
            "Direct P2P enabled; NVLink is NOT verified by this capability check.\n"
            "Verify the PCI bus pair above with nvidia-smi topo -m (NV#) or topo -p2p n.\n"
            "A=%zu bytes per GPU, element=%" PRIu64 ", block=%" PRIu64
            ", grid=%" PRIu64 ", iterations=%" PRIu64 "\n",
            options.a_size, options.element_bytes, options.block_size, grid, options.iterations);
        std::fprintf(stderr, "kernel=%s, SMs=%d, sm_multiplier=%" PRIu64
            " (used only in grid-stride mode)\n",
            options.grid_stride ? "grid-stride" : "reference",
            properties[0].multiProcessorCount, options.sm_multiplier);

        unsigned char* a[2];
        cudaStream_t streams[2];
        std::vector<unsigned char> original[2], actual;
        for (int device = 0; device < 2; ++device) {
            if (options.blocking_stream_sync)
                ATLC_CHECK_CUDA(cudaSetDeviceFlags, cudaDeviceScheduleBlockingSync);
            ATLC_CHECK_CUDA(cudaSetDevice, device);
            ATLC_CHECK_CUDA(cudaStreamCreate, &streams[device]);
            size_t free_bytes, total_bytes;
            ATLC_CHECK_CUDA(cudaMemGetInfo, &free_bytes, &total_bytes);
            if (options.a_size > free_bytes)
                throw std::runtime_error("insufficient free GPU memory for A on GPU " +
                                         std::to_string(device));
            ATLC_CHECK_CUDA(cudaMalloc, reinterpret_cast<void**>(&a[device]), options.a_size);
            initialize(a[device], options.a_size, streams[device], 1234ULL + device);
            if (options.validate) {
                original[device].resize(options.a_size);
                ATLC_CHECK_CUDA(cudaMemcpy, original[device].data(), a[device],
                               options.a_size, cudaMemcpyDeviceToHost);
            }
        }
        if (options.validate) actual.resize(options.a_size);
        ATLC_CHECK_CUDA(cudaSetDevice, 0);
        exchange(a[0], a[1], options, static_cast<unsigned>(grid), streams[0]);
        ATLC_CHECK_CUDA(cudaStreamSynchronize, streams[0]);
        if (options.validate) validate(a, original, actual, true, "warm-up");
        cudaEvent_t start, stop;
        ATLC_CHECK_CUDA(cudaEventCreate, &start);
        ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &stop,
                        options.blocking_event_sync ? cudaEventBlockingSync : cudaEventDefault);
        if (std::printf("iteration,buffer_a_bytes,element_bytes,block_size,grid_size,elapsed_ms\n") < 0)
            throw std::runtime_error("failed to write CSV header");
        for (uint64_t iteration = 0; iteration < options.iterations; ++iteration) {
            ATLC_CHECK_CUDA(cudaEventRecord, start, streams[0]);
            exchange(a[0], a[1], options, static_cast<unsigned>(grid), streams[0]);
            ATLC_CHECK_CUDA(cudaEventRecord, stop, streams[0]);
            ATLC_CHECK_CUDA(cudaEventSynchronize, stop);
            float elapsed_ms;
            ATLC_CHECK_CUDA(cudaEventElapsedTime, &elapsed_ms, start, stop);
            // One warm-up plus iteration+1 timed swaps: even iteration restores originals.
            if (options.validate)
                validate(a, original, actual, iteration % 2 != 0,
                         "iteration " + std::to_string(iteration));
            if (std::printf("%" PRIu64 ",%zu,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.3f\n",
                            iteration, options.a_size, options.element_bytes,
                            options.block_size, grid, double(elapsed_ms)) < 0 ||
                std::fflush(stdout) != 0)
                throw std::runtime_error("failed to write CSV output");
        }
        ATLC_CHECK_CUDA(cudaEventDestroy, stop);
        ATLC_CHECK_CUDA(cudaEventDestroy, start);
        ATLC_CHECK_CUDA(cudaDeviceDisablePeerAccess, 1);
        for (int device = 0; device < 2; ++device) {
            ATLC_CHECK_CUDA(cudaSetDevice, device);
            ATLC_CHECK_CUDA(cudaFree, a[device]);
            ATLC_CHECK_CUDA(cudaStreamDestroy, streams[device]);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "uva_swap_bench: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
