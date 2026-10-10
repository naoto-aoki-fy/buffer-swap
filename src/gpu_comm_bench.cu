#include <cuda_runtime.h>
#include <curand.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <atlc/check_cuda.hpp>
#include <atlc/check_curand.hpp>
#include "bench_options.hpp"

static_assert(sizeof(size_t) >= 8, "This benchmark requires 64-bit size_t");

static void usage() {
    std::fprintf(stderr,
        "Usage: gpu_comm_bench [options] (one process, CUDA devices 0 and 1)\n"
        "  -a, --a-size SIZE       Buffer A bytes per GPU (default 32G; multiple of 8)\n"
        "  -b, --block-size N      Threads per block (default 256)\n"
        "  -g, --grid-size N       Number of blocks (default 1024)\n"
        "  -n, --iterations N      Timed exchanges (default 128)\n"
        "      --validate          Compare both buffers after every exchange; A <= 16M\n"
        "  -h, --help              Show this help on stderr\n"
        "SIZE is a positive integer with optional K/M/G suffix (binary units).\n"
        "Verify NVLink for the selected GPUs with nvidia-smi topo -m; P2P alone\n"
        "does not establish NVLink transport. See README.md.\n");
}

__global__ void exchange(uint64_t* local, uint64_t* remote, uint64_t count) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const uint64_t local_value = local[i];
        const uint64_t remote_value = remote[i];
        local[i] = remote_value;
        remote[i] = local_value;
    }
    // Publish stores to peer memory before this kernel completes.
    __threadfence_system();
}

// Own partially constructed resources too, so exceptions release allocations.
struct Resources {
    uint64_t* a[2] = {nullptr, nullptr};
    cudaStream_t stream[2] = {nullptr, nullptr};
    curandGenerator_t rng[2] = {nullptr, nullptr};
    cudaEvent_t start = nullptr, stop = nullptr;
    bool peer_enabled = false;
    // Checked cleanup on successful runs; destructor handles partial failures.
    void close() {
        for (int device = 0; device < 2; ++device) {
            ATLC_CHECK_CUDA(cudaSetDevice, device);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream[device]);
            if (device == 0) {
                ATLC_CHECK_CUDA(cudaEventDestroy, stop);
                stop = nullptr;
                ATLC_CHECK_CUDA(cudaEventDestroy, start);
                start = nullptr;
            }
            ATLC_CHECK_CUDA(cudaFree, a[device]);
            a[device] = nullptr;
            ATLC_CHECK_CUDA(cudaStreamDestroy, stream[device]);
            stream[device] = nullptr;
        }
        ATLC_CHECK_CUDA(cudaSetDevice, 0);
        ATLC_CHECK_CUDA(cudaDeviceDisablePeerAccess, 1);
        peer_enabled = false;
    }
    ~Resources() {
        for (int device = 0; device < 2; ++device) {
            if (cudaSetDevice(device) != cudaSuccess) continue;
            if (stream[device]) cudaStreamSynchronize(stream[device]);
            if (rng[device]) curandDestroyGenerator(rng[device]);
            if (device == 0) {
                if (stop) cudaEventDestroy(stop);
                if (start) cudaEventDestroy(start);
            }
            if (a[device]) cudaFree(a[device]);
            if (stream[device]) cudaStreamDestroy(stream[device]);
        }
        if (peer_enabled && cudaSetDevice(0) == cudaSuccess)
            cudaDeviceDisablePeerAccess(1);
    }
};

static void validate(const Resources& resources, size_t bytes,
                     const std::vector<unsigned char> (&original)[2],
                     bool swapped, const std::string& stage) {
    std::vector<unsigned char> actual(bytes);
    for (int device = 0; device < 2; ++device) {
        ATLC_CHECK_CUDA(cudaSetDevice, device);
        ATLC_CHECK_CUDA(cudaMemcpy, actual.data(), resources.a[device], bytes,
                       cudaMemcpyDeviceToHost);
        if (actual != original[swapped ? 1 - device : device])
            throw std::runtime_error("buffer A validation failed on GPU " +
                                     std::to_string(device) + " after " + stage);
    }
    ATLC_CHECK_CUDA(cudaSetDevice, 0);
}

int main(int argc, char** argv) {
    try {
        const auto options = bench::parse_options(argc, argv);
        if (options.help) { usage(); return 0; }
        int devices;
        ATLC_CHECK_CUDA(cudaGetDeviceCount, &devices);
        if (devices < 2) throw std::runtime_error("two visible CUDA GPUs are required");
        cudaDeviceProp properties[2];
        for (int device = 0; device < 2; ++device) {
            ATLC_CHECK_CUDA(cudaGetDeviceProperties, &properties[device], device);
            if (!properties[device].unifiedAddressing)
                throw std::runtime_error("both GPUs must support unified virtual addressing");
            char bus_id[32];
            ATLC_CHECK_CUDA(cudaDeviceGetPCIBusId, bus_id, sizeof(bus_id), device);
            std::fprintf(stderr, "CUDA device %d: %s, PCI bus %s, memory=%zu bytes\n",
                         device, properties[device].name, bus_id,
                         properties[device].totalGlobalMem);
        }
        bench::validate_launch_limits(options, properties[0].maxThreadsPerBlock,
                                     properties[0].maxThreadsDim[0],
                                     properties[0].maxGridSize[0]);
        ATLC_CHECK_CUDA(cudaSetDevice, 0);
        cudaFuncAttributes attributes;
        ATLC_CHECK_CUDA(cudaFuncGetAttributes, &attributes, exchange);
        bench::validate_launch_limits(options, attributes.maxThreadsPerBlock,
                                     properties[0].maxThreadsDim[0],
                                     properties[0].maxGridSize[0]);
        int accessible;
        ATLC_CHECK_CUDA(cudaDeviceCanAccessPeer, &accessible, 0, 1);
        if (!accessible)
            throw std::runtime_error("GPU 0 cannot directly access GPU 1 memory via CUDA P2P");
        std::fprintf(stderr,
            "Transport: CUDA P2P; NVLink UNVERIFIED. CUDA P2P may use PCIe.\n"
            "Only label these results NVLink after checking the selected PCI bus IDs\n"
            "against nvidia-smi topo -m (NV# link); see README.md.\n"
            "A=%zu bytes per GPU, block=%u, grid=%u, iterations=%" PRIu64 "\n",
            options.a_size, options.block_size, options.grid_size, options.iterations);
        Resources resources;
        ATLC_CHECK_CUDA(cudaDeviceEnablePeerAccess, 1, 0);
        resources.peer_enabled = true;
        std::vector<unsigned char> original[2];
        for (int device = 0; device < 2; ++device) {
            ATLC_CHECK_CUDA(cudaSetDevice, device);
            ATLC_CHECK_CUDA(cudaStreamCreate, &resources.stream[device]);
            ATLC_CHECK_CURAND(curandCreateGenerator, &resources.rng[device],
                             CURAND_RNG_PSEUDO_DEFAULT);
            ATLC_CHECK_CURAND(curandSetStream, resources.rng[device], resources.stream[device]);
            size_t free_bytes, total_bytes;
            ATLC_CHECK_CUDA(cudaMemGetInfo, &free_bytes, &total_bytes);
            if (options.a_size > free_bytes)
                throw std::runtime_error("insufficient free memory for buffer A on GPU " +
                                         std::to_string(device));
            ATLC_CHECK_CUDA(cudaMalloc, reinterpret_cast<void**>(&resources.a[device]),
                           options.a_size);
            ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed,
                             resources.rng[device], 1234ULL + device);
            ATLC_CHECK_CURAND(curandGenerate, resources.rng[device],
                             reinterpret_cast<unsigned int*>(resources.a[device]),
                             options.a_size / sizeof(unsigned int));
            ATLC_CHECK_CUDA(cudaStreamSynchronize, resources.stream[device]);
            ATLC_CHECK_CURAND(curandDestroyGenerator, resources.rng[device]);
            resources.rng[device] = nullptr;
            if (options.validate) {
                original[device].resize(options.a_size);
                ATLC_CHECK_CUDA(cudaMemcpy, original[device].data(), resources.a[device],
                               options.a_size, cudaMemcpyDeviceToHost);
            }
        }
        ATLC_CHECK_CUDA(cudaSetDevice, 0);
        const cudaStream_t stream = resources.stream[0];
        const uint64_t count = options.a_size / sizeof(uint64_t);
        ATLC_CHECK_CUDA(cudaEventCreate, &resources.start);
        ATLC_CHECK_CUDA(cudaEventCreate, &resources.stop);
        exchange<<<options.grid_size, options.block_size, 0, stream>>>(
            resources.a[0], resources.a[1], count);
        ATLC_CHECK_CUDA(cudaGetLastError);
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        bool swapped = true; // One full warm-up swap, checked against initial data.
        if (options.validate) validate(resources, options.a_size, original, swapped, "warm-up");
        if (std::printf("iteration,buffer_a_bytes,block_size,grid_size,elapsed_ms\n") < 0)
            throw std::runtime_error("failed to write CSV header");
        for (uint64_t iteration = 0; iteration < options.iterations; ++iteration) {
            ATLC_CHECK_CUDA(cudaEventRecord, resources.start, stream);
            exchange<<<options.grid_size, options.block_size, 0, stream>>>(
                resources.a[0], resources.a[1], count);
            ATLC_CHECK_CUDA(cudaGetLastError);
            ATLC_CHECK_CUDA(cudaEventRecord, resources.stop, stream);
            ATLC_CHECK_CUDA(cudaEventSynchronize, resources.stop);
            float elapsed_ms;
            ATLC_CHECK_CUDA(cudaEventElapsedTime, &elapsed_ms, resources.start, resources.stop);
            swapped = !swapped;
            if (options.validate)
                validate(resources, options.a_size, original, swapped,
                         "iteration " + std::to_string(iteration));
            if (std::printf("%" PRIu64 ",%zu,%u,%u,%.6f\n", iteration,
                            options.a_size, options.block_size, options.grid_size,
                            double(elapsed_ms)) < 0 || std::fflush(stdout) != 0)
                throw std::runtime_error("failed to write CSV output");
        }
        if (options.validate)
            std::fprintf(stderr, "byte-for-byte validation passed on both GPUs (warm-up and all iterations)\n");
        resources.close();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
