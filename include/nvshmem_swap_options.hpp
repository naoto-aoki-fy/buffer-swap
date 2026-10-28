#pragma once

#include "bench_options.hpp"

namespace nvshmem_swap {
static_assert(sizeof(size_t) >= 8, "This benchmark requires 64-bit size_t");
struct Options {
    size_t a_size = size_t{32} << 30;
    size_t chunk_size = size_t{256} << 10;
    uint64_t element_bytes = 8, block_size = 256, sm_multiplier = 1, iterations = 128;
    uint64_t method = 1;
    bool validate = false, help = false;
};
inline Options parse_options(int argc, char** argv) {
    Options o;
    const option long_options[] = {
        {"a-size", required_argument, nullptr, 'a'},
        {"chunk-size", required_argument, nullptr, 'c'},
        {"element-bytes", required_argument, nullptr, 'e'},
        {"block-size", required_argument, nullptr, 't'},
        {"sm-multiplier", required_argument, nullptr, 'm'},
        {"iterations", required_argument, nullptr, 'n'},
        {"validate", no_argument, nullptr, 256},
        {"method", required_argument, nullptr, 257},
        {"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}
    };
    opterr = 0;
    int c;
    while ((c = getopt_long(argc, argv, "a:c:e:t:m:n:h", long_options, nullptr)) != -1) {
        switch (c) {
            case 'a': o.a_size = bench::parse_size(optarg); break;
            case 'c': o.chunk_size = bench::parse_size(optarg); break;
            case 'e': o.element_bytes = bench::positive_integer(optarg); break;
            case 't': o.block_size = bench::positive_integer(optarg); break;
            case 'm': o.sm_multiplier = bench::positive_integer(optarg); break;
            case 'n': o.iterations = bench::positive_integer(optarg); break;
            case 256: o.validate = true; break;
            case 257: o.method = bench::positive_integer(optarg); break;
            case 'h': o.help = true; break;
            default: throw std::invalid_argument("unknown option or missing option argument");
        }
    }
    if (optind != argc) throw std::invalid_argument("unexpected positional argument");
    if (o.method < 1 || o.method > 3)
        throw std::invalid_argument("method must be 1, 2 or 3");
    if (o.element_bytes != 1 && o.element_bytes != 2 && o.element_bytes != 4 && o.element_bytes != 8)
        throw std::invalid_argument("element bytes must be 1, 2, 4 or 8");
    if (!o.chunk_size || o.chunk_size % o.element_bytes)
        throw std::invalid_argument("chunk size must be positive and a multiple of element bytes");
    if (o.validate && o.a_size > (size_t{16} << 20))
        throw std::invalid_argument("--validate requires A <= 16 MiB per GPU");
    return o;
}
inline uint64_t multiply(uint64_t a, uint64_t b) {
    if (!a || !b || a > UINT64_MAX / b)
        throw std::invalid_argument("zero value or configuration multiplication overflow");
    return a * b;
}
inline uint64_t add(uint64_t a, uint64_t b) {
    if (a > UINT64_MAX - b) throw std::invalid_argument("configuration addition overflow");
    return a + b;
}
struct Layout {
    uint64_t grid_blocks, chunk_bytes, b_bytes, signal_bytes, rounds, heap_required;
    uint64_t signal_generations, private_bytes;
};
inline Layout layout(const Options& o, uint64_t sms, uint64_t max_threads,
                     uint64_t max_block_x, uint64_t max_grid_x) {
    if (o.method < 1 || o.method > 3)
        throw std::invalid_argument("method must be 1, 2 or 3");
    if (!o.a_size || !o.block_size || o.block_size > max_threads || o.block_size > max_block_x)
        throw std::invalid_argument("block size exceeds hardware limits or zero A size");
    if ((o.element_bytes != 1 && o.element_bytes != 2 && o.element_bytes != 4 && o.element_bytes != 8) ||
        !o.chunk_size || o.chunk_size % o.element_bytes)
        throw std::invalid_argument("invalid element bytes or unaligned chunk size");
    Layout l{};
    l.grid_blocks = multiply(sms, o.sm_multiplier);
    if (l.grid_blocks > max_grid_x || l.grid_blocks > INT32_MAX)
        throw std::invalid_argument("grid blocks exceed hardware limits");
    l.chunk_bytes = o.chunk_size;
    l.b_bytes = multiply(l.chunk_bytes, l.grid_blocks);
    l.signal_bytes = multiply(l.grid_blocks, sizeof(uint64_t));
    l.rounds = o.a_size / l.b_bytes + (o.a_size % l.b_bytes != 0);
    l.signal_generations = multiply(l.rounds, o.method == 1 ? 1 : 2);
    // Reserve a distinct signal generation for warm-up and every timed swap.
    multiply(add(o.iterations, 1), l.signal_generations);
    // Conservative room for allocator alignment/runtime symmetric allocations.
    l.private_bytes = o.method == 1 ? 0 : o.a_size;
    l.heap_required = add(add(add(o.method == 1 ? o.a_size : 0, l.b_bytes),
                              l.signal_bytes), uint64_t{1} << 20);
    add(l.heap_required, l.private_bytes);
    return l;
}
inline void check_capacity(const Layout& l, uint64_t heap_bytes, uint64_t free_bytes) {
    if (heap_bytes < l.heap_required)
        throw std::invalid_argument("insufficient NVSHMEM_SYMMETRIC_SIZE for symmetric buffers and 1 MiB reserve");
    if (add(heap_bytes, l.private_bytes) > free_bytes)
        throw std::invalid_argument("insufficient free GPU memory for the NVSHMEM symmetric heap and private A");
}
} // namespace nvshmem_swap
