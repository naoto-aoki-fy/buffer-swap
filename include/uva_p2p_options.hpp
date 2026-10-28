#pragma once

#include "bench_options.hpp"
#include <algorithm>

namespace uva_p2p {
static_assert(sizeof(size_t) >= 8, "This benchmark requires 64-bit size_t");

struct Options {
    size_t a_size = size_t{32} << 30;
    uint64_t element_bytes = 8;
    uint64_t block_size = 256;
    uint64_t sm_multiplier = 4;
    uint64_t iterations = 128;
    bool grid_stride = false;
    bool validate = false;
    bool help = false;
};

inline Options parse_options(int argc, char** argv) {
    Options result;
    const option long_options[] = {
        {"a-size", required_argument, nullptr, 'a'},
        {"element-bytes", required_argument, nullptr, 'e'},
        {"block-size", required_argument, nullptr, 't'},
        {"sm-multiplier", required_argument, nullptr, 'm'},
        {"grid-stride", no_argument, nullptr, 257},
        {"iterations", required_argument, nullptr, 'n'},
        {"validate", no_argument, nullptr, 256},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    opterr = 0;
    int c;
    while ((c = getopt_long(argc, argv, "a:e:t:m:n:h", long_options, nullptr)) != -1) {
        switch (c) {
            case 'a': result.a_size = bench::parse_size(optarg); break;
            case 'e': result.element_bytes = bench::positive_integer(optarg); break;
            case 't': result.block_size = bench::positive_integer(optarg); break;
            case 'm': result.sm_multiplier = bench::positive_integer(optarg); break;
            case 'n': result.iterations = bench::positive_integer(optarg); break;
            case 256: result.validate = true; break;
            case 257: result.grid_stride = true; break;
            case 'h': result.help = true; break;
            default: throw std::invalid_argument("unknown option or missing option argument");
        }
    }
    if (optind != argc) throw std::invalid_argument("unexpected positional argument");
    if (result.element_bytes != 1 && result.element_bytes != 2 &&
        result.element_bytes != 4 && result.element_bytes != 8)
        throw std::invalid_argument("element bytes must be 1, 2, 4 or 8");
    if (result.a_size % result.element_bytes)
        throw std::invalid_argument("A size must be divisible by element bytes");
    if (result.validate && result.a_size > (size_t{16} << 20))
        throw std::invalid_argument("--validate requires A <= 16 MiB per GPU");
    return result;
}

inline uint64_t grid_size(const Options& options, uint64_t max_threads,
                          uint64_t max_block_x, uint64_t max_grid_x,
                          uint64_t sm_count) {
    if (!options.block_size || options.block_size > max_threads ||
        options.block_size > max_block_x || options.block_size > UINT32_MAX)
        throw std::invalid_argument("block size exceeds GPU 0 hardware limits");
    const uint64_t elements = options.a_size / options.element_bytes;
    if (options.grid_stride) {
        if (!elements || !sm_count || !options.sm_multiplier || !max_grid_x)
            throw std::invalid_argument("grid-stride launch dimensions must be positive");
        if (sm_count > UINT64_MAX / options.sm_multiplier)
            throw std::invalid_argument("SM multiplier grid size overflow");
        const uint64_t needed = elements / options.block_size +
                                (elements % options.block_size != 0);
        return std::min({needed, sm_count * options.sm_multiplier,
                         max_grid_x, uint64_t{UINT32_MAX}});
    }
    // The reference kernel retains strict coverage, even for exact multiples.
    const uint64_t quotient = elements / options.block_size;
    if (quotient == UINT64_MAX)
        throw std::invalid_argument("grid size overflow");
    const uint64_t grid = quotient + 1;
    if (grid > max_grid_x || grid > UINT32_MAX)
        throw std::invalid_argument("grid size exceeds GPU 0 hardware limits");
    if (grid > UINT64_MAX / options.block_size ||
        grid * options.block_size > UINT64_MAX / options.element_bytes)
        throw std::invalid_argument("launch coverage overflow");
    return grid;
}
} // namespace uva_p2p
