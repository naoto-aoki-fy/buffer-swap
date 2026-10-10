#pragma once

#include <getopt.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace bench {
struct Options {
    size_t a_size = size_t{32} << 30;
    unsigned int block_size = 256;
    unsigned int grid_size = 1024;
    uint64_t iterations = 128;
    bool validate = false;
    bool help = false;
};

// Strict unsigned decimal parsing: reject signs, whitespace and overflow.
inline uint64_t positive_integer(const std::string& text) {
    uint64_t value = 0;
    if (text.empty()) throw std::invalid_argument("empty number");
    for (char c : text) {
        if (c < '0' || c > '9' ||
            value > (UINT64_MAX - static_cast<unsigned>(c - '0')) / 10)
            throw std::invalid_argument("invalid or overflowing number: " + text);
        value = value * 10 + static_cast<unsigned>(c - '0');
    }
    if (!value) throw std::invalid_argument("number must be positive: " + text);
    return value;
}

inline size_t parse_size(std::string text) {
    uint64_t factor = 1;
    if (!text.empty()) {
        switch (text.back()) {
            case 'k': case 'K': factor = uint64_t{1} << 10; break;
            case 'm': case 'M': factor = uint64_t{1} << 20; break;
            case 'g': case 'G': factor = uint64_t{1} << 30; break;
        }
        if (factor != 1) text.pop_back();
    }
    const uint64_t value = positive_integer(text);
    if (value > std::numeric_limits<size_t>::max() / factor)
        throw std::invalid_argument("buffer size overflow");
    return static_cast<size_t>(value * factor);
}

inline unsigned int launch_dimension(const std::string& text) {
    const uint64_t value = positive_integer(text);
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("launch dimension exceeds CUDA integer limits: " + text);
    return static_cast<unsigned int>(value);
}

// Kept independent of CUDA headers so argument/limit checks are host-testable.
inline void validate_launch_limits(const Options& options, int max_threads,
                                  int max_block_x, int max_grid_x) {
    if (options.block_size > static_cast<unsigned int>(max_threads) ||
        options.block_size > static_cast<unsigned int>(max_block_x))
        throw std::invalid_argument("block size exceeds GPU 0/kernel thread limits");
    if (options.grid_size > static_cast<unsigned int>(max_grid_x))
        throw std::invalid_argument("grid size exceeds GPU 0 grid limit");
}

inline Options parse_options(int argc, char** argv) {
    Options result;
    const option long_options[] = {
        {"a-size", required_argument, nullptr, 'a'},
        {"block-size", required_argument, nullptr, 'b'},
        {"grid-size", required_argument, nullptr, 'g'},
        {"iterations", required_argument, nullptr, 'n'},
        {"validate", no_argument, nullptr, 257},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    opterr = 0;
    int c;
    while ((c = getopt_long(argc, argv, "a:b:g:n:h", long_options, nullptr)) != -1) {
        switch (c) {
            case 'a': result.a_size = parse_size(optarg); break;
            case 'b': result.block_size = launch_dimension(optarg); break;
            case 'g': result.grid_size = launch_dimension(optarg); break;
            case 'n': result.iterations = positive_integer(optarg); break;
            case 257: result.validate = true; break;
            case 'h': result.help = true; break;
            default: throw std::invalid_argument("unknown option or missing option argument");
        }
    }
    if (optind != argc) throw std::invalid_argument("unexpected positional argument");
    if (result.a_size % sizeof(uint64_t) != 0)
        throw std::invalid_argument("buffer A size must be a multiple of 8 bytes");
    if (result.validate && result.a_size > (size_t{16} << 20))
        throw std::invalid_argument("--validate requires buffer A <= 16 MiB per GPU");
    return result;
}
} // namespace bench
