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
    size_t b_size = size_t{1} << 30;
    uint64_t iterations = 128;
    bool copy_back = true;
    bool validate = false;
    bool help = false;
    bool blocking_stream_sync = false, blocking_event_sync = false;
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

inline Options parse_options(int argc, char** argv) {
    Options result;
    const option long_options[] = {
        {"a-size", required_argument, nullptr, 'a'},
        {"b-size", required_argument, nullptr, 'b'},
        {"iterations", required_argument, nullptr, 'n'},
        {"skip-copy-back", no_argument, nullptr, 256},
        {"validate", no_argument, nullptr, 257},
        {"blocking-stream-sync", no_argument, nullptr, 258},
        {"blocking-event-sync", no_argument, nullptr, 259},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    opterr = 0;
    int c;
    while ((c = getopt_long(argc, argv, "a:b:n:h", long_options, nullptr)) != -1) {
        switch (c) {
            case 'a': result.a_size = parse_size(optarg); break;
            case 'b': result.b_size = parse_size(optarg); break;
            case 'n': result.iterations = positive_integer(optarg); break;
            case 256: result.copy_back = false; break;
            case 257: result.validate = true; break;
            case 258: result.blocking_stream_sync = true; break;
            case 259: result.blocking_event_sync = true; break;
            case 'h': result.help = true; break;
            default: throw std::invalid_argument("unknown option or missing option argument");
        }
    }
    if (optind != argc) throw std::invalid_argument("unexpected positional argument");
    if (result.a_size > std::numeric_limits<size_t>::max() - result.b_size)
        throw std::invalid_argument("combined buffer size overflow");
    if (result.validate && (result.a_size > (size_t{16} << 20) ||
                            result.b_size > (size_t{16} << 20)))
        throw std::invalid_argument("--validate requires A and B each <= 16 MiB");
    return result;
}
} // namespace bench
