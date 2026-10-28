#include "uva_p2p_options.hpp"
#include <cassert>
#include <functional>
#include <iostream>
#include <vector>

static void rejects(const std::function<void()>& action) {
    bool threw = false;
    try { action(); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

static uva_p2p::Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    optind = 0;
    return uva_p2p::parse_options(int(args.size()), argv.data());
}

int main() {
    const auto defaults = parse({"bench"});
    assert(defaults.a_size == 34359738368ULL && defaults.element_bytes == 8);
    assert(defaults.block_size == 256 && defaults.iterations == 128);
    assert(!defaults.validate && !defaults.help && !defaults.grid_stride);
    assert(defaults.sm_multiplier == 4);
    assert(parse({"bench", "--grid-stride", "--sm-multiplier=2"}).grid_stride);
    assert(parse({"bench", "--sm-multiplier=2"}).sm_multiplier == 2);
    assert(parse({"bench", "-m", "3"}).sm_multiplier == 3);
    assert(uva_p2p::grid_size(defaults, 1024, 1024, INT32_MAX, 80) == 16777217);
    assert(parse({"bench", "--help"}).help);
    const auto custom = parse({"bench", "--a-size=2k", "-e", "2", "--block-size", "32",
                               "--iterations=3", "--validate"});
    assert(custom.a_size == 2048 && custom.element_bytes == 2);
    assert(custom.block_size == 32 && custom.iterations == 3 && custom.validate);
    for (uint64_t element : {1, 2, 4, 8}) {
        for (uint64_t count : {1, 255, 256, 257, 512}) {
            auto options = defaults;
            options.element_bytes = element;
            options.a_size = count * element;
            const uint64_t grid = uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 80);
            assert(grid == count / 256 + 1);
            assert(options.block_size * grid * element > options.a_size);
        }
        assert(parse({"bench", "-a", std::to_string(element), "-e",
                      std::to_string(element), "--validate"}).element_bytes == element);
    }
    for (const auto& args : std::vector<std::vector<std::string>>{
            {"bench", "-a"}, {"bench", "-e"}, {"bench", "-m"},
            {"bench", "-m", "0"}, {"bench", "-m", "-1"},
            {"bench", "-m", "18446744073709551616"},
            {"bench", "--grid-stride=1"}, {"bench", "--unknown"},
            {"bench", "positional"}, {"bench", "-a", "0"}, {"bench", "-a", "1GB"},
            {"bench", "-a", "18446744073709551615G"},
            {"bench", "-a", "18446744073709551616"},
            {"bench", "-a", "7"}, {"bench", "-e", "3"}, {"bench", "-e", "16"},
            {"bench", "-e", "0"}, {"bench", "-t", "0"}, {"bench", "-t", "-1"},
            {"bench", "-n", "0"}, {"bench", "-n", "1G"}, {"bench", "--validate"}})
        rejects([&] { parse(args); });
    auto options = defaults;
    options.block_size = 1025;
    rejects([&] { uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 80); });
    options.block_size = 512;
    rejects([&] { uva_p2p::grid_size(options, 1024, 256, INT32_MAX, 80); });
    options = defaults;
    rejects([&] { uva_p2p::grid_size(options, 1024, 1024, 1, 80); });
    options.element_bytes = 1;
    options.a_size = UINT64_MAX;
    options.block_size = 1;
    rejects([&] { uva_p2p::grid_size(options, 1024, 1024, UINT64_MAX, 80); });
    // Grid fits, but coverage in bytes would wrap at 2^64.
    options.element_bytes = 8;
    options.a_size = UINT64_MAX - 7;
    options.block_size = UINT32_MAX;
    rejects([&] { uva_p2p::grid_size(options, UINT64_MAX, UINT64_MAX, UINT32_MAX, 80); });
    options = defaults;
    options.grid_stride = true;
    assert(uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 80) == 320);
    options.sm_multiplier = 2;
    assert(uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 80) == 160);
    assert(uva_p2p::grid_size(options, 1024, 1024, 7, 80) == 7);
    for (uint64_t element : {1, 2, 4, 8}) {
        options.element_bytes = element;
        for (uint64_t count : {1, 255, 256, 257, 512, 513, 1023, 1024, 1025, 4097}) {
            options.a_size = count * element;
            // Two SMs at two blocks per SM: cover small buffers, then reuse threads.
            const uint64_t grid = uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 2);
            const uint64_t expected = count <= 256 ? 1 : count <= 512 ? 2 :
                                      count <= 768 ? 3 : 4;
            assert(grid == expected);
        }
    }
    options = defaults;
    options.grid_stride = true;
    rejects([&] { uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 0); });
    rejects([&] { uva_p2p::grid_size(options, 1024, 1024, 0, 80); });
    options.sm_multiplier = 0;
    rejects([&] { uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 80); });
    options.sm_multiplier = UINT64_MAX;
    rejects([&] { uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 2); });
    // Ceiling division must not overflow for the largest element count.
    options.sm_multiplier = 4;
    options.element_bytes = 1;
    options.a_size = UINT64_MAX;
    assert(uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 80) == 320);
    options.sm_multiplier = UINT64_MAX;
    assert(uva_p2p::grid_size(options, 1024, 1024, UINT64_MAX, 1) == UINT32_MAX);
    options.block_size = 1025;
    rejects([&] { uva_p2p::grid_size(options, 1024, 1024, INT32_MAX, 80); });
    std::cout << "UVA P2P option and grid tests passed\n";
}
