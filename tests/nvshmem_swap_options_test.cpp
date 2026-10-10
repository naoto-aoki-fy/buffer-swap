#include "nvshmem_swap_options.hpp"
#include <cassert>
#include <functional>
#include <iostream>
#include <vector>

static void rejects(const std::function<void()>& action) {
    bool threw = false;
    try { action(); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}
static nvshmem_swap::Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    optind = 0;
    return nvshmem_swap::parse_options(int(args.size()), argv.data());
}
static nvshmem_swap::Layout layout(const nvshmem_swap::Options& o, uint64_t sms = 112) {
    return nvshmem_swap::layout(o, sms, 1024, 1024, INT32_MAX);
}
int main() {
    const auto defaults = parse({"bench"});
    assert(defaults.a_size == 34359738368ULL && defaults.element_bytes == 8);
    assert(defaults.block_size == 256 && defaults.sm_multiplier == 1 && defaults.iterations == 128);
    assert(!defaults.help && !defaults.validate);
    const auto d = layout(defaults);
    assert(d.grid_blocks == 112 && d.chunk_bytes == 2048 && d.b_bytes == 229376);
    assert(d.signal_bytes == 896 && d.rounds == 149797);
    assert(parse({"bench", "-h"}).help && parse({"bench", "--help"}).help);
    const auto custom = parse({"bench", "--a-size=2k", "--element-bytes=2", "-t", "32",
                               "--sm-multiplier", "2", "-n", "3", "--validate"});
    assert(custom.a_size == 2048 && custom.element_bytes == 2 && custom.block_size == 32);
    assert(custom.sm_multiplier == 2 && custom.iterations == 3 && custom.validate);
    assert(parse({"bench", "-a", "1M"}).a_size == 1048576);
    assert(parse({"bench", "-a", "1g"}).a_size == 1073741824);
    // Arbitrary byte counts, including sub-element tails, are legal.
    for (uint64_t width : {1, 2, 4, 8}) {
        for (uint64_t threads : {1, 32, 128, 256}) {
            for (uint64_t mult : {1, 2, 3}) {
                auto o = defaults;
                o.element_bytes = width; o.block_size = threads; o.sm_multiplier = mult;
                const uint64_t chunk = width * threads, stride = chunk * 4 * mult;
                for (uint64_t bytes : {uint64_t{1}, chunk, chunk + 1, stride - 1, stride, stride + 3}) {
                    o.a_size = bytes;
                    const auto l = layout(o, 4);
                    assert(l.chunk_bytes == chunk && l.b_bytes == stride);
                    assert(l.rounds == (bytes - 1) / stride + 1);
                    assert(l.heap_required == bytes + stride + l.signal_bytes + (1ULL << 20));
                    // Independent ownership check: each byte belongs to exactly
                    // one round/block, with a bounded block-local B address.
                    for (uint64_t byte = 0; byte < bytes; ++byte) {
                        const auto block = (byte / chunk) % l.grid_blocks;
                        const auto round = (byte / chunk) / l.grid_blocks;
                        const auto staging = block * chunk + byte % chunk;
                        assert(round < l.rounds && staging < l.b_bytes);
                    }
                }
                assert(parse({"bench", "-a", "3", "-e", std::to_string(width)}).a_size == 3);
            }
        }
    }
    for (const auto& args : std::vector<std::vector<std::string>>{
        {"bench", "-a"}, {"bench", "-m"}, {"bench", "--unknown"}, {"bench", "position"},
        {"bench", "-a", "0"}, {"bench", "-a", "1GB"}, {"bench", "-a", "1.5G"},
        {"bench", "-a", "-1"}, {"bench", "-a", "+1"}, {"bench", "-a", " 1"},
        {"bench", "-a", "18446744073709551615G"}, {"bench", "-a", "18446744073709551616"},
        {"bench", "-e", "3"}, {"bench", "-e", "16"}, {"bench", "-e", "0"},
        {"bench", "-t", "0"}, {"bench", "-m", "0"}, {"bench", "-m", "-2"},
        {"bench", "-n", "0"}, {"bench", "-n", "1G"}, {"bench", "--validate"}})
        rejects([&] { parse(args); });
    auto o = defaults;
    o.block_size = 1025; rejects([&] { layout(o); });
    o = defaults;
    rejects([&] { nvshmem_swap::layout(o, 112, 1024, 128, INT32_MAX); });
    rejects([&] { nvshmem_swap::layout(o, 112, 1024, 1024, 111); });
    rejects([&] { layout(o, 0); });
    o.sm_multiplier = UINT64_MAX; rejects([&] { layout(o); });
    o = defaults; o.iterations = UINT64_MAX; rejects([&] { layout(o); });
    o.iterations = UINT64_MAX / d.rounds; rejects([&] { layout(o); });
    o = defaults; o.a_size = UINT64_MAX; rejects([&] { layout(o); });
    rejects([&] { nvshmem_swap::multiply(UINT64_MAX, 8); });
    rejects([&] { nvshmem_swap::add(UINT64_MAX, 1); });
    nvshmem_swap::check_capacity(d, uint64_t{33} << 30, uint64_t{40} << 30);
    rejects([&] { nvshmem_swap::check_capacity(d, uint64_t{32} << 30, uint64_t{40} << 30); });
    rejects([&] { nvshmem_swap::check_capacity(d, uint64_t{33} << 30, uint64_t{32} << 30); });
    std::cout << "NVSHMEM option, layout, overflow and capacity tests passed\n";
}
