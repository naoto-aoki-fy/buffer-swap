#include "bench_options.hpp"
#include <cassert>
#include <functional>
#include <iostream>
#include <vector>

static void rejects(const std::function<void()>& action) {
    bool threw = false;
    try { action(); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

static bench::Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    optind = 0; // Reset GNU getopt between independent test invocations.
    return bench::parse_options(int(args.size()), argv.data());
}

int main() {
    const auto defaults = parse({"bench"});
    assert(defaults.a_size == 34359738368ULL && defaults.block_size == 256);
    assert(defaults.grid_size == 1024 && defaults.iterations == 128);
    assert(!defaults.validate && !defaults.help);
    assert(bench::parse_size("1") == 1);
    assert(bench::parse_size("3K") == 3072);
    assert(bench::parse_size("2m") == 2097152);
    assert(bench::parse_size("32G") == 34359738368ULL);
    assert(bench::parse_size("1k") == 1024);
    assert(bench::parse_size("2M") == 2097152);
    assert(bench::parse_size("1g") == 1073741824ULL);
    assert(bench::positive_integer("18446744073709551615") == UINT64_MAX);
    assert(parse({"bench", "-n", "18446744073709551615"}).iterations == UINT64_MAX);
    assert(parse({"bench", "-a", "18446744073709551608"}).a_size == UINT64_MAX - 7);
    for (const auto* text : {"", "0", "-1", "+2", " 2", "2 ", "1.5G", "1T",
                             "G", "1GB", "18446744073709551616", "18446744073709551615G"})
        rejects([&] { bench::parse_size(text); });
    const auto custom = parse({"bench", "--a-size", "1000", "--block-size=3",
                               "--grid-size", "7", "--iterations=3", "--validate"});
    assert(custom.a_size == 1000 && custom.block_size == 3 && custom.grid_size == 7);
    assert(custom.iterations == 3 && custom.validate);
    const auto short_opts = parse({"bench", "-a", "8", "-b", "1", "-g", "1", "-n", "2"});
    assert(short_opts.a_size == 8 && short_opts.block_size == 1 && short_opts.grid_size == 1);
    assert(parse({"bench", "-h"}).help && parse({"bench", "--help"}).help);
    assert(parse({"bench", "-a", "16M", "--validate"}).validate);
    for (const auto& args : std::vector<std::vector<std::string>>{
            {"bench", "-a"}, {"bench", "-b"}, {"bench", "-g"}, {"bench", "-n"},
            {"bench", "--unknown"}, {"bench", "positional"},
            {"bench", "--b-size", "8"}, {"bench", "--skip-copy-back"},
            {"bench", "--validate=1"}, {"bench", "-n", "0"}, {"bench", "-n", "1G"},
            {"bench", "-n", "18446744073709551616"},
            {"bench", "--validate"}, {"bench", "-a", "16777224", "--validate"},
            {"bench", "-a", "1"}, {"bench", "-a", "1003"},
            {"bench", "-a", "18446744073709551615"}})
        rejects([&] { parse(args); });
    for (const auto* option : {"-b", "-g"}) {
        for (const auto* text : {"0", "-1", "+1", "1K", " 1", "1.5", "2147483648",
                                 "18446744073709551615", "18446744073709551616"})
            rejects([&] { parse({"bench", option, text}); });
        assert(parse({"bench", option, "2147483647"}).a_size == defaults.a_size);
    }
    bench::validate_launch_limits(defaults, 256, 256, 1024);
    rejects([&] { bench::validate_launch_limits(defaults, 128, 256, 1024); });
    rejects([&] { bench::validate_launch_limits(defaults, 256, 128, 1024); });
    rejects([&] { bench::validate_launch_limits(defaults, 256, 256, 512); });
    std::cout << "option parsing tests passed\n";
}
