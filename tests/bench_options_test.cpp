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
    assert(defaults.pipeline_copy_back);
    const auto serial = parse({"bench", "--no-pipeline-copy-back"});
    assert(!serial.pipeline_copy_back && serial.copy_back);
    assert(parse({"bench", "--pipeline-copy-back"}).pipeline_copy_back);
    assert(parse({"bench", "--no-pipeline-copy-back", "--pipeline-copy-back"}).pipeline_copy_back);
    assert(!parse({"bench", "--pipeline-copy-back", "--no-pipeline-copy-back"}).pipeline_copy_back);
    for (const auto* flag : {"--pipeline-copy-back", "--no-pipeline-copy-back"}) {
        const auto skipped = parse({"bench", flag, "--skip-copy-back"});
        assert(!skipped.copy_back);
        assert(skipped.pipeline_copy_back == (std::string(flag) == "--pipeline-copy-back"));
        rejects([&] { parse({"bench", std::string(flag) + "=1"}); });
    }
    assert(defaults.blocking_stream_sync && defaults.blocking_event_sync);
    assert(!defaults.stream_query && !defaults.event_query && !defaults.query_yield);
    assert(parse({"bench", "--stream-query"}).stream_query);
    assert(!parse({"bench", "--stream-query"}).event_query);
    assert(parse({"bench", "--event-query"}).event_query);
    assert(!parse({"bench", "--event-query"}).stream_query);
    const auto polling = parse({"bench", "--stream-query", "--event-query", "--query-yield"});
    assert(polling.stream_query && polling.event_query && polling.query_yield);
    const auto reset = parse({"bench", "--stream-query", "--event-query", "--query-yield",
                             "--no-stream-query", "--no-event-query", "--no-query-yield"});
    assert(!reset.stream_query && !reset.event_query && !reset.query_yield);
    for (const auto* flag : {"--stream-query", "--event-query", "--query-yield",
                             "--no-stream-query", "--no-event-query", "--no-query-yield"})
        rejects([&] { parse({"bench", std::string(flag) + "=1"}); });
    const auto stream_sync = parse({"bench", "--no-blocking-stream-sync"});
    assert(!stream_sync.blocking_stream_sync && stream_sync.blocking_event_sync);
    const auto event_sync = parse({"bench", "--no-blocking-event-sync"});
    assert(event_sync.blocking_stream_sync && !event_sync.blocking_event_sync);
    const auto both_sync = parse({"bench", "--no-blocking-stream-sync", "--no-blocking-event-sync"});
    assert(!both_sync.blocking_stream_sync && !both_sync.blocking_event_sync);
    rejects([&] { parse({"bench", "--no-blocking-stream-sync=1"}); });
    rejects([&] { parse({"bench", "--no-blocking-event-sync=1"}); });
    assert(defaults.a_size == 34359738368ULL && defaults.b_size == 1073741824ULL);
    assert(defaults.iterations == 128 && defaults.copy_back && !defaults.validate);
    assert(bench::parse_size("1") == 1);
    assert(bench::parse_size("3K") == 3072);
    assert(bench::parse_size("2m") == 2097152);
    assert(bench::parse_size("32G") == 34359738368ULL);
    assert(bench::positive_integer("18446744073709551615") == UINT64_MAX);
    for (const auto* text : {"", "0", "-1", "+2", " 2", "2 ", "1.5G", "1T",
                             "G", "1GB", "18446744073709551616", "18446744073709551615G"})
        rejects([&] { bench::parse_size(text); });
    const auto odd = parse({"bench", "--a-size", "1003", "-b", "257",
                            "--iterations=3", "--skip-copy-back", "--validate"});
    assert(odd.a_size == 1003 && odd.b_size == 257 && odd.iterations == 3);
    assert(!odd.copy_back && odd.validate);
    assert(parse({"bench", "-h"}).help);
    for (const auto& args : std::vector<std::vector<std::string>>{
            {"bench", "-a"}, {"bench", "--unknown"}, {"bench", "positional"},
            {"bench", "-n", "0"}, {"bench", "-n", "1G"},
            {"bench", "--validate"},
            {"bench", "-a", "18446744073709551615", "-b", "1"}})
        rejects([&] { parse(args); });
    std::cout << "option parsing tests passed\n";
}
