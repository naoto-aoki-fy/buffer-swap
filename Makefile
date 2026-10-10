-include config.mk
LDLIBS ?= -lcurand -lnccl
NVCC ?= nvcc --forward-unknown-to-host-compiler
INCLUDE ?= -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) $(LDLIBS) --cudart=shared
BENCH_BIN ?= bin/gpu_comm_bench

.PHONY: target
target: $(BENCH_BIN)

$(BENCH_BIN): src/gpu_comm_bench.cu include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) $(BENCH_BIN)
