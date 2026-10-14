-include config.mk
LDLIBS ?= -lcurand -lnccl
NVCC ?= nvcc --forward-unknown-to-host-compiler
INCLUDE ?= -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) $(LDLIBS) --cudart=shared
NCCL_BENCH_BIN ?= bin/nccl_swap_bench
UVA_BENCH_BIN ?= bin/uva_swap_bench
NVSHMEM_BENCH_BIN ?= bin/nvshmem_swap_bench

.PHONY: target
target: $(NCCL_BENCH_BIN)

$(NCCL_BENCH_BIN): src/nccl_swap_bench.cu include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) --cudart=shared -o $@

$(UVA_BENCH_BIN): src/uva_swap_bench.cu include/uva_p2p_options.hpp include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) --cudart=shared -o $@

$(NVSHMEM_BENCH_BIN): src/nvshmem_swap_bench.cu include/nvshmem_swap_options.hpp include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $(NVCC_LDFLAGS) --cudart=shared -o $@

.PHONY: clean
clean:
	$(RM) $(NCCL_BENCH_BIN) $(UVA_BENCH_BIN) $(NVSHMEM_BENCH_BIN)
