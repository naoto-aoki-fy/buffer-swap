-include config.mk
LDLIBS ?= -lcurand -lnccl
NVCC ?= nvcc --forward-unknown-to-host-compiler
INCLUDE ?= -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) $(LDLIBS) --cudart=shared
NCCL_BENCH_BIN ?= bin/nccl_swap_bench
NCCL_SKEW_TEST_BIN ?= bin/nccl_swap_skew_test
UVA_BENCH_BIN ?= bin/uva_swap_bench
NVSHMEM_BENCH_BIN ?= bin/nvshmem_swap_bench
NVSHMEM_SKEW_TEST_BIN ?= bin/nvshmem_swap_skew_test

.PHONY: target
target: $(NCCL_BENCH_BIN) $(UVA_BENCH_BIN) $(NVSHMEM_BENCH_BIN)

$(NCCL_BENCH_BIN): src/nccl_swap_bench.cu include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) --cudart=shared -o $@

# Test-only build: delay rank submissions and copies in the real NCCL exchange.
$(NCCL_SKEW_TEST_BIN): tests/nccl_swap_skew_test.cu src/nccl_swap_bench.cu include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) --cudart=shared -o $@

$(UVA_BENCH_BIN): src/uva_swap_bench.cu include/uva_p2p_options.hpp include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) --cudart=shared -o $@

$(NVSHMEM_BENCH_BIN): src/nvshmem_swap_bench.cu include/nvshmem_swap_options.hpp include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) -rdc=true $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) --cudart=shared -lnvshmem_host -lnvshmem_device -o $@

# Test-only build: inject rank/block skew around the real kernel's NVSHMEM calls.
$(NVSHMEM_SKEW_TEST_BIN): tests/nvshmem_swap_skew_test.cu src/nvshmem_swap_bench.cu include/nvshmem_swap_options.hpp include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) -rdc=true $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) --cudart=shared -lnvshmem_host -lnvshmem_device -o $@

.PHONY: clean
clean:
	$(RM) $(NCCL_SKEW_TEST_BIN) $(NCCL_BENCH_BIN) $(UVA_BENCH_BIN) $(NVSHMEM_BENCH_BIN) $(NVSHMEM_SKEW_TEST_BIN)
