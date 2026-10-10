-include config.mk
LDLIBS ?= -lcurand -lnccl
NVCC ?= nvcc --forward-unknown-to-host-compiler
INCLUDE ?= -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) $(LDLIBS) --cudart=shared
BENCH_BIN ?= bin/nccl_swap_bench
# Deliberately independent of MPI/NCCL vendor link flags and LDLIBS.
UVA_BIN ?= bin/uva_swap_bench
UVA_LDFLAGS ?=
UVA_LDLIBS ?= -lcurand
NVSHMEM_BIN ?= bin/nvshmem_swap_bench
NVSHMEM_HOME ?= /usr/local/nvshmem
NVSHMEM_CFLAGS ?= -I$(NVSHMEM_HOME)/include -rdc=true
NVSHMEM_LDFLAGS ?= -L$(NVSHMEM_HOME)/lib
NVSHMEM_LDLIBS ?= -lnvshmem_host -lnvshmem_device -lcurand
NVSHMEM_MPI_LDFLAGS ?= -lmpi

.PHONY: target
target: $(BENCH_BIN)

$(BENCH_BIN): src/nccl_swap_bench.cu include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) -o $@

$(UVA_BIN): src/uva_swap_bench.cu include/uva_p2p_options.hpp include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< $(UVA_LDFLAGS) $(UVA_LDLIBS) --cudart=shared -o $@

$(NVSHMEM_BIN): src/nvshmem_swap_bench.cu include/nvshmem_swap_options.hpp include/bench_options.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $(NVSHMEM_CFLAGS) $< $(NVSHMEM_LDFLAGS) $(NVSHMEM_MPI_LDFLAGS) $(NVSHMEM_LDLIBS) --cudart=shared -o $@

.PHONY: clean
clean:
	$(RM) $(BENCH_BIN) $(UVA_BIN) $(NVSHMEM_BIN)
