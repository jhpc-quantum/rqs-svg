-include config.mk
LDLIBS ?= -lcurand -lnccl -ldl
NVCC ?= nvcc --forward-unknown-to-host-compiler
CXX ?= g++
INCLUDE ?= -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
# 0 = baseline, 1 = ncclMemAlloc only, 2 = ncclMemAlloc + ncclCommRegister.
NCCL_BUFFER_MODE ?= 2
NVCC_CFLAGS += -DQCS_NCCL_BUFFER_MODE=$(NCCL_BUFFER_MODE)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) -L./lib $(LDLIBS) --cudart=shared
# Set NVSHMEM=1 to enable NVSHMEM registered-PUT exchange.
# NVSHMEM_HOME must point to an installation built with MPI support.
NVSHMEM ?= 0
ifeq ($(NVSHMEM),1)
NVCC_CFLAGS += -DQCS_ENABLE_NVSHMEM -rdc=true
NVCC_LDFLAGS += -lnvshmem_host -lnvshmem_device
ifneq ($(strip $(NVSHMEM_HOME)),)
NVCC_CFLAGS += -I$(NVSHMEM_HOME)/include
NVCC_LDFLAGS += -L$(NVSHMEM_HOME)/lib
endif
endif
QCS_BIN ?= bin/qcs
LIBQCS_SO ?= lib/libqcs.so
# Embed the absolute library directory in the executable's runtime search path.
QCS_RPATH ?= 1
QCS_LDFLAGS =
ifeq ($(QCS_RPATH),1)
QCS_LDFLAGS += -Wl,-rpath,$(abspath $(dir $(LIBQCS_SO)))
endif

.PHONY: target
target: $(QCS_BIN)

.PHONY: sharedlibrary
sharedlibrary: $(LIBQCS_SO)

$(QCS_BIN): src/qcs_main.cpp src/qcs_args.c src/qcs_args.h include/qcs.h $(LIBQCS_SO)
	mkdir -p $(dir $@)
	$(CXX) $(INCLUDE) src/qcs_main.cpp src/qcs_args.c -L$(dir $(LIBQCS_SO)) -lqcs $(QCS_LDFLAGS) -o $@

$(LIBQCS_SO): src/qcs.cu src/measurement_reduce.cuh src/measurement_launch.hpp src/work_buffer.cuh src/work_buffer.hpp src/nccl_buffer.cuh src/swap_launch.hpp include/qcs.h src/pauli_reduce.cuh src/pauli_string.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) -fPIC -shared src/qcs.cu -o $@ $(NVCC_LDFLAGS)

include Makefile.test

.PHONY: gengetopt
gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

.PHONY: clean
clean:
	$(RM) $(QCS_BIN) $(LIBQCS_SO) $(TEST_BINARIES)
