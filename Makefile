-include config.mk
LDLIBS ?= -lcurand -lnccl -ldl
NVCC ?= nvcc --forward-unknown-to-host-compiler
INCLUDE ?= -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) -L./lib $(LDLIBS) --cudart=shared
# Set NVSHMEM=1 to compile and enable the NVSHMEM exchange path.
# NVSHMEM_HOME must point to an installation built with MPI support.
NVSHMEM ?= 0
ifeq ($(NVSHMEM),1)
NVCC_CFLAGS += -DQCS_ENABLE_NVSHMEM -rdc=true
NVCC_LDFLAGS += -lnvshmem_host -lnvshmem_device
endif
QCS_BIN ?= bin/qcs
LIBQCS_SO ?= lib/libqcs.so

.PHONY: target
target: $(QCS_BIN)

.PHONY: sharedlibrary
sharedlibrary: $(LIBQCS_SO)

$(QCS_BIN): src/qcs_main.cpp src/qcs_args.c src/qcs_args.h include/qcs.h $(LIBQCS_SO)
	mkdir -p $(dir $@)
	$(NVCC) src/qcs_main.cpp src/qcs_args.c -L./lib -lqcs -o $@

$(LIBQCS_SO): src/qcs.cu include/qcs.h
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) -fPIC -shared src/qcs.cu -o $@ $(NVCC_LDFLAGS)

.PHONY: gengetopt
gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

.PHONY: clean
clean:
	$(RM) $(QCS_BIN) $(LIBQCS_SO)
