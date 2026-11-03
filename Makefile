-include config.mk
LDLIBS ?= -lcurand -lnccl -ldl
NVCC ?= nvcc --forward-unknown-to-host-compiler
CXX ?= g++
INCLUDE ?= -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
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

$(LIBQCS_SO): src/qcs.cu src/measurement_reduce.cuh src/work_buffer.cuh src/work_buffer.hpp include/qcs.h
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) -fPIC -shared src/qcs.cu -o $@ $(NVCC_LDFLAGS)

MEASUREMENT_REDUCE_TEST ?= bin/measurement_reduce_test
$(MEASUREMENT_REDUCE_TEST): tests/measurement_reduce_test.cu src/measurement_reduce.cuh src/work_buffer.hpp
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) tests/measurement_reduce_test.cu -o $@ $(LDFLAGS_VENDOR) --cudart=shared

.PHONY: test-measurement-reduce
test-measurement-reduce: $(MEASUREMENT_REDUCE_TEST)
	./$(MEASUREMENT_REDUCE_TEST)

WORK_BUFFER_TEST ?= bin/work_buffer_test
$(WORK_BUFFER_TEST): tests/work_buffer_test.cpp src/work_buffer.hpp
	mkdir -p $(dir $@)
	$(CXX) -std=c++17 -Wall -Wextra -Werror $< -o $@

.PHONY: test-work-buffer
test-work-buffer: $(WORK_BUFFER_TEST)
	$(WORK_BUFFER_TEST)

# MPI_CXX must use the MPI installation linked into libqcs.
MPI_CXX ?= mpicxx
MEASUREMENT_INTEGRATION_TEST ?= bin/measurement_integration_test
$(MEASUREMENT_INTEGRATION_TEST): tests/measurement_integration_test.cpp include/qcs.h $(LIBQCS_SO)
	mkdir -p $(dir $@)
	$(MPI_CXX) -std=c++17 $(INCLUDE) $< -L$(dir $(LIBQCS_SO)) -lqcs $(QCS_LDFLAGS) -o $@

.PHONY: test-measurement-integration
test-measurement-integration: $(MEASUREMENT_INTEGRATION_TEST)
	./$(MEASUREMENT_INTEGRATION_TEST)

INITIAL_STATE_TEST ?= bin/initial_state_test
$(INITIAL_STATE_TEST): tests/initial_state_test.cu src/qcs.cu src/measurement_reduce.cuh src/work_buffer.cuh src/work_buffer.hpp include/qcs.h
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< -o $@ $(NVCC_LDFLAGS)

.PHONY: test-initial-state
test-initial-state: $(INITIAL_STATE_TEST)
	./$(INITIAL_STATE_TEST)

.PHONY: gengetopt
gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

.PHONY: clean
clean:
	$(RM) $(QCS_BIN) $(LIBQCS_SO) $(MEASUREMENT_REDUCE_TEST) $(MEASUREMENT_INTEGRATION_TEST) $(WORK_BUFFER_TEST) $(INITIAL_STATE_TEST)

# Compile private instrumented source copies, including the recorded CUB tree.
# Vendor flags must include the same MPI headers/libraries as the production build.
MEASUREMENT_METADATA ?= constant
MEASUREMENT_HARNESS_SOURCE ?= $(CURDIR)
MEASUREMENT_HARNESS_DIR ?= bin/measurement-harness-$(MEASUREMENT_METADATA)
MEASUREMENT_HARNESS = $(MEASUREMENT_HARNESS_DIR)/measurement_harness
MEASUREMENT_HARNESS_ARGS ?= regression 12

$(MEASUREMENT_HARNESS_DIR)/qcs.cu: scripts/prepare_measurement_harness.py $(MEASUREMENT_HARNESS_SOURCE)/src/qcs.cu $(wildcard $(MEASUREMENT_HARNESS_SOURCE)/src/measurement_reduce.cuh $(MEASUREMENT_HARNESS_SOURCE)/src/work_buffer.*)
	python3 scripts/prepare_measurement_harness.py --source $(MEASUREMENT_HARNESS_SOURCE) --output $(MEASUREMENT_HARNESS_DIR) --metadata $(MEASUREMENT_METADATA)

$(MEASUREMENT_HARNESS): tests/measurement_harness.cu tests/measurement_sampling_cases.hpp tests/measurement_probe.hpp $(MEASUREMENT_HARNESS_DIR)/qcs.cu
	$(NVCC) $(NVCC_CFLAGS) -I$(abspath $(MEASUREMENT_HARNESS_DIR)) tests/measurement_harness.cu -o $@ $(NVCC_LDFLAGS)

.PHONY: measurement-harness test-measurement-distributed benchmark-measurement
measurement-harness: $(MEASUREMENT_HARNESS)
test-measurement-distributed: $(MEASUREMENT_HARNESS)
	$(MPIEXEC) -n $(RANKS) ./$(MEASUREMENT_HARNESS) $(MEASUREMENT_HARNESS_ARGS)
benchmark-measurement: $(MEASUREMENT_HARNESS)
	$(MPIEXEC) -n $(RANKS) ./$(MEASUREMENT_HARNESS) benchmark $(BENCH_QUBITS) $(BENCH_WARMUPS) $(BENCH_REPETITIONS)

MPIEXEC ?= mpiexec
RANKS ?= 1
BENCH_QUBITS ?= 12
BENCH_WARMUPS ?= 5
BENCH_REPETITIONS ?= 30

$(MEASUREMENT_HARNESS_DIR)/measurement_integration_test: tests/measurement_integration_private.cu tests/measurement_integration_test.cpp tests/measurement_probe.hpp $(MEASUREMENT_HARNESS_DIR)/qcs.cu
	$(NVCC) $(NVCC_CFLAGS) -I$(abspath $(MEASUREMENT_HARNESS_DIR)) tests/measurement_integration_private.cu -o $@ $(NVCC_LDFLAGS)

.PHONY: test-measurement-metadata
test-measurement-metadata: $(MEASUREMENT_HARNESS_DIR)/measurement_integration_test $(MEASUREMENT_HARNESS)
	$(MPIEXEC) -n $(RANKS) ./$(MEASUREMENT_HARNESS_DIR)/measurement_integration_test
	$(MPIEXEC) -n $(RANKS) ./$(MEASUREMENT_HARNESS) regression 12
