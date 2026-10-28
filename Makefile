-include config.mk
LDLIBS ?= -lcurand -lnccl -ldl
NVCC ?= nvcc --forward-unknown-to-host-compiler
INCLUDE ?= -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) -L./lib $(LDLIBS) --cudart=shared
QCS_BIN ?= bin/qcs
LIBQCS_SO ?= lib/libqcs.so
MEASUREMENT_TEST_BIN ?= bin/test-measurement
MPIEXEC ?= mpirun
TEST_NP ?= 1

.PHONY: target
target: $(QCS_BIN)

.PHONY: sharedlibrary
sharedlibrary: $(LIBQCS_SO)

$(QCS_BIN): src/qcs_main.cpp src/qcs_args.c src/qcs_args.h include/qcs.h $(LIBQCS_SO)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) src/qcs_main.cpp src/qcs_args.c -lqcs $(NVCC_LDFLAGS) -o $@

$(LIBQCS_SO): src/qcs.cu include/qcs.h
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) -fPIC $(NVCC_LDFLAGS) -shared src/qcs.cu -o $@

# Includes qcs.cu directly to exercise internal N=3 template instantiations.
$(MEASUREMENT_TEST_BIN): tests/measurement.cu src/qcs.cu include/qcs.h
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) tests/measurement.cu $(NVCC_LDFLAGS) -o $@

.PHONY: test-measurement test-measurement-host
test-measurement: $(MEASUREMENT_TEST_BIN)
	$(MPIEXEC) -np $(TEST_NP) ./$(MEASUREMENT_TEST_BIN)

test-measurement-host:
	python3 tests/measurement_host.py

.PHONY: gengetopt
gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

.PHONY: clean
clean:
	$(RM) $(QCS_BIN) $(LIBQCS_SO) $(MEASUREMENT_TEST_BIN)
