# Building and runtime setup

[README](../README.md) · [Building](building.md) · [Usage and API](usage.md) · [Advanced configuration](advanced-configuration.md)

Run the commands in these guides from the repository root.

## Prerequisites and build

The project requires NVIDIA's CUDA toolkit as well as NCCL and MPI.

Before building, set up the
[`atlc`](https://github.com/naoto-aoki-fy/atlc)
repository, which is required during the build process.

Define `CFLAGS_VENDOR`, `LDFLAGS_VENDOR`, and `GENCODE_FLAGS` in `config.mk`, which is automatically included by the Makefile:

```make
CFLAGS_VENDOR = -I/foo/bar
LDFLAGS_VENDOR = -L/foo/bar -lfoobar
GENCODE_FLAGS = -gencode=arch=compute_xx,code=sm_xx
```

These options can be obtained using the [`nvccoptions`](https://github.com/naoto-aoki-fy/nvccoptions) utility.

The host compiler (`CXX`, default `g++`) must also find the atlc headers.
If they are outside its default search path, add
`INCLUDE = -I./include -I/path/to/atlc/include` to `config.mk`.

Then, `make` will build `bin/qcs`.

```sh
make
```

The default `NCCL_BUFFER_MODE=2` uses NCCL allocation and explicit registration.
It requires NCCL headers and runtime >= 2.31.2 and NCCL exchange (`NVSHMEM=0`).
Setup automatically sets `NCCL_PXN_DISABLE=1` if unset or empty; explicit non-empty
values are preserved and must equal `1`. To select CUDA allocation or NCCL
allocation without registration, use `make -B NCCL_BUFFER_MODE=0` or
`make -B NCCL_BUFFER_MODE=1`, respectively. Force a rebuild when changing modes.
See [NCCL buffer registration](nccl-buffer-registration.md) for compatibility
restrictions and validation.

## Runtime library search path

By default, `bin/qcs` embeds the absolute directory containing `libqcs.so`
in its runtime library search path using `-Wl,-rpath`. No `LD_LIBRARY_PATH`
setting is needed to locate `libqcs.so`. Other dependencies, such as CUDA,
NCCL, and MPI, must still be available to the runtime loader. Rebuild the
executable if you move the repository.

To disable the embedded search path, rebuild with:

```sh
make clean
make QCS_RPATH=0
```

With this option, use `source ./env.bash` or configure the runtime library
search path yourself. Rebuild the executable when changing `QCS_RPATH`.

For programs without an embedded runtime search path that load or link against
`lib/libqcs.so`, source the provided shell setup file:

```sh
source ./env.bash
```

The setup file appends this repository's absolute `lib` directory path to
`LD_LIBRARY_PATH` and its `include` directory to `CPATH`.

## Shared library

To build RQS-SVG as a shared library at `lib/libqcs.so`:

```sh
make sharedlibrary
```

See [Usage and API](usage.md#using-the-shared-library-from-c) for linking examples.
