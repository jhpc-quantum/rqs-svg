# Measurement baseline (2026-11-09)

## Scope and reproducibility

Reviewed local `origin=https://github.com/jhpc-quantum/rqs-svg.git`, branch `main`,
commit `b7891880b5f4a18e01b9d73c2a875dcde90d24ce`, initially clean.
No remote freshness claim: this is the checked-out revision, not a comparison to
GitHub HEAD. Production algorithms were not modified. Test repair commit
`3055879` only narrows a host test's extracted source block.

Source of truth: `src/qcs.cu` (setup, projection, dispatch, sampling and public
APIs), `measurement_backend.hpp`, `measurement_reduce.cuh`,
`measurement_tiled_cdf.cuh`, `measurement_profile.hpp`, `Makefile*`,
`tests/measurement_{backend_test,integration_test,integration_private,harness,
sampling_cases,reduce_test,tiled_cdf_test,backend_benchmark}.*` and measurement
scripts. Reviewed build, performance, rank-sampled, tiled CDF, primitive and NCCL
guides. The performance guide's older MPI/owner-readback table contradicted the
current NCCL helpers; that table is corrected with this report.

## Environment and configuration

Verified host: `z170-1`, Rocky Linux 9.8, x86_64 Linux
`5.14.0-687.30.1.el9_8.x86_64`; Intel Core i7-6700K @ 4 GHz,
4 cores/8 threads, 33,317,560,320 bytes RAM. GCC/G++
`11.5.0 20240719 (Red Hat 11.5.0-14)`, GNU Make 4.3, Python 3.9.25.
Host tests use `g++ -std=c++17 -Wall -Wextra -Werror`.
No `config.mk` is present. CUDA build defaults expand to:

```sh
nvcc --forward-unknown-to-host-compiler -Wformat=2 -I./include -O3 -rdynamic \
  -std=c++17 -Wno-deprecated-gpu-targets -DQCS_NCCL_BUFFER_MODE=2 \
  tests/measurement_backend_benchmark.cu -o bin/measurement_backend_benchmark \
  -L./lib -lcurand -lnccl -ldl --cudart=shared
```

Vendor include/link and GPU architecture flags are unset; NVSHMEM defaults to 0.
Mode 2 requires NCCL >=2.31.2 and registration support. No CUDA binary was built.
`nvcc`, `nvidia-smi`, `nsys`, `ncu`, `compute-sanitizer`, `cuda-gdb`, and
`clang++` are absent from PATH; no `/dev/nvidia*` exists. No CUDA/NCCL headers
were found under `/usr/include`, `/usr/local/include`, or `/opt`, nor CUDA/NCCL
libraries in `ldconfig -p`. These checks do not prove absence of every possible
nonstandard installation; no usable GPU toolchain was identified.
Open MPI is installed outside PATH: `/usr/lib64/openmpi/bin/mpiexec` reports
OpenRTE 4.1.1; `mpicxx --showme` selects g++, `/usr/include/openmpi-x86_64`,
and `/usr/lib64/openmpi/lib` with `-lmpi_cxx -lmpi`. Two local processes running
`/bin/hostname` succeeded. This does not validate GPU-aware MPI or NCCL.

## Source-level contract (GPU runtime verification pending)

| Selection | Distribution and draws | Communication per fresh operation |
| --- | --- | --- |
| grouped (also unset default) | Conditional joint bins for groups of up to four requested bits; one Philox counter per group containing fresh bits | One double NCCL AllReduce per fresh group, then GPU sample/readback |
| rank_sampled | Sample rank/chunk/amplitude conditional on all cached bits; commit only requested fresh bits; one counter per API request | One NCCL AllGather and owner-rooted Broadcast |
| tiled_cdf | GPU 4096-amplitude tile masses/CDF, host rank selection, GPU tile/amplitude selection; commit only fresh bits; one counter per API request | One NCCL AllGather and owner-rooted Broadcast |
| auto | Chooses grouped or rank_sampled only, from replicated metadata | Inherits selected path |

Auto requires 1..32 ranks, no measured global bits, at least eight fresh local
bits and sixteen total fresh bits, and at least 2^20 effective local states
(`local_states >> measured_local`, with measured_local in 0..63). It is a
provisional threshold, not a measured crossover. At small test sizes auto cannot
exercise rank sampling. Benchmark CSV labels the selection `auto`, not actual
dispatch, so interpret cases with this policy.

All paths implement Born probabilities of the effective, possibly unnormalized
branch. Rank/tile sampling a full basis index does not collapse unrequested bits:
only logical outcome metadata is committed. Cached results return without draws,
scans or measurement collectives; empty requests likewise do no measurement work.
Logical masks survive permutation changes; controls and outcomes use current
`perm_l2p`, including rank bits. Public multi-measurement validates bounds,
duplicates and output pointers before work; classical-destination measurement
sorts targets by physical locality and restores caller output order.

Projection is deferred: pending masks record branch conditions without normalizing
amplitudes. Materialization zeros incompatible local amplitudes/ranks using the
current mapping. Gates, reset/state-observation paths must preserve this contract;
clearing measurement state materializes before forgetting outcomes, whereas
internal discard explicitly forgets them. Preserve coherence and original branch
amplitude scale, including when mapping changes.

Philox is keyed by seed and lifetime counter, not launch geometry. Rank/tiled
paths use rank zero's exchanged draw even when seeds differ. Grouped samples on
every rank and requires consistent seed/counter and reduced weights; it has no
authoritative outcome broadcast. Successful fresh operations advance counters;
UINT64_MAX is usable once, then exhaustion rejects fresh draws. Cached access
still works. Different grouping/CDF orders need not produce identical seeded
trajectories across backends. Sequential grouped requests can commit earlier
groups before a later group fails; do not assume whole-request rollback.

Numerics: reject negative/nonfinite weights, nonpositive or nonfinite global
norms; zero-mass ranks are legitimate. Strict CDF intervals, endpoint clamping
with nextafter and positive-bin fallback avoid choosing trailing zero mass.
Double summation order differs across algorithms; underflow and loss of tiny
weights remain possible, and overflow is rejected rather than rescaled. Tiled
status records distinguish zero mass from invalid weights/totals/distributions.
No exact real-arithmetic or cross-hardware bitwise equivalence is established.

Setup collectively rejects unknown/inconsistent backend settings with MPI min/max
reductions. All ranks must call collective APIs with identical arguments/order.
Same CUDA stream orders reductions, NCCL, sampling and readback. Inactive ranks
participate with zero contributions. Rank/tile recoverable local failures travel
in summaries, owner failures in broadcast sentinels; metadata/counters commit
after validation. Transport-boundary CUDA/NCCL errors abort communicator/MPI job;
this is not a universal guarantee against arbitrary rank-local allocation failure,
hardware loss or hangs. Runtime collective ordering/counts still require traces.

## Executed checks

```sh
make test-measurement-backend test-measurement-launch test-work-buffer \
  test-swap-launch test-pauli-string test-measurement-race
python3 tests/test_nccl_buffer_setup.py
python3 scripts/prepare_measurement_harness.py --source . --output /tmp/rqs-baseline-constant --metadata constant
python3 scripts/prepare_measurement_harness.py --source . --output /tmp/rqs-baseline-global --metadata global
/usr/lib64/openmpi/bin/mpiexec -n 2 /bin/hostname
make measurement-backend-benchmark
```

Six C++ executables passed. NCCL setup test initially failed to compile because its
extraction included subsequently added backend setup; after the isolated repair,
five unittest methods passed. Both private source preparations passed; measurement
Python scripts passed `ast.parse` syntax checks. Benchmark build stopped with
`make: nvcc: No such file or directory` (make exit 2). No CUDA tests or backend
benchmark ran. For **all four selections**, latency, bandwidth, memory high-water,
scaling, correctness on GPU and profiling results are **unavailable**, not zero.

## Required GPU baseline before optimization

On a configured allocation, archive the exact revision, dirty diff, binary hash,
resolved compiler/build flags, driver/CUDA/NCCL/MPI versions, GPU model/memory,
node topology, rank placement and clocks. Build from scratch when flags change.
Run primitive reducer/CDF tests and both constant/global private integration tests
for every backend at supported 1/2/4/8 ranks. Run grouped diagnostic harness
separately: its probes and RNG assertions are grouped-specific. Existing private
rank/tile cases cover correlated nonuniform sampling, reordered mappings, cached
outcomes, counters above 2^32, exhaustion, zero-mass/invalid-norm and rank-local
failure/retry; public cases cover gates, coherence, resets and classical outputs.
Their presence is verified, not their runtime success. Auto's rank path needs
larger-state integration coverage beyond the small default cases.

```sh
make test-measurement-reduce test-measurement-tiled-cdf
for metadata in constant global; do
  make bin/measurement-harness-$metadata/measurement_integration_test MEASUREMENT_METADATA=$metadata
  for ranks in 1 2 4 8; do
    for backend in grouped rank_sampled tiled_cdf auto; do
      QCS_MEASUREMENT_BACKEND=$backend mpiexec -n "$ranks" \
        ./bin/measurement-harness-$metadata/measurement_integration_test
    done
  done
done
make measurement-backend-benchmark
python3 scripts/benchmark_measurement.py --no-build --ranks 4 --qubits 24 \
  --warmups 5 --repetitions 30 --backends grouped rank_sampled tiled_cdf auto \
  --mpiexec 'mpiexec' --output results/baseline-4-q24
```

Use one rank per GPU and only supported allocations; repeat q20/24/28 and reversed
backend order in separate output directories, then multi-node runs. The benchmark
covers five state families, target widths/layouts, identity/rotated mappings and
none/partial/all cached metadata. Setup is excluded; elapsed host API plus GPU
completion is reduced to rank maximum; raw samples and p10/median/p90 are retained.
Its `first_call_ms` is first per case, not necessarily process-cold. It has no
timed correctness oracle and uses some unnormalized fixtures deliberately.

The runner records revision/diff/config, commands and selected environment, but
not complete tool versions, binary hash or hardware topology; backend overrides
are inferred from filenames/arguments, not explicit command environment records.
Capture these externally. `compare_measurement.py` instead compares a historical
CUB commit and is not the current four-backend baseline. Profile a separate
`measurement-backend-profile` build with Nsight Systems to validate allocation
reuse, copies, stream dependencies, collectives and waits. No performance choice
is justified until actual paired timing and distributed correctness data exist.
