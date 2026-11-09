# Experimental tiled race measurement

Set `QCS_MEASUREMENT_BACKEND=tiled_race` consistently on every rank before
creating the simulator. This is explicitly opt-in. `grouped`, `rank_sampled`,
`tiled_cdf`, and the `auto` selection policy are unchanged.

Each rank builds the existing device-resident tile hierarchy for the effective
distribution, conditioned on earlier measurements. It samples a local index
using domain 0 and constructs an independent exponential clock using domain 1.
The comparison key is `log(-log(U)) - log(local_mass)`, avoiding division
overflow and underflow even for extreme finite positive masses. The minimum
clock selects a rank proportional to its mass; the independent local sample
therefore follows the global Born distribution. Zero-mass ranks are inert.
An all-zero effective distribution fails collectively.

The implementation inherits the tiled CDF weight arithmetic: individual squared
amplitudes and local totals must be representable as finite doubles. Overflowing
weights/totals fail; weights that underflow to zero are zero effective mass.
Log-domain race keys do not repair weight construction underflow or overflow.
Finite random precision also limits probabilities resolvable in practice.

## RNG and lifecycle

The simulator's existing initialization-time rank-zero seed broadcast is the
authoritative seed exchange. The race backend retains an immutable snapshot at
that point; later changes to the private legacy seed field do not reseed it.
Philox4x32-10 takes the public 64-bit measurement counter, the 32-bit rank, and
the 32-bit domain in separate counter lanes, with the master seed as its key.
Internal draws do not advance the public counter. Each successful fresh API
call advances it once; cached calls and failed calls consume no counter value.
The last counter value is usable once, and resets leave the cursor untouched.
The 52-bit midpoint uniform grid is strictly inside `(0,1)`.

The simulator creates a full-size MPI byte datatype and a commutative custom
operation during collective initialization and releases them before MPI
finalization, including setup exception cleanup. The reduction independently
ORs error status and minimizes valid candidates by `(key, rank, uint64 index)`.
Thus a valid winner cannot mask another rank's failure. Invalid effective
states and recoverable local preparation errors fail on every rank before
metadata or counter changes. As with existing backends, fatal CUDA/MPI failures
may require aborting the job.

Projection remains deferred and unnormalized. Only requested fresh logical bits
are recorded; the rest of the sampled basis index is discarded. Existing
logical mapping, cached calls, and classical destination handling are reused.

## Communication and profiling

A normal fresh sample executes exactly **one MPI_Allreduce**, including error
propagation, with no seed or result broadcast. Cached calls execute zero sampling
collectives. This path uses host MPI independently of the configured NCCL buffer
mode. Local hierarchy construction, sampling, two small device-to-host copies,
and stream synchronization precede that Allreduce.

Setup communication is separate: the existing two backend-consistency
Allreduces and existing initialization seed broadcast remain. Datatype and
operation creation are local MPI calls, with no new setup collective. Other
simulator initialization collectives remain outside sampling. Benchmark barriers
and timing aggregation Allreduces are also outside the sampling call. The
existing `tiled_cdf` and `rank_sampled` sampling paths each use a rank exchange
and a result broadcast (MPI or NCCL according to buffer mode).

One collective invocation may involve multiple network rounds. A smaller call
count does not establish a speedup: every supported rank now samples locally,
and host readback and MPI implementation costs can dominate.

## Validation and measurement status

CPU tests run successfully in the implementation environment:

```sh
make -f Makefile.test test-measurement-race test-measurement-backend
```

Coverage includes the Philox known-answer vector, rank/domain separation and
large counters, uneven masses, zero-mass and single-supported-rank candidates,
extreme finite masses, full-width indices, deterministic ties, associative
error propagation, and statistical independence of the winning local sample.

GPU/MPI coverage is wired into the existing private/public integration harness:

```sh
make test-measurement-race-integration RANKS=1
make test-measurement-race-integration RANKS=2
make test-measurement-race-integration RANKS=4
```

It includes uneven correlated distributions, cross-rank agreement, cached calls,
partial-measurement coherence, logical mapping and classical destinations,
counter exhaustion/retry, invalid states on one rank, global zero mass, and
single-supported-rank extreme finite effective masses.

No CUDA toolkit, MPI compiler, or GPU runtime is available in the implementation
environment. GPU compilation, distributed validation, and latency measurements
are **pending**; there are no measured GPU latency results or speedup claims.

On a configured GPU system, compare the same configurations with:

```sh
make benchmark-measurement-backends RANKS=4 \
  BENCH_BACKENDS='tiled_race tiled_cdf rank_sampled'
```

Repeat on available rank/node counts, retain first-call and warmed single-shot
results, and record GPU/network/MPI versions. See
[measurement performance](measurement-performance.md) for benchmark and tracing
instructions. The existing paired benchmark accepts `--backends tiled_race
tiled_cdf rank_sampled`. Build its profile variant and use correlated CUDA/MPI
trace events: `tiled_race.local_gpu_build_and_sample` includes local enqueue and
readback work, `tiled_race.allreduce` brackets communication, and tiled hierarchy
ranges expose kernels. Report GPU execution/readback and MPI wall time separately;
host NVTX enqueue ranges alone are not GPU execution timings.
