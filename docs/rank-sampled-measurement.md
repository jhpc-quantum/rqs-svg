# Rank-sampled distributed measurement

Set `QCS_MEASUREMENT_BACKEND=rank_sampled` on every MPI rank before creating the
simulator. `grouped` is the default. Unknown values and inconsistent selections
are rejected collectively during setup. For example:

```sh
QCS_MEASUREMENT_BACKEND=rank_sampled mpirun -np 4 ./bin/measurement_integration_test
```

Both multi-qubit APIs and single-qubit measurement use the selected backend.
Caller ordering, classical destinations, and argument checks are unchanged.
There is no automatic selection or cuStateVec dependency.

The backend computes effective probability weights in contiguous 4096-amplitude
chunks with CUDA block reductions. Previously measured bits are translated using
the current logical-to-physical mapping: local conditions filter amplitudes and
global conditions exclude incompatible ranks without modifying the state.
Host chunk weights use O(local state size / 4096) storage; rank weights use
O(rank count). No histogram of requested outcomes is allocated.

An MPI all-gather exchanges rank masses. Rank zero broadcasts one Philox uniform
draw using the existing seed/counter mapping. The rank CDF residual selects a
chunk on the chosen rank; a CUDA scan of that chunk samples a local basis index.
The selected rank broadcasts the index and all ranks extract requested physical
bits into caller order. Communication uses three MPI collectives per call with
fresh targets, independent of target count. Already recorded targets need no
communication or random draw. Invalid weights/norms leave metadata and the RNG
cursor unchanged. The last counter value can be used once before exhaustion.

The selected rank is only a sampling device. Only newly requested logical bits
are added to measured masks, values, lists, and pending projection metadata.
Unmeasured global bits retain coherence across ranks. Existing gate projection
and branch normalization conventions remain unchanged: kept amplitudes are not
renormalized when projections materialize.

## Coverage and limitations

Run the existing public integration suite with each backend and 1, 2, 4, or 8
MPI ranks. It covers full and partial requests, reordered and noncontiguous
subsets, classical destinations, reversed mapping, deterministic and biased
entangled states, repeated measurements, gates after measurement, projected
state amplitudes, and cross-rank agreement. A product-state regression verifies
that measuring the local zero in `|+>_global tensor |0>_local` preserves global
Pauli X expectation one. The private integration executable additionally checks
one-draw metadata updates, counter exhaustion, and zero/NaN/infinite norms.

No builds, tests, or validation commands were run for this implementation, as
requested. CUDA/MPI correctness and performance remain unverified. This initial
implementation allocates temporary storage per fresh call and copies chunk
weights to the host; large-state memory traffic, allocations, and MPI latency
may limit performance. It does not promise improvement over the grouped backend.

See [performance benchmarking and profiling](measurement-performance.md) for
paired backend workloads, GB200 NVL4 launch commands, and timing methodology.
