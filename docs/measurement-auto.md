# Automatic measurement dispatch

## Decision and evidence status

No validated GPU crossover measurements are available. The current environment
has neither `nvcc` nor `nvidia-smi`; the repository's
[performance audit](measurement-performance.md) records no hardware samples or
profiles. Synthetic runner fixtures are not performance evidence. Consequently,
this change isolates deterministic metadata construction and selection without
retuning the provisional thresholds or enabling tiled CDF automatically.
There is no measured crossover point and no demonstrated speedup for this policy.

## Enabled rules

An unset backend still means `grouped`. Explicit backend selections still bypass
auto selection, including `tiled_cdf`, `tiled_race`, and `tiled_race_nccl`.
For opt-in `auto`, use `rank_sampled` only when all these conditions hold:

- Rank count is between 1 and 32 inclusive.
- No global physical bit was previously measured.
- At least 8 fresh local targets and at least 16 total fresh targets remain.
- `local_states >> measured_local` is at least 2^20.

Otherwise use `grouped`. These are retained compatibility thresholds, **not
measured crossover points**. Invalid negative or oversized count inputs to the
host selector also fall back to grouped. All-cached, empty, global-only, small,
and previously globally conditioned requests select grouped. No new fast path
is introduced: grouped retains its cached, global-only and inactive-rank paths.
An inactive rank must never choose a different backend from active ranks.

Both public single-target and multi-target measurement paths use the same auto
selector. Metadata construction converts requested and previously measured logical
bits through the replicated physical mapping. Physical masks remove cached targets
and count each target once. The public API's existing target validation still
applies. Only counts, local statevector size and rank count reach the selector.
The host conversion tests cover low, high, and mixed physical controls; equal
counts deliberately produce equal decisions regardless of physical positions.

No new collective is performed. Consistency relies on the existing contract that
all ranks share the mapping, measurement mask, request, dimensions and backend
configuration. Existing setup collectives validate backend configuration. Outcomes,
probability masses, active-rank compatibility, RNG state and runtime observations
are excluded from dispatch inputs.

## Proposed evidence-driven successor (not enabled)

Use the [paired benchmark and trace procedure](measurement-performance.md) to
validate optimized grouped, rank-sampled and tiled-CDF implementations first.
Do not enable tiled CDF until its numerical correctness and full API performance
are validated in the proposed workload region. Collect repeated rank-maximum API
latencies, first-call costs, percentile spread and stage traces across multiple GPU
architectures, single-node and multi-node placements, both API forms, mappings,
state families, target widths, and cached/control patterns. Separate fitting and
held-out workloads; repeat backend order to detect bias.

A future replicated workload descriptor should preserve ordered physical targets
and measured local masks in addition to counts. Estimate successive grouped
reduction domains using the actual group boundaries (up to four request entries,
including cached entries), not simply ceil(fresh targets / 4). Account for which
physical bits have become controls at each step: the effective count L/2^m alone
does not describe memory access or launch costs. Extract actual chunk/tile counts
from implementation constants, then compare complete grouped reduction plus
collective costs against rank-summary/chunk transfer and tiled scan/CDF costs.
Global-only cases and globally conditioned inactive ranks need separate evidence.

Rank count is available now. GPU architecture and communication topology must only
enter a future policy through a reliably replicated configuration or an existing
validated replicated descriptor. Do not infer topology from rank-local device
queries or add dispatch collectives. Unknown topology/architecture and unmeasured
regions should use grouped in a newly fitted policy. Require a repeatable margin
beyond noise before switching a region, and document measured crossovers, sample
counts, hardware and uncertainty. No coefficients, tile thresholds, group-count
thresholds or topology-specific choices are proposed numerically without data.

## Validation and coverage

`make -B test-measurement-backend test-measurement-launch test-work-buffer` passed.
Host dispatch coverage includes exact legacy state-size, fresh-count and rank-count
boundaries, cached/empty/global-only requests, physical local/global classification,
low/high controls and invalid metadata. Environment mode parsing covers every
explicit backend. The private GPU integration test adds a dispatch invariance check
under rank-dependent RNG changes and recorded-value changes; it has not executed.

`make bin/measurement-harness-constant/measurement_integration_test` remains blocked
before compilation by the pre-existing harness copier mismatch for the
`first_stage<N>` launch. CUDA compilation and GPU execution are also unavailable.
Once repaired on GPU hardware, run the integration matrix in the performance report
for explicit and automatic backends. Its small states exercise grouped auto only;
large states with sufficient fresh targets are additionally required to exercise
the auto rank-sampled branch. Production CUDA integration remains uncompiled here.
