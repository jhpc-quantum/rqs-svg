# Inactive ranks in distributed measurement

`rank_sampled` and `tiled_cdf` skip state and chunk/tile processing when cached
global outcomes exclude the current rank. They still validate recoverable local
failures and contribute zero mass in the existing rank exchange. The device RNG
draw remains available on rank zero even when it is excluded. Collective order,
counter commits, and automatic dispatch are unchanged.

Inactive `rank_sampled` calls transfer one uniform instead of chunk weights.
Inactive tiled builds initialize one summary without reducing tiles or building
a CDF. Existing workspace capacity is retained; an inactive first build uses
minimal storage and a later active build grows it normally.

Run metadata integration tests for both explicit backends with 2, 4, and 8 ranks.
The cached-global tests vary the eligible subset down to one rank, exercise
excluded-rank failures and retry, and reuse workspaces across eligibility changes.
Run `make test-measurement-tiled-cdf` for primitive coverage as well.

For hardware profiling, use the metadata integration executable built with
`-DQCS_MEASUREMENT_PROFILE` and capture CUDA/NCCL activity with Nsight Systems.
Compare active and inactive ranks within the cached-global cases, recording kernel
and transfer durations, API measurement latency, and NCCL AllGather/Broadcast
durations at each rank count. Compare against the parent commit with identical
hardware and inputs. Inactive ranks should show only summary/RNG work; collective
counts should match. Hardware timings were unavailable in the implementation
environment (no CUDA compiler or GPU tools).
