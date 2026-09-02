# vLLM KV-cache offload: make it correct, bounded, and affordable

Out-of-tree fixes and hardening for vLLM's `OffloadingConnector`, developed against
`vllm 0.1.dev20051+g487ecf187` serving GLM-5.3-Flash (`Glm5Next`, EXL3 4bpw) with TP=2
across two NVIDIA DGX Spark (GB10) hosts.

It started as a size cap for the filesystem tier. Getting offload to actually work then
turned up five upstream defects, two of which produce **wrong output with no error**.
Everything here loads through vLLM's supported extension points — the two tier managers
via `secondary_tiers[].module_path`, the rest as a start-time source patch — so vLLM
itself is never forked.

## What was wrong, and what it cost

| # | Defect | Symptom |
|---|---|---|
| 1 | The divisibility assert covers KV groups that opt out of prefix caching | `OffloadingConnector` cannot initialise at all |
| 2 | Those same groups stay in the hit lookup | offload is **write-only**: `CPU_to_GPU` stays at exactly 0 forever |
| 3 | A secondary tier only ever touches the local node's CPU region | **silent KV corruption** when the TP group spans hosts |
| 4 | The `/dev/shm` region is never unlinked | orphans accumulate until the startup memory gate fails |
| 5 | One uniform region row size for every group | 7× storage blow-up; ~90% of it from one drafter group |

Full analysis, including the exact upstream code sites and suggested fixes, is in
[`UPSTREAM_ISSUE.md`](UPSTREAM_ISSUE.md).

## Measured results

Re-sending a 15.5k-token session after evicting it from both the GPU pool and the CPU
tier, so the restore is served from disk. Answers verified against a per-session needle,
with no cross-session contamination.

| | before | after |
|---|---|---|
| `cached_tokens` on revisit | 0 | **13,312** |
| `CPU_to_GPU` | 0 B, forever | **333–349 MB** |
| TTFT | 15.5 s | **2.5 s (6.0–6.3×)** |

Storage, after the layout fixes:

| | before | after |
|---|---|---|
| region row | 51.81 MiB | **25.91 MiB** |
| CPU tier blocks @ 2 GiB | 39 | **79** |
| restore window @ 2 GiB | ~25k tokens | **~50k tokens** |
| disk, same workload | 7.1 GB | **3.6 GB** |
| offload cost | 75.8 KB/token | **37.6 KB/token** (6.9× → 3.4× vs the GPU pool) |

## Contents

| File | What it is |
|---|---|
| `vllm_bounded_fs_tier.py` | Both tier managers: `BoundedFileSystemTierManager` (byte cap + LRU) and `PeerMirroredFileSystemTierManager` (per-rank cascade, own-slot writes) |
| `peer_kv_agent.py` | Runs in each remote rank's container; performs that rank's own cascade and promotion against its own region and `_r<rank>` files |
| `patch_offload_nonshareable_groups.py` | Idempotent, sentinel-guarded source edits (seven applied, plus an opt-in experiment knob), gated on `GLM53_OFFLOAD_GROUP_FILTER=1`, fail-closed if upstream drifts |
| `tests/` | Contract checks against the real vLLM classes, so an engine upgrade fails here rather than silently corrupting KV |
| `probes/` | The Mamba-necessity experiment and its offline unit tests |

## The patch edits

1–3. Drop `participates_in_prefix_caching == False` groups from the offload lookup, store
and load sets — mirroring what `KVCacheCoordinator.verify_and_split_kv_cache_groups`
already does. Without this every hit is capped at 0 (defect 2).

4. Filter the `config.py` divisibility assert the same way, so an operator does **not**
have to force `--prefix-match-unit` down to the scratch group's block size and drag the
tuned prefix-cache grain along with it (defect 1).

5. Clamp `hashes_per_chunk` to ≥ 1. That assert was also guarding real arithmetic:
`update_offload_keys()` would call `islice(hashes, -1, None, 0)` **for every group**, and
`events.py` both divides by it and asserts it is positive.

6. Exclude fine-grained SWA/draft groups (`alignment_chunk_count is not None`, vLLM's own
marker). The DFlash2 drafter emitted a 54.33 MB row every 64 tokens — 1276 of 1410 files,
90% of 72 GB. Excluding it took the same workload to 7.1 GB. Safe because draft tokens are
verified by the target model, so unrestored draft KV costs acceptance rate while it
refills, never correctness (defect 5).

7. `GLM53_OFFLOAD_REGION_COPIES=<GPUs per node>` sizes region rows by the workers that
actually *share* a region. `tiering/spec.py` picks a worker's slot as
`torch.accelerator.current_device_index() % world_size`, which is the global rank only when
all workers sit on one node; with one GPU per host it is 0 everywhere, so every worker uses
slot 0 while rows are still sized for `world_size`. Opt-in and clamped — a value below the
real GPUs-per-node would make co-located workers share a slot and corrupt each other.

## Operational notes

* **Size the CPU tier by the restore path, not the store path.** vLLM promotes every hit
  block into the CPU tier before issuing a single CPU→GPU load, so the tier size *is* the
  maximum restorable prefix:
  `tokens ≈ (cpu_bytes_to_use / row_bytes / storable_groups) × tokens_per_chunk`.
  Cascade bandwidth is never the constraint. A tier that is too small does not merely slow
  restores — it makes offload **inert**: measured `cached=0` with the blocks present on
  disk, 13 tier hits found, and 4 `kv_offload_tiering_promotion_allocation_failures_total`.
  Always probe a tier size before deploying it.
* **Fund the tier from the KV pool, not on top of it.** On unified memory (GB10) the CPU
  tier and the GPU pool draw on the same RAM; adding a tier on top drove host `available`
  to 0.
* **`max_bytes` bounds every rank.** The head enforces it with LRU and mirrors each
  deletion to the peers; each peer also sweeps its own shard oldest-first against the same
  cap as a backstop, so a dropped message cannot leak disk.
* **Clear `/dev/shm/vllm_offload_*.mmap` before launch** (defect 4). Count with a shell
  glob, not `ls | wc -l` — `ls` exits 2 on no match, and under `set -e -o pipefail` that
  silently kills the launch the moment the cleanup has nothing left to do.
* **`PYTHONHASHSEED` must be fixed and identical everywhere**, or vLLM seeds its
  block-content hash chain randomly and identical tokens map to different filenames on
  every restart.

## Known ceiling

A residual **1.64×** remains: rows are uniform across groups, so a 2.48 MiB MLA payload
still occupies a 25.91 MiB row. Removing it needs ragged per-group rows across
`config.py`, `cpu/spec.py`, `shared_offload_region.py` and the CPU worker's offset
arithmetic.

Below that sits a **~2× floor over the pool** that is not a layout bug: the four Mamba
groups are ~76 of the 78.7 MiB of real payload per chunk, because offload snapshots
recurrent state at *every* chunk boundary while the live pool keeps only the current state
per request. Dropping those snapshots was tested and found to corrupt restores
intermittently (see below), so the floor stands.

## Is the Mamba recurrent state needed for a restore? Yes — measured

The Mamba groups are ~97% of the real offload payload, so dropping them would cut storage
~5x and widen the restore window ~5x. `probes/mamba_probe.py` tests it by excluding them
and scoring recall of five needles placed inside the restored region.

An initial 6 sessions across two arms showed no damage at all, which looked like a large
free win. It was underpowered. With the env plumbing corrected and a third trial:

| | sessions | failures |
|---|---|---|
| state offloaded (control) | 6 | **0** |
| state excluded | 9 | **1** |

The failure returned 384 tokens of neither content nor reasoning on its first restored
request — the same degenerate signature as the multi-host KV corruption — and its later
turns only looked healthy because the prompt was by then correctly cached in GPU.

**Conclusion: the recurrent state is load-bearing, at least intermittently, and a needle
probe at n=3 will miss it.** `GLM53_OFFLOAD_EXCLUDE_MAMBA` defaults to off and logs a
warning when forced on. Any future work that sparsifies these snapshots must be validated
at high trial count against this failure signature, not at n=3.

An exact-match oracle was tried before the needle one and rejected: greedy output is not
bit-reproducible across differing prefill paths, so it flagged noise as corruption.
`probes/` includes offline unit tests (22 checks) for the probe's placement, scoring,
metrics parsing and verdict logic — a live iteration here costs ~15 minutes, an offline one
milliseconds.

## Licence

Apache-2.0. Not affiliated with the vLLM project.
