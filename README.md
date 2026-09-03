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
| `patch_offload_streaming_restore.py` | Streamed restore (twelve scheduler edits, four worker, one metadata, two manager), gated on `GLM53_OFFLOAD_STREAM_RESTORE=1`. Removes tier size as the cap on restore size |
| `tests/` | Contract checks against the real vLLM classes, so an engine upgrade fails here rather than silently corrupting KV. `test_streaming_patch.py` applies the whole patch chain to pristine upstream sources and checks anchors, ordering, indentation and idempotence offline |
| `probes/` | The Mamba-necessity experiment, the four W28 gates (addressability, allocator survey, VMM host access, paged bandwidth), and offline unit tests |

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

## Can the CPU staging tier be removed? No — measured, after two wrong answers

Skipping the CPU tier and gathering KV straight to disk with `pwritev`/`preadv` would
remove the staging copy, the uniform-row padding and the restore-window cap at once. It is
not possible on this hardware, and the reason is worth recording because two plausible
intermediate answers were both wrong.

**1. `cudaMalloc` memory is not host-addressable** (`probes/w28_gate.py`): CPU read →
SIGSEGV, `pwritev` → `EFAULT`. GB10 reports `is_integrated=1` and is genuinely coherent,
which makes "it already lives in host RAM" sound right. **Coherent is not
host-addressable.**

**2. Other allocations are** (`probes/w28b_gate.py`): `cudaMallocManaged` and
`cudaHostAlloc` are host-writable, `pwritev`-able (including `O_DIRECT`) and GPU-visible;
ATS is active so the GPU can address plain `malloc`'d memory; and
`CUDAPluggableAllocator` can point a pool at any of them. A contiguous elementwise
benchmark put managed memory at **0.91x** of device bandwidth, which looked like a cheap
trade.

**3. That benchmark was the wrong shape.** Decode does not stream a pool, it gathers
scattered pages out of it. Re-measured with a random block gather
(`probes/w28e_paged.py`), 2 GiB pool, 64 KiB blocks:

| backend | paged gather | vs device |
|---|---|---|
| `cudaMalloc` (today) | 190–198 GB/s | 1.0x |
| `cudaHostAlloc` (pinned) | 2.8 GB/s | ~70x slower |
| `cudaMallocManaged` | 1.2–1.3 GB/s | ~150x slower |

**4. And device memory cannot be made CPU-visible by any route.** A better-targeted idea
is to leave the pool as device memory (attention keeps its full gathers) and take a second,
CPU-readable view of the same pages purely for the sequential copy out — which would sidestep
the gather penalty entirely. Every route is refused
(`probes/w28d_vmm_gate.py`, `probes/w28f_export.py`):

| route | result |
|---|---|
| `cuMemSetAccess`, HOST or HOST_NUMA on a DEVICE allocation | `CUDA_ERROR_NOT_SUPPORTED` |
| device range → `cuMemGetHandleForAddressRange(DMA_BUF_FD)` | `CUDA_ERROR_INVALID_VALUE` |
| `cuMemExportToShareableHandle(POSIX_FILE_DESCRIPTOR)` | fd returned; `mmap` and `pread` both `EINVAL` |
| torch `expandable_segments` tensor, CPU read | SIGSEGV |

The shareable-handle fd is an IPC token for import by another CUDA process, not a readable
byte stream. dma-buf export being refused is consistent with GB10 having no BAR1 — the same
aperture GPUDirect Storage needs.

The mechanism is in the device attributes: `pageableMemoryAccess=1` and
`pageableMemoryAccessUsesHostPageTables=1`. Every host-addressable allocation is reached by
the GPU through **host** page tables — fine for a contiguous stream, ruinous for scattered
gathers, which is exactly attention's pattern. `cudaMemAdvise`/`cudaMemPrefetchAsync` are
rejected as `invalid argument` here because on an integrated part there is one physical
copy and nothing to migrate, so there is no placement hint that rescues it.

**Conclusion: the CPU staging tier is structural on this hardware**, and so are the
restore-window cap and the row padding. GPUDirect Storage is separately impossible (no
BAR1). The lesson, which cost two wrong verdicts: benchmark with the access pattern of the
real workload, not the one that is easy to write.

## The CPU tier is mandatory; its SIZE is not

The staging copy out of device memory cannot be avoided (above). But that is a copy, not a
cache — and vLLM's CPU tier is both, which is why it costs gigabytes. Upstream's guidance
is explicit that `cpu_bytes_to_use` should exceed the aggregate GPU KV cache, and because a
restore stages every hit block before one CPU→GPU load, that also makes tier size the
maximum restorable prefix (measured: 9 blocks → `cached=0`; 39 → ~25k tokens; 79 → ~50k).

`probes/w30_probe.py` measures whether a streamed restore — K successive load jobs through
a small fixed buffer — would be affordable:

```
a (fixed per job) = -0.028 ms      i.e. zero
b (marginal)      = 0.0203 ms/MiB  -> 48.2 GB/s
r2                = 1.000
```

There is no per-job overhead in the transfer path; a 307 MiB restore takes 6.2 ms. The only
cost is the scheduler round trip, measured at 14.6 ms per decode step, so streaming
1400 MiB through a 64 MiB buffer costs ~330 ms — against a cold prefill of the same prefix
measured in seconds.

So the gigabytes are an artefact of the tier being a cache, not a requirement of moving
bytes to disk.

### Removing the cap — measured

`patch_offload_streaming_restore.py` implements the streamed restore in three steps:

1. **The primitive.** `OffloadingConnectorWorker.get_finished` reports `finished_recving`
   as soon as *any* of a request's load jobs completes, so the base scheduler would resume
   the request against a partially-restored prefix. The worker is told which job is the
   last one and releases the request only then.
2. **Per-group batching.** The restore is issued as one job per KV group. Valid because
   `CPUGPUOffloadingWorker._transfer` walks groups positionally and short-circuits on
   `group_size == 0`, so a job carrying one group works as long as `group_sizes` and
   `block_indices` keep full length with zeros elsewhere.
3. **Per-batch promotion.** This is what actually removes the cap. `TieringOffloadingManager
   .lookup` promoted every hit block *during the prefix scan* and returned `MISS` once the
   tier was full, truncating the scan — that is the whole mechanism behind
   `tier size == maximum restorable prefix`. It now reports the hit, and the connector
   promotes each batch immediately before issuing its load, so only one batch need ever be
   resident.

**Measured A/B.** Identical 9-block (256 MiB) primary tier, identical documents, identical
22 GB of disk residue, identical 18,385-token GPU pool — the only difference is
`GLM53_OFFLOAD_STREAM_RESTORE`:

| per revisit | control (vLLM pristine) | arm (streamed) |
|---|---|---|
| read from disk | 233 MiB (12 blocks) | 256 MiB |
| **delivered to GPU** | **0 MiB** | **256 MiB** |
| load jobs issued | **0** | 10 (5 groups × 2 workers) |
| `cached_tokens` | **0** | **7168** |
| restores that happened | **0 / 6** | **6 / 6** |
| needle recall | 5/5 | 5/5 on 5 of 6, 4/5 on 1 |

The control's failure mode is worse than a cap: **it pays the full disk read — 233 MiB,
which is 9 × 25.91 MiB, the entire primary tier — then issues zero load jobs and delivers
zero bytes.** The scan promotes until the tier is full, the next promotion fails, lookup
returns MISS, and the truncated 1-chunk hit falls below the model's 2-chunk Mamba alignment
unit, so it rounds to nothing. Identical to the byte on all six revisits.

The arm moves **10 offload blocks through a 9-block tier**, which is the decoupling: under
the eager path the 10th promotion is exactly what fails.

**The ceiling tracks the prompt, not the tier.** Re-running with a 15.6k-token prompt
(4.35 chunks instead of 3.99) and 3 trials on the same 9-block tier:

| | 14.3k prompt | 15.6k prompt |
|---|---|---|
| `cached_tokens` | 7168 (2 chunks) | **10752 (3 chunks)** |
| bytes CPU→GPU | 256 MiB | 307 MiB (~12 blocks) |
| restores | 6/6 | **9/9** |
| needle recall | 5/5 on 5 of 6 | **5/5 on 9 of 9** |
| stalled restores | — | **0** |
| warm TTFT speedup | 1.3–1.8× | **2.9–3.3×** |

The 7168 was Mamba alignment (cache mode `align`, 2 chunks per unit), and it moved when the
prompt did. The single 4/5 did not reproduce in 9 revisits — both anomalies across the two
runs were the *first* revisit after a boot, before the disk cascade had settled.

**At prod scale** (1M context, 18.8 GB pin, 1,704,433-token pool, 79-block tier), a 125k
session evicted by flushing the whole pool and then revisited:

| | |
|---|---|
| `cached_tokens` | **118,272 — 94.8% of a 124,816-token prompt** |
| CPU→GPU | 1862 MiB (~72 blocks), 959 MiB of it from disk |
| TTFT | **148.3 s → 8.4 s (17.6×)** |
| needle recall / stalls / `NV_ERR_NO_MEMORY` | 5/5 · 0 · 0 |

Read that carefully, though: 72 blocks against a 79-block tier means **the tier was not
binding at 125k**, so the eager path would likely have served it too, and the 17.6× is
restore-vs-cold-prefill — the value of offload in general, not of streaming. A chunk costs
~2.18 blocks here (Mamba groups store snapshots, not one block per chunk), so a 79-block
tier holds ~36 chunks and begins to bind above **~130k tokens**.

**At 250k it does bind**, and this is the run that isolates what streaming buys:

| | 125k | **250k** |
|---|---|---|
| `cached_tokens` | 118,272 (94.8%) | **243,712 (97.6%)** |
| CPU→GPU | 1862 MiB (~72 blk) | **3675 MiB (~142 blk)** |
| vs the 79-block tier | 0.9× — not binding | **1.8× — binding** |
| chained load jobs | 10 | 12 |
| TTFT | 148.3 s → 8.4 s (17.6×) | **250.6 s → 8.3 s (30.2×)** |
| recall / stalls | 5/5 · 0 | **5/5 · 0** |

**142 blocks through a 79-block tier** is something the eager path structurally cannot do —
its 80th promotion fails, `lookup` returns MISS, and the scan truncates. Restoring 97.6% of
a quarter-million-token prompt in 8.3 s against a 250 s cold prefill is only available
streamed.

It cannot deadlock: `prepare_load` pins a block with `ref_cnt += 1` and `complete_load`
drops it back to 0, returning it to the evictable set, so batch *k*'s blocks are reusable
the moment its load completes. Progress is guaranteed as long as a single batch fits, which
is why batches are capped at half the tier (`GLM53_OFFLOAD_STREAM_BATCH_BLOCKS`, 0 = auto):
one group's batch is otherwise unbounded, since a 1M-token restore is ~280 offload blocks
against a 79-block tier.

The residual risk is liveness, not correctness. Once lookup reports the larger hit vLLM has
committed to it, and a queued batch holds no reference — so another request's stores could
in principle evict its blocks from the disk tier mid-restore. The driver `touch()`es every
queued key each step to keep them most-recently-used against exactly that; if a batch
stalls anyway, that one request hangs, loudly and logged, and self-heals when the client
disconnects. It never runs the model against a prefix that was not restored.

## Licence

Apache-2.0. Not affiliated with the vLLM project.
