# Restore a 250k-token KV cache from disk in 8 seconds instead of prefilling it in 250

![vLLM](https://img.shields.io/badge/vLLM-out--of--tree%2C%20never%20forked-1f6feb)
![Hardware](https://img.shields.io/badge/2%C3%97%20DGX%20Spark-GB10%20%C2%B7%20TP%3D2-76b900)
![Context](https://img.shields.io/badge/context-1M%20tokens-8250df)
![Restore](https://img.shields.io/badge/250k%20restore-8.3s%20vs%20250.6s-2da44e)
![Licence](https://img.shields.io/badge/licence-Apache--2.0-lightgrey)

**Time to first token, same 250k-token prompt, same server:**

```text
cold prefill   ██████████████████████████████████████████████████  250.6 s
from disk      ██················································    8.3 s   30.2x faster
```

Out-of-tree fixes and hardening for vLLM's `OffloadingConnector`, developed against
`vllm 0.1.dev20051+g487ecf187` serving GLM-5.3-Flash (`Glm5Next`, EXL3 4bpw) at 1M context,
TP=2 across two NVIDIA DGX Spark (GB10) hosts.

Everything loads through vLLM's supported extension points — the two tier managers via
`secondary_tiers[].module_path`, the rest as an idempotent start-time source patch. vLLM
itself is never forked.

## Results

A **250k-token session**, evicted by flushing the entire 1.7M-token GPU pool, then
revisited so the KV genuinely comes back off disk. Answers verified against five needles at
10/30/50/70/90% depth, so a partially-restored prefix cannot pass as a full one.

| | cold prefill | restored from disk |
|---|---|---|
| **TTFT** | **250.6 s** | **8.3 s — 30.2× faster** |
| `cached_tokens` | — | **243,712 of 249,638 (97.6%)** |
| needle recall | 5/5 | **5/5** |

```text
how much of the 249,638-token prompt came back off disk

restored from disk  ████████████████████████████████████████████████·  243,712  (97.6%)
re-prefilled        ·················································    5,926   (2.4%)
```

Where those 8.3 s go — **the restore itself is about half a second**:

```mermaid
pie showData
    title Warm TTFT of a 250k restore, in seconds
    "Re-prefill the tail Mamba alignment leaves" : 5.95
    "Tokenise + first-token decode + scheduling" : 1.80
    "Disk read, 1865 MiB at 5.34 GB/s" : 0.37
    "Streaming batches, 6 at ~15 ms" : 0.09
    "CPU to GPU, 3675 MiB at 48.7 GB/s" : 0.07
```

| | |
|---|---|
| re-prefilling the 5,926-token tail Mamba alignment leaves behind | 5.95 s |
| disk read, 1865 MiB @ **5.34 GB/s** | 0.37 s |
| CPU→GPU, 3675 MiB @ **48.7 GB/s** | 0.07 s |
| 6 streaming batches @ ~15 ms | 0.09 s |
| tokenisation, first-token decode, scheduling | ~1.8 s |

The leftover tail is bounded by a constant (≤ 7168 tokens), while the prefill you skip grows
linearly with depth — so this improves as sessions get deeper:

```text
         cold prefill                                 restored
  125k   ██████··································   148 s   █·······   8.4 s    17.6x
  250k   ██████████······························   251 s   █·······   8.3 s    30.2x
    1M   ████████████████████████████████████████  ~1000 s  █·······  ~8 s     ~125x  (projected)
```

| session | cold prefill | restored | speedup |
|---|---|---|---|
| 125k | 148 s | 8.4 s | **17.6×** |
| 250k | 251 s | 8.3 s | **30.2×** |
| 1M | ~1000 s | ~8 s | ~125× *(projected, not measured)* |

### The result stock vLLM cannot produce

That 250k restore moves **142 offload blocks through a 79-block CPU tier**. Stock vLLM
promotes every hit block *during* the prefix scan, so its 80th promotion fails, `lookup`
returns `MISS`, and the scan truncates there.

```mermaid
flowchart TD
    S["Prefix scan asks: is this block offloaded?"] --> H{"Found in the disk tier?"}
    H -- no --> M0["MISS, scan ends here"]
    H -- yes --> P{"Promote it into the CPU tier now?"}

    P -- "STOCK: yes, during the scan" --> F{"Did the tier have room?"}
    F -- "yes" --> HP["HIT_PENDING: scan defers to next step.<br/>Nothing has pinned this block yet,<br/>so LRU evicts it to make room for the next.<br/>Never converges."]
    F -- "no, tier full" --> MI["MISS: scan truncates.<br/>Tier size becomes the restore ceiling."]
    HP --> DEAD["Request stuck in<br/>waiting_by_reason=deferred"]
    MI --> WASTE["233 MiB read off disk,<br/>0 bytes delivered to the GPU"]

    P -- "STREAMED: no, defer it" --> HIT["Report a plain HIT.<br/>Scan completes with zero deferrals."]
    HIT --> BATCH["Connector promotes each batch<br/>immediately before its load.<br/>Only ONE batch is ever resident."]
    BATCH --> WIN["142 blocks delivered<br/>through a 79-block tier"]

    style WASTE fill:#f8d7da,stroke:#c33
    style DEAD fill:#f8d7da,stroke:#c33
    style WIN fill:#d4edda,stroke:#2a2
```

An A/B with everything else held identical — same 9-block tier, same documents, same disk
contents, same GPU pool — toggling only `GLM53_OFFLOAD_STREAM_RESTORE`:

| per revisit | stock | streamed |
|---|---|---|
| read from disk | 233 MiB | 256 MiB |
| **delivered to GPU** | **0 MiB** | **256 MiB** |
| load jobs issued | **0** | 10 |
| `cached_tokens` | **0** | **7168** |
| restores that happened | **0 / 6** | **6 / 6** |

> [!IMPORTANT]
> Stock does not merely restore *less* — **it pays the full disk read and then throws all of
> it away.** 233 MiB is exactly the whole tier: the scan promotes until the tier is full, the
> next promotion fails, and the truncated hit falls below the model's Mamba alignment unit,
> so it rounds to nothing and **no load job is ever issued**. Identical to the byte on all
> six revisits.

Repeated at 15.6k with 3 trials: **9/9 restores, 5/5 needle recall on 9 of 9, 0 stalls.**

### Capacity

```text
restorable context, in tokens

GPU KV pool    18.8 GB   ████████························  1.70M
disk tier     100 GiB    ████████████████████████████████  6.80M   ~4x the pool
```

100 GiB of disk holds **~6.8M tokens** of restorable context — about **4× the 1.7M-token GPU
pool** — at a measured 14.8 KB/token (`blocks = 2.00 × chunks + 5.9`, r² clean across the
125k and 250k runs). That is ~27 sessions at 250k, or ~52 at 125k.

## How the streamed restore works

`patch_offload_streaming_restore.py`, gated on `GLM53_OFFLOAD_STREAM_RESTORE=1`:

1. **Let a restore span several load jobs.** `OffloadingConnectorWorker.get_finished`
   reports `finished_recving` as soon as *any* of a request's load jobs completes, which
   would resume the request against a partially-restored prefix. The worker is now told
   which job is the last and releases the request only then.
2. **One job per KV group.** Valid because `CPUGPUOffloadingWorker._transfer` walks groups
   positionally and short-circuits on `group_size == 0`, so a job carrying a single group
   works as long as `group_sizes` and `block_indices` keep full length with zeros elsewhere.
3. **Promote per batch.** This is what removes the cap. `TieringOffloadingManager.lookup`
   promoted during the prefix scan, and **both** outcomes tied restore size to tier size:
   `MISS` truncates the scan, and `HIT_PENDING` defers it into a livelock where LRU evicts
   blocks promoted a step earlier, because `prepare_load` has not pinned them yet. So
   `lookup` now reports a plain HIT and does not promote at all; the connector promotes each
   batch immediately before issuing its load, and only one batch is ever resident.

```mermaid
sequenceDiagram
    participant Sch as Scheduler
    participant Mgr as TieringOffloadingManager
    participant Tier as CPU tier, 79 blocks
    participant W as GPU worker

    Note over Sch,W: batch k, then batch k+1. Only one is ever resident.
    Sch->>Mgr: stream_residency(batch k keys)
    Mgr->>Tier: promote from disk
    Tier-->>Mgr: resident
    Sch->>Mgr: prepare_load(batch k)
    Mgr->>Tier: ref_cnt += 1, now pinned and not evictable
    Sch->>W: load job, marked NOT final
    W-->>Sch: done, but finished_recving is withheld
    Sch->>Mgr: complete_load(batch k)
    Mgr->>Tier: ref_cnt drops to 0, back in the evictable set
    Note over Tier: batch k blocks are reusable again,<br/>which is exactly what batch k+1 needs
    Sch->>Mgr: stream_residency(batch k+1 keys)
    Mgr->>Tier: promote, reusing batch k slots
    Sch->>W: load job, marked final
    W-->>Sch: finished_recving, request resumes
```

**It cannot deadlock.** `prepare_load` pins a block with `ref_cnt += 1` and `complete_load`
drops it back to 0, returning it to the evictable set — so batch *k*'s blocks are reusable
the moment its load completes. Progress is guaranteed as long as a single batch fits, which
is why batches are capped at half the tier (`GLM53_OFFLOAD_STREAM_BATCH_BLOCKS`, 0 = auto).

> [!WARNING]
> **The residual risk is liveness, not correctness.** Once `lookup` reports the larger hit
> vLLM has committed to it, and a queued batch holds no reference — so another request's
> stores could in principle evict its blocks from the disk tier mid-restore. The driver
> `touch()`es every queued key each step to keep them most-recently-used against that. A
> batch that stalls anyway hangs *that one request*, logged as an `ERROR`, and self-heals
> when the client disconnects. It never runs the model against a prefix that was not
> restored.

## The data path

```mermaid
flowchart LR
    subgraph N1["HEAD NODE — DGX Spark GB10"]
        direction LR
        G1["GPU KV pool<br/><b>1.70M tokens</b><br/>18.8 GB"]
        C1["CPU staging tier<br/><b>79 blocks / 2 GiB</b><br/><i>a buffer, not a cache</i>"]
        D1["Disk tier<br/><b>100 GiB cap, LRU</b><br/>~6.8M tokens"]
    end
    subgraph N2["PEER NODE — rank 1"]
        direction LR
        C2["CPU region<br/><i>own slot only</i>"]
        D2["Disk shard<br/><i>_r1 files</i>"]
    end

    G1 -- "store<br/>48.7 GB/s" --> C1
    C1 -- "cascade" --> D1
    D1 -- "read<br/>5.34 GB/s" --> C1
    C1 == "restore 48.7 GB/s<br/><b>one batch at a time</b>" ==> G1
    C1 -. "peer agent" .-> C2
    C2 --> D2
    D1 -. "LRU deletions<br/>mirrored" .-> D2

    classDef gpu  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    classDef cpu  fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    classDef disk fill:#fed7aa,stroke:#ea580c,stroke-width:2px
    class G1 gpu
    class C1,C2 cpu
    class D1,D2 disk
    style N1 fill:#f6f8fa,stroke:#d0d7de,stroke-width:1px
    style N2 fill:#f6f8fa,stroke:#d0d7de,stroke-width:1px
```

Each rank writes only its **own** slot of the shared region and owns its own `_r<rank>`
files. Getting that wrong is upstream defect 3: a secondary tier that only ever touches the
local node's region restores zeros for every remote rank — silent KV corruption, which is
how it was found.

## Five upstream defects found on the way

| # | Defect | Symptom |
|---|---|---|
| 1 | The divisibility assert covers KV groups that opt out of prefix caching | `OffloadingConnector` cannot initialise at all |
| 2 | Those same groups stay in the hit lookup | offload is **write-only**: `CPU_to_GPU` stays at exactly 0 forever |
| 3 | A secondary tier only ever touches the local node's CPU region | **silent KV corruption** when the TP group spans hosts |
| 4 | The `/dev/shm` region is never unlinked | orphans accumulate until the startup memory gate fails |
| 5 | One uniform region row size for every group | 7× storage blow-up, ~90% of it from one drafter group |

Two of these produce wrong output with no error. Exact code sites and suggested fixes are in
[`UPSTREAM_ISSUE.md`](UPSTREAM_ISSUE.md).

Fixing 5 also halved the on-disk footprint — 51.81 MiB → **25.91 MiB** per region row, and
75.8 → **37.6 KB/token** on the same workload.

## Contents

| File | What it is |
|---|---|
| `patch_offload_streaming_restore.py` | The streamed restore: 19 idempotent, sentinel-guarded edits (12 scheduler, 4 worker, 1 metadata, 2 manager), gated on `GLM53_OFFLOAD_STREAM_RESTORE=1` |
| `patch_offload_nonshareable_groups.py` | Fixes defects 1, 2 and 5 plus row sizing, gated on `GLM53_OFFLOAD_GROUP_FILTER=1` |
| `vllm_bounded_fs_tier.py` | `BoundedFileSystemTierManager` (byte cap + LRU) and `PeerMirroredFileSystemTierManager` (per-rank cascade, own-slot writes — fixes defect 3) |
| `peer_kv_agent.py` | Runs in each remote rank's container; performs that rank's own cascade and promotion against its own region and `_r<rank>` files |
| `tests/` | Offline suites — the patch chain against pristine upstream sources, and the batch-split arithmetic |
| `probes/` | Every measurement harness used in this work, including the needle probes and the restore-cost fit behind the numbers above |

Both patches are idempotent, sentinel-guarded, and **fail closed** if upstream drifts: an
anchor that no longer matches exactly once aborts the patch rather than half-applying it.

## Usage

```jsonc
// kv_transfer_config
{"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
 "kv_connector_extra_config": {
   "spec_name": "TieringOffloadingSpec",
   "cpu_bytes_to_use": 2147483648,          // staging buffer, not a cache
   "eviction_policy": "lru",
   "secondary_tiers": [{
     "type": "PeerMirroredFileSystemTierManager",
     "module_path": "vllm_bounded_fs_tier",  // put this file on PYTHONPATH
     "root_dir": "/kvoffload",
     "max_bytes": 107374182400,              // hard cap, enforced on every rank
     "evict_to_ratio": 0.9,
     "n_read_threads": 8, "n_write_threads": 8}]}}
```

Then apply the patches at container start (before the engine imports vLLM) and set:

```sh
GLM53_OFFLOAD_GROUP_FILTER=1
GLM53_OFFLOAD_STREAM_RESTORE=1
GLM53_OFFLOAD_REGION_COPIES=<GPUs per node>   # see note below
```

### Operational notes

> [!TIP]
> **The CPU tier is a staging buffer, not a cache.** With the streamed restore its size no
> longer caps restore depth — it sets the batch size, hence the number of engine steps
> (a 1M restore is ~15 batches ~ 0.2 s at a 2 GiB tier). Size it for how many restores you
> want *in flight at once*, not for how deep they are. This inverts the usual advice.

* **Fund the tier from the KV pool, not on top of it.** On unified memory (GB10) the CPU
  tier and the GPU pool draw on the same RAM.
* **`max_bytes` bounds every rank.** The head enforces it with LRU and mirrors each deletion
  to the peers; each peer also sweeps its own shard oldest-first against the same cap as a
  backstop, so a dropped message cannot leak disk.
* **`GLM53_OFFLOAD_REGION_COPIES`** sizes region rows by the workers that actually *share* a
  region. `tiering/spec.py` picks a worker's slot as
  `torch.accelerator.current_device_index() % world_size`, which is the global rank only
  when all workers sit on one node; with one GPU per host it is 0 everywhere, so every
  worker uses slot 0 while rows are still sized for `world_size`. Opt-in and clamped — a
  value below the real GPUs-per-node would make co-located workers share a slot and corrupt
  each other.
* **Clear `/dev/shm/vllm_offload_*.mmap` before launch** (defect 4). Count with a shell
  glob, not `ls | wc -l` — `ls` exits 2 on no match, and under `set -e -o pipefail` that
  silently kills the launch the moment the cleanup has nothing left to do.
* **`PYTHONHASHSEED` must be fixed and identical everywhere**, or vLLM seeds its
  block-content hash chain randomly and identical tokens map to different filenames on every
  restart.

## Running the tests

```sh
python3 tests/test_stream_split.py        # split arithmetic, 8 property classes
python3 tests/test_streaming_patch.py     # the patch chain vs pristine upstream
python3 probes/test_w30_2b_probe.py       # the needle probe's own scoring
python3 probes/test_w30_probe.py
python3 probes/test_mamba_probe.py
```

These need only a Python 3 interpreter. The two patch-chain tests apply the edits to real
vLLM sources — point them at a copy with `PRISTINE=/path/to/site-packages/vllm`, and they
skip with a clear message if it is absent.

They are not decoration. `test_streaming_patch.py` caught four real defects before any
container start, including an `@override` decorator an anchor had clipped — which would have
silently detached `prepare_load` from its decorator — and a scheduler anchor that only exists
after the other patch has run. `test_stream_split.py` extracts the shipped function from the
patched source rather than testing a copy of it, and covers the case where losing the
intra-chunk offset would shift blocks by less than one chunk: KV that loads without error and
is quietly wrong.

`tests/test_bounded_fs_tier.py` and `tests/test_peer_cascade.py` are contract tests against
the real vLLM classes and import `vllm`, so they need it installed. That is their purpose —
an engine upgrade that moves an API fails there rather than silently corrupting KV.

## Limits

> [!NOTE]
> Every number above is measured on this stack unless explicitly marked *projected*. The
> A/B toggles one environment variable and holds hardware, model, prompts, disk contents and
> pool size constant.


* **Restore depth is now bounded by Mamba `align`, not by the tier.** The restorable prefix
  rounds down to an aligned chunk boundary, leaving ≤ 7168 tokens to re-prefill — which is
  most of the 8.3 s above.
* **Below ~130k tokens the restore fits a 2 GiB tier**, so stock vLLM serves it too and
  streaming buys nothing. The gain is in deep sessions.
* **Concurrent deep restores are unmeasured.** Every result here is a single sequential
  restore. At the default batch size only 2 deep restores hold a full batch at once; further
  ones retry and serialise rather than fail. Lower `GLM53_OFFLOAD_STREAM_BATCH_BLOCKS` (8 →
  9 concurrent, ~1.1 s per 1M restore) if you need more.
* **A residual 1.64× storage overhead remains**: rows are uniform across groups, so a
  2.48 MiB MLA payload still occupies a 25.91 MiB row. Removing it needs ragged per-group
  rows across `config.py`, `cpu/spec.py`, `shared_offload_region.py` and the CPU worker's
  offset arithmetic.
* **The first restore after a boot behaves as cold**, before the disk cascade has settled.
  Both anomalies across 15 revisits were first-after-boot; don't read one as corruption.

## Licence

Apache-2.0. Not affiliated with the vLLM project.
