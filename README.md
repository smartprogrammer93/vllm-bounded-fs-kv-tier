# vllm-bounded-fs-kv-tier

A drop-in, **size-bounded** filesystem secondary tier for vLLM's KV-cache offloading —
with LRU eviction and a configurable byte cap.

## The problem

vLLM's built-in filesystem tier (`vllm/v1/kv_offload/tiering/fs/manager.py`) writes one
`.bin` file per KV block and **never removes anything**. There is no capacity limit, no
eviction, and no quota. If your `root_dir` sits on a volume you cannot afford to fill,
the cache will grow until the filesystem is full.

Audited against `vllm 0.1.dev20051+g487ecf187`:

| | finding |
|---|---|
| `FileSystemTierManager` implements | `lookup`, `submit_store`, `submit_load`, `get_finished_jobs`, `take_events`, `drain_jobs`, `on_new_request`, `on_request_finished`, `on_schedule_end`, `shutdown` |
| does **not** implement | `touch()` — so there is no recency tracking at all |
| capacity / eviction / quota | absent from the `fs` tier, the `obj` tier, and the `SecondaryTierManager` ABC |

The repo's test suite asserts these facts, so if a future vLLM grows its own capacity
handling the tests fail loudly rather than letting this quietly duplicate it.

## What this adds

* **Hard byte cap** (`max_bytes`), accepting `214748364800`, `"200GiB"` or `"200GB"`.
* **LRU eviction** down to a low-watermark (`evict_to_ratio`, default `0.9`).
* **`touch()` implemented**, so recency is real LRU rather than FIFO.
* **Restart-safe**: adopts `.bin` files left by a previous run (oldest first) and trims
  before serving, so a restart cannot stack a fresh cache on top of an old one.
* **No vLLM patching.** It loads through vLLM's supported out-of-tree extension point
  (`secondary_tiers[].module_path`), so it survives engine upgrades.

## Why eviction needs no locking

Deleting a file out from under an in-flight load is safe *by construction* in vLLM. The
built-in load task catches `OSError` (which includes `FileNotFoundError`), records how many
blocks succeeded, and `get_finished_jobs()` calls `self._lookup_manager.mark_miss(failed)`.

The affected blocks simply become a **cache miss and are recomputed** — no corruption, no
failed request. So the evictor never has to coordinate with readers.

## Accounting

`batch_store_block` writes exactly one file of `block_size` **bytes** per key, so usage is
`len(tracked_files) * block_size`. No `stat()` per file, no directory walks on the hot path.

Tracking happens at *submit* rather than completion, which deliberately over-counts a failed
store. That is self-healing: eviction tolerates `FileNotFoundError` and drops the entry.

## Compatibility: check this before you start

vLLM's offloading framework requires **every KV cache group's block size to be a multiple
of its hash granularity** (`offloading/config.py`):

```python
tokens_per_block = group.kv_cache_spec.block_size * (dcp if AttentionSpec else 1)
assert group.tokens_per_block % tokens_per_hash == 0
```

`tokens_per_hash` is derived by `resolve_kv_cache_block_sizes()` as an LCM-style alignment
across groups, so it grows *away* from the smallest group rather than toward it.

That makes offloading unusable for models with a **small-block compressed KV group** — for
example a DeepSeek-V4-style sparse indexer with `compress_ratio == index_kpool == 4`, whose
group has a 4-token block while the attention groups use 64. Observed:

```
tokens_per_block=4 not divisible by tokens_per_hash=64      # speculation on
tokens_per_block=4 not divisible by tokens_per_hash=3328    # speculation off
```

The error suggests `--enable-prefix-caching` aligns block sizes; that is a red herring if
prefix caching is already enabled. A 4-token group can only pass if the hash is ≤ 4, which
is unreachable when other groups need 64.

**Check first** — if any KV group's block size is smaller than the hash granularity,
no offloading connector will initialise, and this tier never gets a chance to run.

A **workaround needing no patch** is `--prefix-match-unit 4` (or any value dividing the
smallest group's block size): `resolve_kv_cache_block_sizes` validates the requested unit
only against *participating* groups, so it is accepted and the assert then passes. Note it
multiplies stored blocks roughly 7.5×, since hashing happens at that finer granularity.

A ready-to-file upstream report is in [UPSTREAM_ISSUE.md](UPSTREAM_ISSUE.md), including the
five index-desync sites a proper fix has to touch.

## Install

Put `vllm_bounded_fs_tier.py` anywhere on the engine's `PYTHONPATH` (e.g. mount it at
`/opt/vllm-ext` and set `PYTHONPATH=/opt/vllm-ext`).

## Configure

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 1073741824,
    "eviction_policy": "lru",
    "secondary_tiers": [
      {
        "type": "BoundedFileSystemTierManager",
        "module_path": "vllm_bounded_fs_tier",
        "root_dir": "/kvoffload",
        "max_bytes": "200GiB",
        "evict_to_ratio": 0.9,
        "n_read_threads": 8,
        "n_write_threads": 8
      }
    ]
  }
}
```

Passed to vLLM as `--kv-transfer-config '<that json>'`.

### Options

| key | default | meaning |
|---|---|---|
| `max_bytes` | *required* | hard cap; int bytes or `"200GiB"` / `"200GB"` / `"500MB"` |
| `evict_to_ratio` | `0.9` | evict down to this fraction of the cap, so eviction is batched rather than per-block |

All other keys (`root_dir`, `n_read_threads`, `n_write_threads`, `enable_kv_events`,
`locality`) are passed straight through to vLLM's `FileSystemTierManager`.

## A gotcha worth knowing

Set **`PYTHONHASHSEED`** to a fixed value (e.g. `0`) on every engine process. vLLM seeds its
block-content hash chain with random bytes otherwise, so identical token content produces
different filenames on each run — meaning the on-disk cache is silently invalidated by every
restart, and cannot be shared between instances.

## Tests

```bash
python3 tests/test_bounded_fs_tier.py
```

Runs anywhere vLLM is importable — no GPU and no CUDA context required. Covers the parent
contract, `parse_bytes`, real-file LRU eviction, tolerance of already-deleted files, and
restart adoption.

## License

Apache-2.0, matching vLLM.

## Also here: a fix for the offload read path

`patch_offload_nonshareable_groups.py` fixes a vLLM bug that makes KV offloading
**write-only** on any model with a KV cache group whose spec sets
`participates_in_prefix_caching = False` (GLM5Next's `KpoolTailSpec`, for
example). Such a group holds per-request scratch and can never report a
prefix hit, but `OffloadingConnectorScheduler` still queries it, and
`_lookup_complete_chunks` returns 0 as soon as any queried group misses. Result:
stores succeed, `CPU_to_GPU` stays at exactly 0 forever, and every request
re-prefills in full.

The patch mirrors what vLLM core already does in
`KVCacheCoordinator.verify_and_split_kv_cache_groups`. Measured on
GLM-5.3-Flash (TP=2), re-sending a 15.5k-token session after evicting it:

| | before | after |
|---|---|---|
| `cached_tokens` | 0 | 13,312 |
| `CPU_to_GPU` | 0 B | 349 MB |
| TTFT | 15.5 s | 2.5 s (**6.0-6.3x**) |

Answers verified against a per-session needle, with no cross-session
contamination.

See `UPSTREAM_ISSUE.md` for the full analysis, including a third defect: a
secondary (disk) tier **silently corrupts KV when the tensor-parallel group
spans hosts**, because the tier only ever touches the local node's CPU region.

## And: disk offload made correct on a multi-host TP group

`peer_kv_agent.py` + `PeerMirroredFileSystemTierManager` fix the third defect —
a secondary (disk) tier silently corrupting KV when the tensor-parallel group
spans hosts, because vLLM builds secondary tiers scheduler-side over the local
node's CPU region only. The head-side tier mirrors every cascade and promotion
to an agent in each remote rank's container, which does byte-identical I/O on
its own region and its own `_r<rank>` files, and no job is released until every
peer acknowledges — so disk -> CPU is ordered before CPU -> GPU on all ranks.

Restores served from disk, needle-verified, on 2x DGX Spark (TP=2, two hosts):
**15.5 s -> 2.6 s (5.8-6.0x)**, 13,312 tokens cached, 8 FS-tier block hits per
session, both ranks holding their own shard. The same config without the
cascade returned 512 empty tokens.

## Production notes

* `patch_offload_nonshareable_groups.py` carries six idempotent edits, all gated
  on `GLM53_OFFLOAD_GROUP_FILTER=1` and fail-closed on upstream drift:
  the three lookup/store/load filters, the `config.py` divisibility-assert
  filter (which removes the need to force `--prefix-match-unit` down to the
  scratch group's block size, so a deployment keeps its tuned prefix-cache
  grain), a `max(1, ...)` clamp on `hashes_per_chunk` (the assert was also
  guarding real arithmetic — `islice(..., -1, None, 0)` and two divisions), and
  the fine-grained SWA/draft-group exclusion that cut one workload's offload
  footprint from 72 GB to 7.1 GB.
* Size the CPU tier by the **restore** path, not the store path. vLLM promotes
  every hit block into the CPU tier before issuing a single CPU->GPU load, so
  the tier size is the maximum restorable prefix. Cascade bandwidth is never the
  constraint; a tier that is too small returns `cached=0` and raises
  `kv_offload_tiering_promotion_allocation_failures_total`.
* `max_bytes` bounds every rank: the head enforces it with LRU and mirrors each
  deletion to the peers, and each peer additionally sweeps its own shard
  oldest-first against the same cap as a backstop.
* Clear `/dev/shm/vllm_offload_*.mmap` before launch. vLLM never unlinks it, and
  on unified memory the orphans eventually fail the startup memory gate.

## W27: cutting the offload footprint in half, twice

KV that leaves the GPU pool ballooned ~7x on this deployment. Two causes, both found by
dumping the live `/dev/shm` regions rather than reasoning about the code:

`tiering/spec.py::create_worker` picks a worker's slot inside the region with
`torch.accelerator.current_device_index() % world_size`. That equals the global rank only
when all workers share one node. The region is one file **per node**, and with one GPU per
host `current_device_index()` is 0 everywhere -- so every worker uses slot 0 while
`cpu/spec.py` still sizes each row as `world_size * per-worker bytes`. Half of every row,
and half of `cpu_bytes_to_use`, was dead. Confirmed: slot 0 populated, slot 1 all-zero on
both hosts.

* **Store only the worker's own slot** (`GLM53_OFFLOAD_RANK_LOCAL_ROWS=1`,
  `GLM53_OFFLOAD_LOCAL_SLOT=<local device index>`) -- halves disk. Implemented by
  narrowing the offsets the parent hands to the io helpers, so its job bookkeeping,
  thread pool and partial-load failure handling are reused untouched.
* **Size rows by the workers that actually share a region**
  (`GLM53_OFFLOAD_REGION_COPIES=<GPUs per node>`, edit 7) -- doubles the blocks a given
  `cpu_bytes_to_use` buys, and since the CPU tier bounds a restore, doubles the restore
  window. Opt-in and clamped; setting it below the real GPUs-per-node would make
  co-located workers share a slot and corrupt each other, so it is never defaulted.

| | before | after |
|---|---|---|
| region row | 51.81 MiB | 25.91 MiB |
| CPU tier blocks @ 2 GiB | 39 | **79** |
| restore window @ 2 GiB | ~25k tokens | **~50k tokens** |
| disk, same workload | 7.1 GB | **3.6 GB** |
| offload cost | 75.8 KB/token | **37.6 KB/token** (6.9x -> 3.4x vs the pool) |

Needle-verified after each step. A residual 1.64x remains because rows are uniform across
groups (the MLA group's real payload is 2.48 MiB in a 25.91 MiB row); removing it needs
ragged per-group rows. Below that lies a ~2x floor that is not a layout bug at all: Mamba
groups snapshot recurrent state at every chunk boundary, where the live pool keeps only
the current state per request.
