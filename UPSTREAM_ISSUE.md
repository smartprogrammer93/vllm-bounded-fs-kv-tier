# Upstream defect: KV offloading applies a prefix-caching constraint to groups that opt out

Ready-to-file report for `vllm-project/vllm`. Observed on
`vllm 0.1.dev20051+g487ecf187`, GLM-5.3-Flash (`Glm5Next`), TP=2, `mp` executor.

## Summary

`OffloadingConnector` cannot initialise on any model that has a KV cache group with
`participates_in_prefix_caching == False` and a block size smaller than the prefix-hash
granularity — even though `resolve_kv_cache_block_sizes()` deliberately excludes exactly
those groups, and `KVCacheSpec`'s own docstring says they must be excluded from "the hybrid
divisibility assert".

## Reproduction

Serve a model whose KV groups include a small-block, non-prefix-caching scratch group. On
GLM-5.3-Flash the sparse-indexer tail (`KpoolTailSpec`, `block_size == index_kpool == 4`)
is such a group. Enable offloading:

```
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec",
  "cpu_bytes_to_use":4294967296}}'
```

Result:

```
AssertionError: tokens_per_block=4 not divisible by tokens_per_hash=64.
Hybrid models (e.g. Mamba+Attention) need --enable-prefix-caching to align block sizes.
```

`--enable-prefix-caching` was already enabled; the hint is misleading. With speculative
decoding disabled the same assert fires with `tokens_per_hash=3328`.

## Root cause

`distributed/kv_transfer/kv_connector/v1/offloading/config.py` builds its group list from
**every** entry and asserts over all of them:

```python
groups = tuple(
    OffloadingGroupConfig(tokens_per_block=..., layer_names=...)
    for group in kv_cache_config.kv_cache_groups        # <-- unfiltered
)
_, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
for group in groups:
    assert group.tokens_per_block % tokens_per_hash == 0
```

But `v1/core/kv_cache_utils.py::resolve_kv_cache_block_sizes` filters precisely these
groups out when deriving that value, and says why:

```python
# Groups that opt out of prefix caching (e.g. GLM5Next's KpoolTailSpec, a
# 1-block/req scratch buffer) never have block_hashes computed, so their
# block_size must not constrain the hash granularity.
hashing_sizes = [bs for g, bs in zip(groups, group_block_sizes)
                 if g.kv_cache_spec.participates_in_prefix_caching] or group_block_sizes
```

`KVCacheSpec.participates_in_prefix_caching`'s docstring names the three consumers that
must ignore such groups: "the global `cache_config.block_size` min, **the hybrid
divisibility assert**, and the `attention_groups` hit-lookup". The offloading config is
that assert, and it does not apply the filter.

Note also that `hashes_per_chunk` would be `4 // 64 == 0` for such a group, so the assert
is guarding a genuine downstream division — the fix is to exclude the group, not relax the
check.

## Suggested fix

Filter by `participates_in_prefix_caching` in `offloading/config.py`, mirroring
`resolve_kv_cache_block_sizes`. **This is not a one-line change**: `config.groups` is
consumed positionally, so the same filter must be applied everywhere an index derived from
`spec.tokens_per_block` is used to index `kv_cache_config.kv_cache_groups`. We found five
such crossings:

| file | site |
|---|---|
| `offloading/config.py` | group list construction (the assert) |
| `offloading/worker.py` | `CanonicalKVCacheRef` list, one per group (packed path) |
| `offloading/scheduler.py` | `resolve_mamba_align_size` — `kv_cache_groups[idx]` |
| `offloading/scheduler.py` | `from_spec` full-attention scan, eagle-group set, `GroupOffloadConfig` loop |
| `v1/kv_offload/cpu/gpu_worker.py` | `assert len(group_sizes) == len(self.layer_refs_per_group)` |

Patching the first four surfaces the fifth at runtime, and vLLM core also passes
per-group block-id tuples covering *all* groups
(`assert len(new_block_id_groups) == len(self.group_states)`), which needs narrowing too.
A cleaner fix may be to keep the group list whole and carry an explicit
"offloadable" flag plus an index map, rather than filtering.

## Workaround (no patching)

Set the prefix-hash granularity so it divides the small group:

```
--prefix-match-unit 4
```

`resolve_kv_cache_block_sizes` validates the requested unit only against *participating*
groups (`64 % 4 == 0`), so it is accepted, and the assert then passes for every group.
Offloading initialises and stores work with stock vLLM.

**Cost:** hashing at 4-token instead of 64-token granularity multiplied our on-disk
footprint by roughly 7.5× (~700 KB per token stored, versus ~11 KB/token of GPU KV). On a
unified-memory machine that amplification is what made offloading unusable for us, so the
workaround is not a substitute for the fix.

## Secondary observation

With speculative decoding enabled and no group self-identifying as a draft group,
`scheduler.py` does `eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))`,
logging:

```
KV offloading: EAGLE/MTP draft attention groups [0,1,2,3,4,5,6] detected.
The trailing chunk of these groups will be excluded from offloading due to volatility.
```

i.e. **every** group is treated as volatile draft KV. Disabling speculation removes the
classification entirely. If that fallback is intended to be conservative it may be worth
narrowing, since as written it applies draft-volatility semantics to ordinary
full-attention groups.

---

# Second defect: non-shareable KV groups cap every offload hit at 0

Same build and model. After working around the assert above with
`--prefix-match-unit 4`, offloading initialises and **stores work**, but no load
ever happens:

```
vllm:kv_offload_total_bytes_total{...,transfer_type="GPU_to_CPU"} = 1.18e+10   # 11.8 GB
vllm:kv_offload_total_bytes_total{...,transfer_type="CPU_to_GPU"} = 0.0        # always
vllm:kv_offload_tiering_block_hits_total{...,tier="1:..."}        = 0          # always
```

Every request re-prefills in full (`cached_tokens=0`).

## Root cause

`OffloadingConnectorScheduler.__init__` puts **every** KV cache group into the
lookup set:

```python
full_attention_groups: list[int] = []
sliding_window_groups: list[int] = []
for group_config in self.config.kv_group_configs:     # <-- unfiltered
    ...
self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
```

and `_lookup_complete_chunks` bails out as soon as *any* queried group reports
no hit chunks:

```python
if num_hit_chunks == 0:
    return 0
```

A group whose spec sets `participates_in_prefix_caching = False` holds nothing
shareable between requests, so it can never report a hit — and therefore caps
every lookup at 0.

vLLM core already knows this. `KVCacheCoordinator.verify_and_split_kv_cache_groups`:

```python
# Skip groups that opt out of prefix caching (e.g. GLM5Next's kpool
# tail): their blocks are per-request scratch, never shareable, so they
# must not participate in hit lookup (their manager-level hooks already
# no-op). Their slot in the per-group hit tuple stays empty.
if not g.kv_cache_spec.participates_in_prefix_caching:
    continue
```

and `KpoolTailSpec.participates_in_prefix_caching` names this exact failure as
the reason the property exists:

```python
# Exclude it from the structural prefix-caching machinery so it neither
# drags the global block size down to ``kpool`` nor CAPS THE HYBRID HIT AT 0.
```

The offloading connector's hit lookup is the third consumer that
`KVCacheSpec.participates_in_prefix_caching`'s own docstring says must apply the
filter ("the global `cache_config.block_size` min, the hybrid divisibility
assert, and the `attention_groups` hit-lookup"). It applies neither this filter
nor the one in defect 1.

## Evidence

GLM-5.3-Flash has six groups; ours reports:

```
managers=[('FullAttentionManager', 3328, True), ('KpoolTailManager', 4, False),
          ('MambaManager', 3328, True) x4]        # (name, block_size, participates)
```

Group 1 is the scratch group. Counting the files the FS tier wrote, per group
directory (`..._g<idx>/`), after 15 requests:

| group | manager | participates | files stored |
|---|---|---|---|
| 0 | FullAttention | True | 47 |
| **1** | **KpoolTail** | **False** | **2** |
| 2-5 | Mamba | True | 47 each |

Group 1 is queried on every lookup and has essentially nothing to return, so
`_lookup_complete_chunks` returns 0 every time.

## Fix

Mirror core, in three places. Note that `_lookup_groups` /
`_sliding_window_groups` hold group *indices*, so omitting an index disturbs no
positional layout — `OffloadingConfig.groups` must be left intact (see the
warning below).

1. `__init__` — record the non-shareable groups, drop them from
   `_lookup_groups` / `_sliding_window_groups`.
2. store collection in `build_connector_meta` — skip their chunks. Nothing can
   look them up, so storing them is pure cost. The downstream per-group loop is
   driven by `keys_to_store` membership and emits `group_sizes.append(0)` for
   them on its own.
3. `update_state_after_alloc` — do not request their keys on load (never
   stored => guaranteed miss) and append `0` to `group_sizes` / `block_indices`
   so the worker's positional lists stay aligned.
   `CPUGPUOffloadingWorker._transfer` already short-circuits
   `if group_size == 0: continue`, and its `src_offset == num_src_blocks` /
   `dst_offset == num_dst_blocks` asserts still balance because no block ids are
   contributed either.

### Do NOT filter `OffloadingConfig.groups`

Our first attempt did, and it desynchronises at least five positional consumers:
`file_mapper.py` (`_g<group_idx>` path), `offloading/base.py`
(`tokens_per_block`), `scheduler.resolve_mamba_align_size`
(`kv_cache_groups[idx]`), the worker's per-group `CanonicalKVCacheRef` lists, and
`cpu/gpu_worker.py`'s `assert len(group_sizes) == len(self.layer_refs_per_group)`.
Core also hands the connector per-group block-id tuples covering every group
(`assert len(new_block_id_groups) == len(self.group_states)`).

### One caveat worth checking upstream

`_lookup_complete_chunks` applies the mamba hit-window alignment only inside
`if self._sliding_window_groups:`. On GLM-5.3-Flash the scratch group is the only
`SlidingWindowSpec`, so filtering it empties that tuple and the `round_down` to
`_mamba_align_size` stops running. It is harmless here because every remaining
group has `tokens_per_chunk == 3328`, which makes the hit naturally aligned, but
the alignment looks like it should be gated on `_mamba_align_size is not None`
rather than on the presence of sliding-window groups.

## Result

With the fix, on the same probe (seed a 15,523-token session, evict it with 14
larger sessions, re-send it):

| | before | after |
|---|---|---|
| `cached_tokens` on re-send | 0 | **13,312** |
| `CPU_to_GPU` bytes | 0 | **349 MB** |
| `tiering_block_hits{tier=fs}` | 0 | **8** |
| wall time | 16.26 s | **5.63 s** |

13,312 == 4 x 3328, i.e. every complete chunk of the prompt was restored.

---

# Third defect: a secondary tier silently corrupts KV when the TP group spans nodes

This one produces **wrong output with no error**, so it matters more than the
other two.

`FileSystemTierManager` is constructed with a memoryview of the *primary* CPU
tier (`tiering/spec.py`: `primary_kv_view = primary_tier.get_kv_memoryview()`),
and both directions operate on it alone:

```python
def submit_store(self, job_metadata):        # CPU -> disk
    task = functools.partial(batch_store_block, paths, self._primary_kv_view, ...)
def submit_load(self, job_metadata):         # disk -> CPU
    batch_load_block(paths, self._primary_kv_view, ...)
```

That CPU tier is a per-node `/dev/shm` mmap
(`shared_offload_region.py`: `Created mmap file /dev/shm/vllm_offload_<engine_id>.mmap`),
and the secondary tier is instantiated **scheduler-side only** — in EngineCore
and the API server, never in the workers.

So when a tensor-parallel group spans hosts (`--nnodes 2`, one `vllm serve`
process per node):

* rank 0's shard is cascaded CPU -> disk and promoted disk -> CPU normally;
* rank 1's CPU region is on the other host and is never touched by any FS tier;
* the scheduler nonetheless tracks one logical block set and issues the load to
  every worker (`pending_count = num_workers`);
* each worker copies CPU -> GPU from **its own** region, so rank 1 copies
  whatever stale bytes occupy those slots.

The request is then marked as having a valid cached prefix while half the heads
hold garbage. Measured on our pair: the restore reported `cached_tokens=13312`
and completed in 5.63 s instead of 16.26 s, and the model emitted 512
consecutive tokens of neither `content` nor `reasoning_content` — the same
prompt answers correctly on a cold prefill and on a GPU prefix-cache hit.
Confirmation that only one rank participates:

```
head   : 15 GB under kvoffload/, directories ..._r0/...
worker : kvoffload/ empty, 0 files
```

`replicated_layout` in `offloading/config.py` already refuses to engage when
`parallel_config.nnodes_within_dp != 1`, with the comment "Shared /dev/shm mmap
layout is single-node mp only" — so the single-node assumption is understood in
one place but not enforced where it silently changes results.

**Suggested fix:** refuse to configure a secondary tier when the KV-parallel
group spans nodes (fail closed at startup), or instantiate the tier per rank so
each cascades its own CPU region. A CPU-only tier is unaffected, because the
scheduler drives every rank's region symmetrically.

### We implemented the per-rank option, out-of-tree

`peer_kv_agent.py` + `PeerMirroredFileSystemTierManager` (in
`vllm_bounded_fs_tier_peer.py`) make the disk tier correct on a multi-host TP
group without patching vLLM, using the supported out-of-tree
`secondary_tiers[].module_path` extension point.

The head-side tier mirrors every cascade and promotion to an agent running in
each remote rank's container. The agent performs byte-identical I/O against
*that* rank's own `/dev/shm` region and its own `_r<rank>` files —
`FileMapper.base_path` excludes rank ("rank lives outside the hash"), so the
shards are siblings in one namespace. Crucially the head does not release a
job's `JobResult` until every peer has acknowledged, which keeps disk -> CPU
ordered before CPU -> GPU on all ranks. A peer failure fails the job, so
`mark_miss` turns it into a recompute — a wasted prefill instead of a silently
half-restored prefix.

Measured on the same probe, restores served from disk (8 FS-tier block hits per
session, both ranks holding 220 `.bin` files / 11 GB):

| session | cold TTFT | restored TTFT | speedup | cached | needle |
|---|---|---|---|---|---|
| 1 | 15.45 s | 2.59 s | **6.0x** | 13,312 | OK |
| 2 | 15.57 s | 2.67 s | **5.8x** | 13,312 | OK |

No cross-session contamination. Before the per-rank cascade the identical
configuration returned 512 consecutive empty tokens.

Known limitations of our workaround, all easy to lift and none affecting
correctness: the peer's disk is unbounded (only the head tier enforces
`max_bytes`, and a peer-side miss degrades to a recompute); each rank stores
whole block rows, so the other rank's inert half doubles disk usage; and the CPU
primary tier must be large enough to accept a promotion — a 4 GiB tier against a
1 GiB GPU pool produced `kv_offload_tiering_promotion_allocation_failures_total`
and every lookup missed, while 8 GiB worked.

---

# Fourth defect: the CPU offload region is never unlinked, and it breaks the next boot

`SharedOffloadRegion` creates `/dev/shm/vllm_offload_<engine_id>.mmap` and nothing
removes it when the engine stops or crashes. Every boot mints a fresh
`engine_id`, so the orphans accumulate — one per run, at `cpu_bytes_to_use` each.

On a unified-memory machine (NVIDIA GB10) that is not merely wasted tmpfs: shm
counts against the free device memory vLLM measures at startup, so after a few
restarts the boot gate fails outright:

```
ValueError: Free memory on device cuda:0 (102.79/121.63 GiB) on startup is less
than desired GPU memory utilization (0.85, 103.38 GiB).
```

We hit exactly this with 8.1 GB of orphaned regions from earlier runs. It is
self-inflicted denial of service after a crash loop, and the error message points
the operator at `gpu_memory_utilization`, which is the wrong lever.

**Suggested fix:** unlink the region on graceful shutdown, and on startup reap
regions whose `engine_id` belongs to no live engine. Our deployment works around
it by clearing `/dev/shm/vllm_offload_*.mmap` on both hosts before launch.

---

# Fifth issue: one uniform region row size makes fine-grained groups pathological

Not a bug so much as a scaling cliff worth documenting. The offload region uses a
single row size for every KV group (derived from an average over all blocks), but
a group's *store rate* is set by its own `tokens_per_chunk`. A group whose chunk
is far smaller than the full-attention alignment therefore writes a full-size row
per tiny chunk.

Measured on GLM-5.3-Flash with DFlash2 speculative decoding, `block_size` =
54.33 MB:

| group | manager | block_size | files written |
|---|---|---|---|
| 0 | FullAttention | 3584 | 28 |
| 1 | KpoolTail | 4 | 0 (excluded, defect 2) |
| 2-5 | Mamba | 3584 | 28 each |
| **6** | **SlidingWindow (DFlash2 drafter)** | **64** | **1276** |

Group 6 alone was 90% of 72 GB — it emits a 54.33 MB row every 64 tokens,
~683 KB per token per rank, against ~11 KB/token in the GPU pool. Excluding it
dropped the same workload from **72 GB to 7.1 GB** with restores unaffected
(draft KV is verified by the target model, so not restoring it costs acceptance
rate while it refills, never correctness — and vLLM already treats these groups'
trailing chunk as volatile for the same reason).

**Suggested fix:** either size rows per group, or skip groups whose
`alignment_chunk_count is not None` (vLLM's own marker for this shape) by
default, or at minimum warn when a group's projected store volume exceeds the
full-attention group's by an order of magnitude.

## Cost model worth knowing before enabling offload at all

Even with both pathological groups excluded, the remaining 5 groups cost
~76 KB/token per rank (5 x 54.33 MB per 3584-token chunk) because the Mamba
groups checkpoint recurrent state. Against ~11 KB/token in the GPU pool that is
~7x. Where the CPU tier and the GPU pool draw on the *same* unified memory, every
GiB moved from pool to tier loses ~97k instantly-available pool tokens and buys
only a ~14k-token restore window — so the tier should be sized as a deliberate
capacity trade, and the disk tier is the part that actually adds capacity.
