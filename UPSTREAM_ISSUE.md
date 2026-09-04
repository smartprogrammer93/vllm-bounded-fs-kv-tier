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


# Third defect: the CPU offload region is never unlinked, and it breaks the next boot

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

# Fourth issue: one uniform region row size makes fine-grained groups pathological

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
