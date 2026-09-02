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
