#!/usr/bin/env python3
"""Make vLLM's KV-offload READ path work on models with non-shareable KV groups.

THE BUG
-------
A KV cache group whose spec sets ``participates_in_prefix_caching = False``
holds nothing shareable between requests. On GLM5Next that is ``KpoolTailSpec``:
a 1-block-per-request circular scratch buffer (``block_size == index_kpool``,
overwritten by ``pos % kpool``).

vLLM core knows such groups must be kept out of hit lookup.
``KVCacheCoordinator.verify_and_split_kv_cache_groups``:

    # Skip groups that opt out of prefix caching (e.g. GLM5Next's kpool
    # tail): their blocks are per-request scratch, never shareable, so they
    # must not participate in hit lookup (their manager-level hooks already
    # no-op). Their slot in the per-group hit tuple stays empty.

and ``KpoolTailSpec.participates_in_prefix_caching`` states the consequence of
not doing so, in as many words:

    Exclude it from the structural prefix-caching machinery so it neither
    drags the global block size down to ``kpool`` nor CAPS THE HYBRID HIT AT 0.

``OffloadingConnectorScheduler`` never consults the property. Its ``__init__``
puts EVERY group into ``self._lookup_groups``, and ``_lookup_complete_chunks``
bails with ``return 0`` as soon as any queried group reports no hit chunks:

    if num_hit_chunks == 0:
        return 0

The scratch group can never report a hit, so every offload lookup returns 0.
Offloading becomes WRITE-ONLY: it burns CPU memory, disk and write bandwidth
and never serves a read. Symptom on the metrics endpoint --
``kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}`` climbs into the
tens of GB while ``{transfer_type="CPU_to_GPU"}`` stays at exactly 0.0 forever.

THE FIX
-------
Mirror what core already does, in the three places that need it:

  1. ``__init__`` -- record the non-shareable group indices and drop them from
     ``_lookup_groups`` / ``_sliding_window_groups``.
  2. store collection -- do not store their chunks. Nothing can ever look them
     up, so it is pure cost. The downstream per-group loop is driven by
     ``keys_to_store`` membership, so it emits ``group_sizes.append(0)`` for
     them by itself; no second edit needed there.
  3. ``update_state_after_alloc`` -- do not request their keys on load (never
     stored => guaranteed miss) and contribute an empty group so the worker's
     positional ``group_sizes`` / ``block_indices`` stay aligned.

WHY THIS IS POSITIONALLY SAFE (and the earlier attempt was not)
---------------------------------------------------------------
An earlier version of this patch filtered ``OffloadingConfig.groups`` itself.
That list is consumed POSITIONALLY in at least five places -- ``file_mapper.py``
builds the ``_g<group_idx>`` path from it, ``offloading/base.py`` derives
``tokens_per_block``, ``scheduler.resolve_mamba_align_size`` indexes
``kv_cache_groups[idx]``, the worker builds one ``CanonicalKVCacheRef`` list per
group, and ``cpu/gpu_worker.py`` asserts
``len(group_sizes) == len(self.layer_refs_per_group)``. Removing an entry
desynchronises all of them, and vLLM core also hands the connector per-group
block-id tuples covering every group
(``assert len(new_block_id_groups) == len(self.group_states)``).

This version leaves ``config.groups`` completely intact. ``_lookup_groups`` and
``_sliding_window_groups`` are tuples of group *indices* used to index
``config.kv_group_configs``, so omitting an index changes no layout. The load
path keeps emitting one ``group_sizes`` / ``block_indices`` entry per group,
using 0 for the skipped group -- and ``CPUGPUOffloadingWorker._transfer`` already
short-circuits ``if group_size == 0: continue``, while its
``assert src_offset == num_src_blocks`` / ``assert dst_offset == num_dst_blocks``
still balance because we contribute no block ids for that group either.

CORRECTNESS
-----------
Not restoring the scratch group is exactly what core does on a GPU prefix-cache
hit: it is excluded from hit lookup and ``KpoolTailManager`` no-ops the
manager-level cache hooks. Its contents are transient and, per its own
docstring, "never shareable across requests" -- there is nothing to restore.

Idempotent (unique sentinel per edit) and fails closed if upstream drifts.
Gated on GLM53_OFFLOAD_GROUP_FILTER=1; vLLM is left pristine otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SCHED = VLLM / "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
CONFIG = VLLM / "distributed/kv_transfer/kv_connector/v1/offloading/config.py"
CPUSPEC = VLLM / "v1/kv_offload/cpu/spec.py"

MARK = "[glm53-offload-nonshareable]"

# --- edit 1: __init__ -- compute the set, filter the lookup groups -----------
INIT_OLD = '''        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)
'''
INIT_NEW = '''        # LOCAL [glm53-offload-nonshareable]: groups that opt out of prefix
        # caching (GLM5Next's KpoolTailSpec, a 1-block/req circular scratch
        # buffer) hold nothing shareable, so they never receive a store and can
        # never report a hit. _lookup_complete_chunks returns 0 the moment any
        # queried group has no hit chunks, so leaving them in the lookup set
        # caps EVERY offload hit at 0 and offloading becomes write-only. vLLM
        # core already excludes them from its own hit lookup
        # (KVCacheCoordinator.verify_and_split_kv_cache_groups); mirror that.
        self._non_shareable_groups: frozenset[int] = frozenset(
            idx
            for idx, group in enumerate(kv_cache_config.kv_cache_groups)
            if not group.kv_cache_spec.participates_in_prefix_caching
        )
        if self._non_shareable_groups:
            logger.info(
                "KV offloading: groups %s opt out of prefix caching "
                "(per-request scratch, never shareable); excluded from "
                "lookup, store and load.",
                sorted(self._non_shareable_groups),
            )

        # LOCAL [glm53-offload-nonshareable] W27a diagnostic: the offload region
        # uses ONE uniform row stride for every group (an average over all
        # groups), but the scheduler stores one row per (chunk, group). Log each
        # group's REAL payload next to the row it is written into, so the waste
        # is a measured number rather than an inference. If real << row, per-group
        # row sizing (W27c) is worth building; if they are close, the cost is
        # genuine and we stop optimising.
        try:
            _row = int(self.config.kv_group_configs[0].tokens_per_block)
            for _g, _kvg in enumerate(kv_cache_config.kv_cache_groups):
                _spec = _kvg.kv_cache_spec
                _real = int(
                    getattr(_spec, "page_size_bytes", 0) * len(_kvg.layer_names)
                )
                logger.info(
                    "KV offloading [W27a]: group %d %s block_size=%s layers=%d "
                    "real_payload=%.2f MiB",
                    _g,
                    type(_spec).__name__,
                    getattr(_spec, "block_size", "?"),
                    len(_kvg.layer_names),
                    _real / 1024 ** 2,
                )
        except Exception:
            logger.warning("KV offloading [W27a]: payload diagnostic failed",
                           exc_info=True)

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.group_idx in self._non_shareable_groups:
                continue
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)
'''
INIT_SENTINEL = "_non_shareable_groups: frozenset[int] = frozenset("

# --- edit 2: store collection -- skip non-shareable groups -------------------
STORE_OLD = '''            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )

                start_chunk_idx = group_state.next_stored_chunk_idx
'''
STORE_NEW = '''            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                # LOCAL [glm53-offload-nonshareable]: excluded from
                # _lookup_groups, so nothing can ever look these up. Storing
                # them is pure cost (disk, bandwidth, CPU tier capacity).
                if group_config.group_idx in self._non_shareable_groups:
                    continue
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )

                start_chunk_idx = group_state.next_stored_chunk_idx
'''
STORE_SENTINEL = "# LOCAL [glm53-offload-nonshareable]: excluded from"

# --- edit 3: load path -- request no keys, contribute an empty group ---------
LOAD_OLD = '''            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            tokens_per_block = group_config.tokens_per_block
'''
LOAD_NEW = '''            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            # LOCAL [glm53-offload-nonshareable]: never stored, so asking for
            # their keys is a guaranteed miss. Contribute an empty group so the
            # worker's positional group_sizes/block_indices stay aligned: it
            # short-circuits `if group_size == 0: continue`, and its
            # src_offset/dst_offset asserts still balance because we add no
            # block ids for this group either.
            if group_config.group_idx in self._non_shareable_groups:
                group_sizes.append(0)
                block_indices.append(0)
                continue

            tokens_per_block = group_config.tokens_per_block
'''
LOAD_SENTINEL = "# LOCAL [glm53-offload-nonshareable]: never stored"


# --- edit 4: the divisibility assert must skip the same groups ------------
# Without this, OffloadingConnector cannot initialise at all unless the operator
# forces --prefix-match-unit down to the scratch group's block size (4), which
# drags the GPU prefix-cache hash granularity along with it. Filtering the
# assert instead lets a deployment keep its tuned grain (64 here) untouched.
# resolve_kv_cache_block_sizes() already excludes exactly these groups when it
# derives tokens_per_hash, and KVCacheSpec.participates_in_prefix_caching's
# docstring names "the hybrid divisibility assert" as a consumer that must too.
# NOTE: `groups` itself is left intact -- it is consumed positionally.
ASSERT_OLD = """    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    for group in groups:
        assert group.tokens_per_block % tokens_per_hash == 0, (
"""
ASSERT_NEW = """    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    for group, _kv_group in zip(groups, kv_cache_config.kv_cache_groups):
        # LOCAL [glm53-offload-nonshareable]: groups that opt out of prefix
        # caching never have block_hashes computed, so their block_size must not
        # constrain the hash granularity -- exactly the filter
        # resolve_kv_cache_block_sizes applies when deriving tokens_per_hash.
        if not _kv_group.kv_cache_spec.participates_in_prefix_caching:
            continue
        assert group.tokens_per_block % tokens_per_hash == 0, (
"""
ASSERT_SENTINEL = "for group, _kv_group in zip(groups, kv_cache_config.kv_cache_groups):"


# --- edit 5: keep hashes_per_chunk >= 1 so nothing can divide by zero -------
# The config.py assert we filter in edit 4 was also guarding real arithmetic:
# for the scratch group hashes_per_chunk == tokens_per_block // tokens_per_hash
# == 4 // 64 == 0, and three consumers break on that --
#   scheduler.update_offload_keys(): islice(hashes, -1, None, 0) -> ValueError,
#     and it iterates EVERY group, so this would raise on the first request;
#   events.py:193  tokens_per_hash = tokens_per_chunk // hashes_per_chunk -> ZeroDivisionError;
#   events.py:263  assert hashes_per_chunk > 0.
# Clamping to 1 keeps all of them well-defined. The value is only ever used to
# walk block_hashes and to label KV events; the scratch group is excluded from
# lookup, store and load by the other edits, so its keys are never consumed --
# but they are now built harmlessly instead of raising.
CLAMP_OLD = """                    hashes_per_chunk=(
                        (tokens_per_block * spec.blocks_per_chunk)
                        // spec.tokens_per_hash
                    ),
"""
CLAMP_NEW = """                    hashes_per_chunk=max(
                        1,
                        (tokens_per_block * spec.blocks_per_chunk)
                        // spec.tokens_per_hash,
                    ),
"""
CLAMP_SENTINEL = "hashes_per_chunk=max("


# --- edit 6: keep tiny-granularity SWA/draft groups out of offload ----------
# The offload region uses ONE uniform row size for every group (the max), so a
# group whose tokens_per_chunk is far below the full-attention alignment writes
# a full-size row per tiny chunk. Measured on GLM-5.3-Flash with DFlash2: the
# drafter group (SlidingWindowManager, block_size=64) produced 1276 of 1410
# files -- 90% of 72 GB -- because it emits a 54.33 MB row every 64 tokens.
# That is ~683 KB per token per rank against 11 KB/token in the GPU pool, which
# makes disk offload strictly worse than simply keeping a larger GPU pool.
#
# `alignment_chunk_count is not None` is vLLM's own marker for exactly this
# shape ("SWA groups have much smaller block sizes than the MLA full-attention
# group"), so it is the selector.
#
# Correctness: these are EAGLE/MTP draft groups. Draft tokens are verified by
# the target model, so draft KV that was not restored costs acceptance rate
# while it refills, never output correctness -- and vLLM already treats the
# trailing chunk of these groups as volatile for the same reason.
SWA_OLD = """        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.group_idx in self._non_shareable_groups:
                continue
"""
SWA_NEW = """        import os as _os

        if _os.environ.get("GLM53_OFFLOAD_SKIP_SMALL_SWA", "1") == "1":
            # Mamba groups must be EXEMPT: their tokens_per_chunk equals the
            # full-attention alignment, so upstream leaves their
            # alignment_chunk_count None and they are not the fine-grained shape
            # this exclusion targets. The Mamba snapshot-interval edit sets that
            # field deliberately, and without this exemption it would make the
            # Mamba groups match here and be dropped from offload entirely --
            # observed as "groups [2,3,4,5,6] are fine-grained", with the GPU
            # prefix cache masking the resulting missing recurrent state.
            _mamba_groups = frozenset(
                idx
                for idx, g in enumerate(kv_cache_config.kv_cache_groups)
                if isinstance(g.kv_cache_spec, MambaSpec)
            )
            _small_swa = frozenset(
                gc.group_idx
                for gc in self.config.kv_group_configs
                if gc.alignment_chunk_count is not None
                and gc.group_idx not in _mamba_groups
            )
            _added = sorted(_small_swa - self._non_shareable_groups)
            if _added:
                logger.info(
                    "KV offloading: groups %s are fine-grained SWA/draft groups "
                    "(tokens_per_chunk << full-attention alignment); excluded "
                    "from offload because the uniform region row size would "
                    "inflate storage by orders of magnitude. Draft KV is "
                    "verified by the target model, so this costs acceptance "
                    "rate while it refills, not correctness. Set "
                    "GLM53_OFFLOAD_SKIP_SMALL_SWA=0 to offload them anyway.",
                    _added,
                )
            self._non_shareable_groups = self._non_shareable_groups | _small_swa

            # EXPERIMENT ONLY (GLM53_OFFLOAD_EXCLUDE_MAMBA=1, default off).
            # Mamba groups are ~97% of the real offload payload, so whether a
            # restore actually NEEDS their recurrent state decides whether
            # offload can ever approach pool parity. This knob excludes them so
            # that can be measured directly. It is NOT a tuning option: if the
            # state is needed, enabling this restores attention KV with stale
            # recurrent state, which is silent wrong output.
            if _os.environ.get("GLM53_OFFLOAD_EXCLUDE_MAMBA") == "1":
                self._non_shareable_groups = (
                    self._non_shareable_groups | _mamba_groups
                )
                logger.warning(
                    "KV offloading: EXPERIMENT -- Mamba groups %s excluded from "
                    "offload (GLM53_OFFLOAD_EXCLUDE_MAMBA=1). If recurrent state "
                    "is required for a restore, output will be silently WRONG.",
                    sorted(_mamba_groups),
                )

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.group_idx in self._non_shareable_groups:
                continue
"""
SWA_SENTINEL = "GLM53_OFFLOAD_SKIP_SMALL_SWA"

W27A_SENTINEL = "KV offloading [W27a]"


# --- edit 7: region rows are sized for world_size copies but only one is used -
# The CPU offload region is ONE FILE PER NODE
# (/dev/shm/vllm_offload_<engine_id>.mmap) and a worker's slot index inside it is
#     rank = torch.accelerator.current_device_index() % world_size
# (tiering/spec.py::create_worker). That is only the global rank when every
# worker sits on one node. On a multi-node TP group with one GPU per host,
# current_device_index() is 0 everywhere, so EVERY worker uses slot 0 and the
# remaining world_size-1 slots of every row are never written.
#
# Verified by dumping both regions on a 2-node TP=2 pair: slot 0 populated,
# slot 1 all-zero on BOTH hosts. Since cpu/spec.py still sizes each row as
# world_size * per-worker bytes, (world_size-1)/world_size of cpu_bytes_to_use
# is dead -- half of it here. Making num_copies the number of workers that
# actually SHARE a region doubles the blocks a given cpu_bytes_to_use buys, and
# the CPU tier is what bounds a restore, so it doubles the restore window.
#
# Opt-in and clamped: GLM53_OFFLOAD_REGION_COPIES=<n>, unset = upstream
# behaviour. Set it to the number of GPUs per node. Setting it BELOW that would
# make co-located workers share one slot and corrupt each other, so it is not
# defaulted.
COPIES_OLD = """            num_copies = 1 if self.replicated_layout else world_size
"""
COPIES_NEW = """            num_copies = 1 if self.replicated_layout else world_size
            # LOCAL [glm53-offload-nonshareable]: see the patch docstring --
            # the region is per-node and the slot index is the LOCAL device
            # index, so rows sized by world_size waste
            # (world_size-1)/world_size of cpu_bytes_to_use on a multi-node TP
            # group. Must equal the number of workers sharing one region.
            import os as _os

            _copies_env = _os.environ.get("GLM53_OFFLOAD_REGION_COPIES")
            if _copies_env and not self.replicated_layout:
                num_copies = max(1, min(int(_copies_env), world_size))
"""
COPIES_SENTINEL = "GLM53_OFFLOAD_REGION_COPIES"


# --- edits 8+9: snapshot Mamba state every K chunks instead of every chunk ----
# THE COST: on a hybrid model the Mamba groups dominate offload storage -- 76 of
# the 78.7 MiB of real payload per 3584-token chunk here -- because offload
# snapshots recurrent state at EVERY chunk boundary while the live GPU pool keeps
# only the current state per request. That is why offload cannot reach pool
# parity by layout work alone.
#
# WHY IT IS SAFE TO SPARSIFY: vLLM already treats a Mamba group as a
# sliding-window group of exactly ONE chunk ("Mamba depends on a single state",
# get_sliding_window_size_in_chunks), so _sliding_window_lookup scans from the
# END for a single hit -- it never needs the interior chunks. And
# _lookup_complete_chunks already rounds the hit window DOWN to
# _mamba_align_size, so a hit can be forced to land exactly where a snapshot
# exists. Setting alignment_chunk_count=K makes the existing, tested
# is_store_reachable_swa_chunk keep only the last chunk of each K-segment
# (position_in_segment >= actual_segment_length - 1 with a reachable_tail of 1).
#
# THE TRADE: a restore rounds down to a multiple of K chunks, so up to
# (K-1) * tokens_per_chunk tokens are re-prefilled. Stored rows per chunk go from
# 1 + n_mamba to 1 + n_mamba/K, which is what widens the restore window:
# K=2 -> 1.67x, K=4 -> 2.5x, K=8 -> 3.33x on this model.
#
# Both halves MUST move together. Storing 1-in-K without widening the alignment
# would let a hit land on an unstored boundary and restore attention KV with no
# matching recurrent state -- silent wrong output. Guarded by one knob:
# GLM53_OFFLOAD_MAMBA_SNAPSHOT_CHUNKS (default 1 = upstream behaviour).
MAMBA_ALIGN_OLD = """    mamba_align_size: int | None = None
    for idx, tokens_per_block in enumerate(spec.tokens_per_block):
"""
MAMBA_ALIGN_NEW = """    import os as _os

    # LOCAL [glm53-offload-nonshareable]: see the patch docstring -- widen the
    # hit-window alignment by the snapshot interval so a hit can only land on a
    # boundary where a Mamba snapshot was actually stored.
    _snap = max(1, int(_os.environ.get("GLM53_OFFLOAD_MAMBA_SNAPSHOT_CHUNKS", "1")))
    mamba_align_size: int | None = None
    for idx, tokens_per_block in enumerate(spec.tokens_per_block):
"""
# Unique to THIS edit: the env var name alone also appears in the knob edit
# below, so using it here would depend on list order to apply at all.
MAMBA_ALIGN_SENTINEL = "_snap = max(1, int("

MAMBA_RET_OLD = """            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size
"""
MAMBA_RET_NEW = """            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    if mamba_align_size is not None and _snap > 1:
        mamba_align_size *= _snap
    return mamba_align_size
"""
MAMBA_RET_SENTINEL = "mamba_align_size *= _snap"

MAMBA_STORE_OLD = """                    alignment_chunk_count=_alignment_chunk_count(
                        tokens_per_block * spec.blocks_per_chunk, sw
                    ),
"""
MAMBA_STORE_NEW = """                    alignment_chunk_count=(
                        # LOCAL [glm53-offload-nonshareable]: keep only the last
                        # chunk of each K-segment for Mamba state; the hit window
                        # is widened to match in resolve_mamba_align_size.
                        _mamba_snapshot_chunks
                        if (
                            _mamba_snapshot_chunks > 1
                            and isinstance(kv_spec, MambaSpec)
                        )
                        else _alignment_chunk_count(
                            tokens_per_block * spec.blocks_per_chunk, sw
                        )
                    ),
"""
# Unique to THIS edit: "_mamba_snapshot_chunks" alone is introduced by the
# knob edit above, so using it here silently skipped this edit entirely.
MAMBA_STORE_SENTINEL = "chunk of each K-segment for Mamba state"

MAMBA_KNOB_OLD = """        eagle_groups = {
            idx
            for idx, g in enumerate(kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }
"""
MAMBA_KNOB_NEW = """        import os as _os2

        _mamba_snapshot_chunks = max(
            1, int(_os2.environ.get("GLM53_OFFLOAD_MAMBA_SNAPSHOT_CHUNKS", "1"))
        )
        if _mamba_snapshot_chunks > 1:
            logger.info(
                "KV offloading: snapshotting Mamba state every %d chunks "
                "(hit window rounds down to match). Cuts offload storage on "
                "hybrid models; costs up to %d-1 chunks of re-prefill.",
                _mamba_snapshot_chunks,
                _mamba_snapshot_chunks,
            )

        eagle_groups = {
            idx
            for idx, g in enumerate(kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }
"""
MAMBA_KNOB_SENTINEL = "_mamba_snapshot_chunks = max("

CONFIG_EDITS = [
    ("config:assert-filter", ASSERT_OLD, ASSERT_NEW, ASSERT_SENTINEL),
]

CPUSPEC_EDITS = [
    ("cpuspec:region-copies", COPIES_OLD, COPIES_NEW, COPIES_SENTINEL),
]

EDITS = [
    ("sched:init-filter", INIT_OLD, INIT_NEW, INIT_SENTINEL),
    ("sched:store-skip", STORE_OLD, STORE_NEW, STORE_SENTINEL),
    ("sched:load-skip", LOAD_OLD, LOAD_NEW, LOAD_SENTINEL),
    ("sched:hashes-clamp", CLAMP_OLD, CLAMP_NEW, CLAMP_SENTINEL),
    ("sched:small-swa-skip", SWA_OLD, SWA_NEW, SWA_SENTINEL),
    ("sched:mamba-align-knob", MAMBA_ALIGN_OLD, MAMBA_ALIGN_NEW,
     MAMBA_ALIGN_SENTINEL),
    ("sched:mamba-align-widen", MAMBA_RET_OLD, MAMBA_RET_NEW,
     MAMBA_RET_SENTINEL),
    ("sched:mamba-snap-knob", MAMBA_KNOB_OLD, MAMBA_KNOB_NEW,
     MAMBA_KNOB_SENTINEL),
    ("sched:mamba-snap-store", MAMBA_STORE_OLD, MAMBA_STORE_NEW,
     MAMBA_STORE_SENTINEL),
]


def apply_edits(path: Path, edits=None) -> str:
    if not path.is_file():
        return f"MISSING {path} - not patched"
    src = path.read_text()
    applied, skipped = [], []
    for label, old, new, sentinel in (EDITS if edits is None else edits):
        if sentinel in src:
            skipped.append(label)
            continue
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"{MARK} FAIL {label}: anchor matched {n} times (expected 1). "
                f"Upstream source drifted; refusing to patch."
            )
        src = src.replace(old, new)
        applied.append(label)
    if applied:
        compile(src, str(path), "exec")  # fail closed on a broken result
        path.write_text(src)
    parts = []
    if applied:
        parts.append("applied " + ",".join(applied))
    if skipped:
        parts.append("already " + ",".join(skipped))
    return "; ".join(parts) or "nothing to do"


def main() -> int:
    if os.environ.get("GLM53_OFFLOAD_GROUP_FILTER", "0") != "1":
        print(f"{MARK} disabled (GLM53_OFFLOAD_GROUP_FILTER != 1); vLLM pristine")
        return 0
    print(f"{MARK} {apply_edits(SCHED)}")
    print(f"{MARK} {apply_edits(CONFIG, CONFIG_EDITS)}")
    print(f"{MARK} {apply_edits(CPUSPEC, CPUSPEC_EDITS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
