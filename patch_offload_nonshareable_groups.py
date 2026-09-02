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

EDITS = [
    ("sched:init-filter", INIT_OLD, INIT_NEW, INIT_SENTINEL),
    ("sched:store-skip", STORE_OLD, STORE_NEW, STORE_SENTINEL),
    ("sched:load-skip", LOAD_OLD, LOAD_NEW, LOAD_SENTINEL),
]


def apply_edits(path: Path) -> str:
    if not path.is_file():
        return f"MISSING {path} - not patched"
    src = path.read_text()
    applied, skipped = [], []
    for label, old, new, sentinel in EDITS:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
