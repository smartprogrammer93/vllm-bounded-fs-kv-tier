#!/usr/bin/env python3
"""W30: streamed restore -- the CPU tier stops bounding the restore size.

WHY
---
vLLM stages every hit block in the CPU tier before one CPU->GPU load, so
`tier size == maximum restorable prefix`. Measured: 9 blocks -> `cached=0`;
39 -> ~25k tokens; 79 -> ~50k. That is why the tier costs gigabytes, and it is a
design choice, not a hardware limit -- the copy out of device memory is
unavoidable on GB10, but the size of the buffer is not.

A restore issued as K successive jobs through a small fixed buffer would remove
the cap. Measured cost of doing so (probes/w30_probe.py):

    a (fixed per job) = -0.028 ms      i.e. zero
    b (marginal)      = 0.0203 ms/MiB  -> 48.2 GB/s, r2 = 1.000

plus one scheduler round trip per job, measured at 14.6 ms per decode step. So
1400 MiB through a 64 MiB buffer costs ~330 ms against a cold prefill of seconds.

THE BLOCKER THIS EDIT REMOVES
-----------------------------
`OffloadingConnectorWorker.get_finished` reports `finished_recving` for a request
as soon as ANY of its load jobs completes:

    req_id = self._load_jobs.pop(job_id, None)
    if req_id is not None:
        finished_recving.add(req_id)

The base scheduler then resumes the request. With a chained restore that would
run the model against a partially-restored prefix -- silent corruption, the
exact failure signature seen elsewhere in this work (a request returning tokens
that are neither content nor reasoning).

So the worker must be told which job is the LAST one for a request, and release
the request only then. This patch adds that and nothing else: it is the
primitive the batching pipeline is built on, and it is a no-op until a scheduler
actually marks a job non-final.

WHAT THE THREE STEPS DO
-----------------------
step 1  -- the primitive above: a restore may span several load jobs.
step 2a -- split the restore into one job per KV group. Valid because
           `CPUGPUOffloadingWorker._transfer` walks groups positionally and
           short-circuits on `group_size == 0`, so a job carrying one group
           works if `group_sizes`/`block_indices` keep full length with zeros
           elsewhere.
step 2b -- stop the tier from bounding the restore. `TieringOffloadingManager
           .lookup` promoted every hit block during the prefix scan and returned
           MISS once the tier was full, which is precisely what made tier size
           the maximum restorable prefix. It now reports the hit and the
           connector promotes each batch immediately before issuing its load,
           so only ONE batch need ever be resident.

           This cannot deadlock: `prepare_load` pins a block with `ref_cnt += 1`
           and `complete_load` drops it back to 0, returning it to the evictable
           set, so batch k's blocks are reusable the moment its load completes.
           Progress is guaranteed as long as a single batch fits.

           The residual risk is liveness, not correctness. Once lookup reports
           the larger hit, vLLM has committed to it, and a queued batch holds no
           reference -- so another request's stores could in principle evict its
           blocks from the disk tier mid-restore. The driver `touch()`es every
           queued key each step to keep them most-recently-used against exactly
           that; if a batch stalls anyway, that one request hangs, loudly and
           logged, and self-heals when the client disconnects. It never runs the
           model against a prefix that was not restored.

Knobs: GLM53_OFFLOAD_STREAM_RESTORE=1 enables everything (vLLM is untouched
otherwise); GLM53_OFFLOAD_STREAM_SEQUENTIAL=0 reverts to step 2a's behaviour of
issuing all batches up front, which keeps the tier as the bound.

Idempotent, sentinel-guarded, fails closed on drift. The metadata field defaults
to None so metadata built by an unpatched scheduler behaves exactly as before.
tests/test_streaming_patch.py applies the whole chain to pristine upstream
sources offline and checks anchors, ordering, indentation and idempotence; run
it before any container start.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
COMMON = VLLM / "distributed/kv_transfer/kv_connector/v1/offloading/common.py"
WORKER = VLLM / "distributed/kv_transfer/kv_connector/v1/offloading/worker.py"

MARK = "[glm53-offload-streaming]"

# --- edit 1: carry the flag on the scheduler -> worker metadata --------------
META_OLD = '''@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None
'''
META_NEW = '''@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None
    # LOCAL [glm53-offload-streaming]: load jobs that are NOT the last one for
    # their request. The worker must not report finished_recving for these, or
    # the base scheduler resumes the request against a partially restored
    # prefix. None/empty == every load job is final, i.e. upstream behaviour.
    nonfinal_load_jobs: set[int] | None = None
'''
META_SENTINEL = "nonfinal_load_jobs"

# --- edit 2: worker records which jobs are non-final -------------------------
REG_OLD = '''        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
'''
REG_NEW = '''        # LOCAL [glm53-offload-streaming]: remember which loads are not the
        # last for their request, so get_finished() withholds the request.
        _nonfinal = getattr(metadata, "nonfinal_load_jobs", None)
        if _nonfinal:
            self._nonfinal_load_jobs.update(_nonfinal)

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
'''
REG_SENTINEL = "remember which loads are not the"

# --- edit 3: withhold the request until its final load lands -----------------
FIN_OLD = '''            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
'''
FIN_NEW = '''            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                # LOCAL [glm53-offload-streaming]: a chained restore issues
                # several load jobs per request. Releasing the request on the
                # first would run the model against a partially restored prefix
                # -- silent wrong output. Only the final job resumes it; the
                # scheduler still sees every completion via completed_jobs.
                if job_id in self._nonfinal_load_jobs:
                    self._nonfinal_load_jobs.discard(job_id)
                else:
                    finished_recving.add(req_id)
'''
FIN_SENTINEL = "a chained restore issues"

# --- edit 4: the set itself, and its reset ----------------------------------
INIT_OLD = '''        self._load_jobs: dict[int, ReqId] = {}
'''
INIT_NEW = '''        self._load_jobs: dict[int, ReqId] = {}
        # LOCAL [glm53-offload-streaming]: load jobs that must not release
        # their request yet (see get_finished).
        self._nonfinal_load_jobs: set[int] = set()
'''
INIT_SENTINEL = "_nonfinal_load_jobs: set[int] = set()"

CLEAR_OLD = '''        self._load_jobs.clear()
'''
CLEAR_NEW = '''        self._load_jobs.clear()
        self._nonfinal_load_jobs.clear()
'''
CLEAR_SENTINEL = "self._nonfinal_load_jobs.clear()"

# ===========================================================================
# W30 step 2a: issue the restore as one load job PER KV GROUP, chained.
#
# A single job requires every hit block resident in the CPU tier at once. This
# splits it per group, which is valid because CPUGPUOffloadingWorker._transfer
# walks groups positionally and short-circuits `if group_size == 0: continue` --
# so a job carrying only group g's blocks works as long as group_sizes and
# block_indices keep their full length, with zeros elsewhere. Its
# src_offset/dst_offset asserts still balance because only group g contributes
# block ids.
#
# All but the last job are marked non-final, so the worker withholds
# finished_recving (step 1) and the request is not resumed against a partially
# restored prefix.
#
# NOTE ON SCOPE: this alone does NOT shrink the CPU tier, because promotion
# still happens during _lookup, which pulls every hit block in before
# update_state_after_alloc runs. It exists to prove the chained-load path
# end-to-end -- correct output with K>1 jobs -- which is the prerequisite for
# step 2b, where promotion becomes per-batch and the tier can shrink to a fixed
# buffer. Sequencing it this way is deliberate: the load path is where this
# work has produced silent corruption twice.
# ===========================================================================

SCHED = VLLM / "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"

SPANINIT_OLD = """        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
"""
SPANINIT_NEW = """        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        # LOCAL [glm53-offload-streaming]: (key_lo, key_hi, blk_lo, blk_hi) per
        # group, so the flat lists can be sliced into per-group load jobs
        # without restructuring the accumulation below.
        _stream_spans: list[tuple[int, int, int, int]] = []
"""
SPANINIT_SENTINEL = "_stream_spans: list[tuple[int, int, int, int]] = []"

SPANSTART_OLD = """            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )
"""
SPANSTART_NEW = """            _k0, _d0 = len(keys_to_load), len(dst_block_ids)
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )
"""
SPANSTART_SENTINEL = "_k0, _d0 = len(keys_to_load), len(dst_block_ids)"

SPANSKIP_OLD = """            if group_config.group_idx in self._non_shareable_groups:
                group_sizes.append(0)
                block_indices.append(0)
                continue
"""
SPANSKIP_NEW = """            if group_config.group_idx in self._non_shareable_groups:
                group_sizes.append(0)
                block_indices.append(0)
                # keep one span per group so span index == group index
                _stream_spans.append((_k0, _k0, _d0, _d0))
                continue
"""
SPANSKIP_SENTINEL = "keep one span per group so span index == group index"

SPANEND_OLD = """            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)
"""
SPANEND_NEW = """            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)
            _stream_spans.append((_k0, len(keys_to_load), _d0, len(dst_block_ids)))
"""
SPANEND_SENTINEL = "_stream_spans.append((_k0, len(keys_to_load)"

ISSUE_OLD = """        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )
"""
ISSUE_NEW = """        # LOCAL [glm53-offload-streaming]: one chained load job per KV group.
        assert not req_status.transfer_jobs
        self._issue_chained_load(
            request, req_status, keys_to_load, dst_block_ids,
            group_sizes, block_indices, _stream_spans,
        )
"""
ISSUE_SENTINEL = "one chained load job per KV group"

HELPER_OLD = """    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
"""
HELPER_NEW = '''    def _issue_chained_load(
        self, request, req_status, keys_to_load, dst_block_ids,
        group_sizes, block_indices, spans,
    ) -> None:
        """LOCAL [glm53-offload-streaming]: emit the restore as one job per group.

        group_sizes/block_indices keep full length with zeros outside the group
        being loaded, which the worker skips. Every job but the last is marked
        non-final so the request is not resumed against a partial prefix.
        """
        batches = []
        for gi, (k0, k1, d0, d1) in enumerate(spans):
            if d1 <= d0:
                continue
            gs = [0] * len(group_sizes)
            bi = [0] * len(block_indices)
            gs[gi] = group_sizes[gi]
            bi[gi] = block_indices[gi]
            batches.append((keys_to_load[k0:k1], dst_block_ids[d0:d1], gs, bi))

        if not batches:
            return
        for idx, (keys, blocks_, gs, bi) in enumerate(batches):
            src_spec = self.manager.prepare_load(keys, req_status.req_context)
            dst_spec = GPULoadStoreSpec(
                blocks_, group_sizes=gs, block_indices=bi
            )
            job_id = self._generate_job_id()
            self._current_batch_load_jobs[job_id] = TransferJob(
                req_id=request.request_id,
                src_spec=src_spec,
                dst_spec=dst_spec,
            )
            req_status.transfer_jobs.add(job_id)
            self._jobs[job_id] = TransferJobStatus(
                req_id=request.request_id,
                pending_count=self.config.num_workers,
                keys=set(keys),
                is_store=False,
            )
            if idx < len(batches) - 1:
                self._stream_nonfinal_jobs.add(job_id)
        logger.debug(
            "Request %s: restore issued as %d chained load job(s)",
            request.request_id, len(batches),
        )

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
'''
HELPER_SENTINEL = "def _issue_chained_load("

NFSET_OLD = """        self._req_status: dict[ReqId, RequestOffloadState] = {}
"""
NFSET_NEW = """        self._req_status: dict[ReqId, RequestOffloadState] = {}
        # LOCAL [glm53-offload-streaming]: load jobs that must not release
        # their request yet; drained into the worker metadata each step.
        self._stream_nonfinal_jobs: set[int] = set()
"""
NFSET_SENTINEL = "_stream_nonfinal_jobs: set[int] = set()"

METAPASS_OLD = """        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=partial_store_jobs | normal_store_jobs,
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )
"""
METAPASS_NEW = """        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=partial_store_jobs | normal_store_jobs,
            jobs_to_flush=self._current_batch_jobs_to_flush,
            # LOCAL [glm53-offload-streaming]: hand the worker the loads that
            # must not release their request, then reset for the next step.
            nonfinal_load_jobs=(set(self._stream_nonfinal_jobs)
                                if self._stream_nonfinal_jobs else None),
        )
        self._stream_nonfinal_jobs.clear()
"""
METAPASS_SENTINEL = "hand the worker the loads that"

SCHED_EDITS = [
    ("sched:nonfinal-set", NFSET_OLD, NFSET_NEW, NFSET_SENTINEL),
    ("sched:span-init", SPANINIT_OLD, SPANINIT_NEW, SPANINIT_SENTINEL),
    ("sched:span-start", SPANSTART_OLD, SPANSTART_NEW, SPANSTART_SENTINEL),
    ("sched:span-skip", SPANSKIP_OLD, SPANSKIP_NEW, SPANSKIP_SENTINEL),
    ("sched:span-end", SPANEND_OLD, SPANEND_NEW, SPANEND_SENTINEL),
    ("sched:helper", HELPER_OLD, HELPER_NEW, HELPER_SENTINEL),
    ("sched:issue-chained", ISSUE_OLD, ISSUE_NEW, ISSUE_SENTINEL),
    ("sched:meta-pass", METAPASS_OLD, METAPASS_NEW, METAPASS_SENTINEL),
]

# ===========================================================================
# W30 step 2b: promote PER BATCH, so the CPU tier stops bounding the restore.
#
# Steps 1 and 2a made a restore span several load jobs, but the tier's
# high-water mark was unchanged: TieringOffloadingManager.lookup() promotes
# every hit block as it is looked up, during the prefix scan, long before
# update_state_after_alloc runs. When the tier is full, _initiate_promotion
# fails and lookup returns MISS, which truncates the scan -- that is exactly
# where `tier size == maximum restorable prefix` comes from (measured: 9 blocks
# -> cached=0, 39 -> 25k tokens, 79 -> 50k).
#
# Here lookup reports a HIT for a block it could not promote, and the connector
# promotes each batch just before issuing its load.
#
# WHY THIS CANNOT DEADLOCK: prepare_load pins a block with ref_cnt += 1 and
# complete_load drops it back to 0, returning it to the evictable set
# (cpu/manager.py:140-166). So batch k's blocks become reusable the moment its
# load completes, and batch k+1's promotion can evict them. Progress is
# guaranteed as long as a SINGLE batch fits, which is why batches are per group.
#
# THE RISK IS LIVENESS, NOT CORRUPTION. Once lookup reports the larger hit, vLLM
# has committed to it and the request waits on finished_recving, which step 1
# withholds until the final batch lands. Two things could stall a batch:
#   * its promotion never gets capacity -- impossible per the argument above;
#   * its blocks get evicted from the DISK tier by another request's stores
#     while the plan is still queued, since a queued batch holds no reference.
# The second is real, so the driver touch()es every remaining key each step,
# making them most-recently-used and so the last things LRU would evict. If a
# batch does stall anyway the request hangs -- visibly, and only that request;
# it is logged as an ERROR and self-heals when the client disconnects and vLLM
# aborts the request. That is a deliberately better failure mode than either
# crashing the server or loading blocks that were never restored.
# ===========================================================================

MANAGER = VLLM / "v1/kv_offload/tiering/manager.py"

# --- manager: report the hit, let the connector do the promotion -------------

DEFER_OLD = """                promoted = self._initiate_promotion(i, key, req_context)
                return LookupResult.MISS if not promoted else LookupResult.HIT_PENDING
"""
DEFER_NEW = """                if self._stream_defer_enabled():
                    # LOCAL [glm53-offload-streaming]: report the hit and do NOT
                    # promote. Promotion during the prefix scan is what ties
                    # restore size to tier size, in BOTH branches of the
                    # original line below:
                    #   * promotion fails (tier full) -> MISS -> the scan stops
                    #     there, so tier size caps the restore;
                    #   * promotion succeeds -> HIT_PENDING -> the scan defers
                    #     and the scheduler re-queries next step. With a tier
                    #     smaller than the restore that never converges: each
                    #     step promotes a few more blocks, nothing has pinned
                    #     the earlier ones (prepare_load has not run), LRU
                    #     evicts them to make room for the next, and the request
                    #     sits in waiting_by_reason{reason=deferred} forever.
                    #     Measured, with a 9-block tier and a 14.3k prefix.
                    # A plain HIT lets the scan complete with no deferral; the
                    # connector then promotes each batch immediately before
                    # issuing its load, so only one batch is ever resident.
                    # stream_residency() is what makes that safe.
                    return LookupResult.HIT
                promoted = self._initiate_promotion(i, key, req_context)
                return LookupResult.MISS if not promoted else LookupResult.HIT_PENDING
"""
DEFER_SENTINEL = "report the hit and do NOT"

DEFERAPI_OLD = """    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
"""
DEFERAPI_NEW = '''    # LOCAL [glm53-offload-streaming]: cached env gate. lookup() is a hot path,
    # so this is resolved once per class rather than per block.
    _STREAM_DEFER = None

    def _stream_defer_enabled(self) -> bool:
        """LOCAL [glm53-offload-streaming]: GLM53_OFFLOAD_STREAM_RESTORE gate."""
        cls = type(self)
        if cls._STREAM_DEFER is None:
            import os as _os

            cls._STREAM_DEFER = _os.environ.get(
                "GLM53_OFFLOAD_STREAM_RESTORE") == "1"
        return cls._STREAM_DEFER

    def stream_residency(self, keys, req_context) -> bool:
        """LOCAL [glm53-offload-streaming]: are `keys` loadable right now?

        Residency is read from the PRIMARY tier. It cannot be read from
        self.lookup(), which under this patch deliberately reports HIT for blocks
        that are only on disk -- and cannot be used to DRIVE promotion either,
        for the same reason: it now returns before _initiate_promotion. So this
        promotes against the secondary tiers directly.

        _initiate_promotion allocates the primary slot immediately with
        ref_cnt -1, so a repeat call on the next step sees HIT_PENDING from
        primary_tier.lookup() rather than allocating twice, and
        on_schedule_end() -> _flush_pending_promotions() submits the batched job
        in this same step (the connector drives this before on_schedule_end).

        A promotion that cannot be allocated is simply retried next step, once
        the in-flight batch's complete_load has returned its blocks to the
        evictable set. That is the forward-progress argument, and it holds only
        while a single batch fits in the primary tier.

        Returns True once every key is resident and pinnable by prepare_load().
        """
        self._maybe_process_finished_jobs()
        ready = True
        for key in keys:
            primary = self.primary_tier.lookup(key, req_context)
            if primary is LookupResult.HIT:
                continue
            ready = False
            if primary is LookupResult.HIT_PENDING:
                continue                # promotion already in flight
            for i, tier in enumerate(self.secondary_tiers):
                if not req_context.load_tier_filter.allows(
                    tier.medium, tier.locality
                ):
                    continue
                if tier.lookup(key, req_context) is LookupResult.HIT:
                    self._initiate_promotion(i, key, req_context)
                    break
        return ready

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
'''
DEFERAPI_SENTINEL = "def stream_residency("

MANAGER_EDITS = [
    ("mgr:api", DEFERAPI_OLD, DEFERAPI_NEW, DEFERAPI_SENTINEL),
    ("mgr:lookup-defer", DEFER_OLD, DEFER_NEW, DEFER_SENTINEL),
]

# --- connector: hold the plan, issue one batch at a time --------------------

SEQ_OLD = """        if not batches:
            return
        for idx, (keys, blocks_, gs, bi) in enumerate(batches):
"""
SEQ_NEW = """        if not batches:
            return
        if self._stream_sequential():
            # LOCAL [glm53-offload-streaming] step 2b: hold the plan and issue
            # one batch at a time, promoting each just before its load, so only
            # one batch need be resident and the tier stops bounding the restore.
            self._stream_plans[request.request_id] = [
                self._stream_split_batches(batches), 0]
            self._stream_advance(request.request_id)
            return
        for idx, (keys, blocks_, gs, bi) in enumerate(batches):
"""
SEQ_SENTINEL = "step 2b: hold the plan and issue"

ADVANCE_OLD = """    def _issue_chained_load(
"""
ADVANCE_NEW = '''    # LOCAL [glm53-offload-streaming]: steps a stalled batch may wait before it
    # is reported as an error. ~600 steps is tens of seconds of engine time.
    _STREAM_STALL_STEPS = 600
    _STREAM_SEQ = None
    _STREAM_BATCH = None

    def _stream_batch_blocks(self) -> int:
        """LOCAL [glm53-offload-streaming]: max offload blocks per batch.

        Forward progress requires that ONE batch fit in the primary tier, so the
        default is half of it -- leaving room for other requests' promotions and
        stores. A group's batch is otherwise unbounded: at 3584 tokens per
        offload block a 1M-token restore is ~280 blocks against a 79-block tier.
        """
        cls = type(self)
        if cls._STREAM_BATCH is None:
            import os as _os

            n = int(_os.environ.get("GLM53_OFFLOAD_STREAM_BATCH_BLOCKS") or 0)
            if n <= 0:
                primary = getattr(self.manager, "primary_tier", None)
                n = max(1, (getattr(primary, "_num_blocks", 0) or 2) // 2)
            cls._STREAM_BATCH = n
            logger.info(
                "KV offload [glm53-offload-streaming]: streaming restores in "
                "batches of at most %d offload block(s) per KV group", n
            )
        return cls._STREAM_BATCH

    def _stream_split_batches(self, batches):
        """LOCAL [glm53-offload-streaming]: split batches to fit the tier.

        Chunk j of a group's slice covers GPU blocks
        [j*bpc - off, (j+1)*bpc - off) relative to the slice start, where
        bpc = self.config.blocks_per_chunk and off = block_indices[gi] % bpc is
        how far into its offload chunk the slice begins (the scheduler derives
        the slice from `num_locally_computed_gpu_blocks // blocks_per_chunk`).

        Splitting on a chunk boundary leaves every sub-batch after the first
        chunk-aligned, so its block_indices -- the field the worker uses to skip
        part of the first offload block -- is exact. Getting `off` wrong would
        shift blocks by less than one chunk: KV that loads without error and is
        silently wrong. The arithmetic is covered by tests/test_stream_split.py.
        """
        limit = self._stream_batch_blocks()
        bpc = self.config.blocks_per_chunk
        out = []
        for keys, blocks_, gs, bi in batches:
            gi = next(i for i, v in enumerate(gs) if v)
            if len(keys) <= limit:
                out.append((keys, blocks_, gs, bi))
                continue
            off = bi[gi] % bpc
            for a in range(0, len(keys), limit):
                b = min(a + limit, len(keys))
                d_lo = max(0, a * bpc - off)
                d_hi = min(len(blocks_), b * bpc - off)
                if d_hi <= d_lo:
                    continue
                sub_gs = [0] * len(gs)
                sub_bi = [0] * len(bi)
                sub_gs[gi] = d_hi - d_lo
                sub_bi[gi] = bi[gi] + d_lo
                out.append((keys[a:b], blocks_[d_lo:d_hi], sub_gs, sub_bi))
        return out


    def _stream_sequential(self) -> bool:
        """LOCAL [glm53-offload-streaming]: issue restore batches one at a time."""
        cls = type(self)
        if cls._STREAM_SEQ is None:
            import os as _os

            cls._STREAM_SEQ = _os.environ.get(
                "GLM53_OFFLOAD_STREAM_SEQUENTIAL", "1") == "1"
        return cls._STREAM_SEQ

    def _stream_advance(self, req_id) -> None:
        """LOCAL [glm53-offload-streaming]: issue the next batch of a restore.

        Driven every step from build_connector_meta, ahead of on_schedule_end so
        a promotion started here is submitted in the same step. Cannot deadlock:
        complete_load returns the previous batch's blocks to the evictable set,
        so this batch's promotion can reuse them.
        """
        plan = self._stream_plans.get(req_id)
        if not plan:
            return
        batches, waits = plan
        req_status = self._req_status.get(req_id)
        if req_status is None:
            # Request went away (finished, preempted, or client aborted).
            self._stream_plans.pop(req_id, None)
            return
        if req_status.transfer_jobs:
            return                      # a batch is still in flight

        keys, blocks_, gs, bi = batches[0]
        # Keep every queued key most-recently-used in each tier. A queued batch
        # holds no reference, so this is what stops another request's stores
        # evicting it from the disk tier mid-restore.
        self.manager.touch(
            [k for b in batches for k in b[0]], req_status.req_context
        )
        residency = getattr(self.manager, "stream_residency", None)
        if residency is not None and not residency(keys, req_status.req_context):
            plan[1] = waits + 1
            if plan[1] % self._STREAM_STALL_STEPS == 0:
                logger.error(
                    "Request %s: offload restore batch stalled for %d steps "
                    "waiting on promotion of %d block(s); the request will hang "
                    "until it is aborted. Blocks were likely evicted from a "
                    "secondary tier mid-restore.",
                    req_id, plan[1], len(keys),
                )
            return                      # not resident yet; retry next step

        src_spec = self.manager.prepare_load(keys, req_status.req_context)
        dst_spec = GPULoadStoreSpec(blocks_, group_sizes=gs, block_indices=bi)
        job_id = self._generate_job_id()
        self._current_batch_load_jobs[job_id] = TransferJob(
            req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
        )
        req_status.transfer_jobs.add(job_id)
        self._jobs[job_id] = TransferJobStatus(
            req_id=req_id,
            pending_count=self.config.num_workers,
            keys=set(keys),
            is_store=False,
        )
        batches.pop(0)
        plan[1] = 0
        if batches:
            self._stream_nonfinal_jobs.add(job_id)
        else:
            self._stream_plans.pop(req_id, None)

    def _issue_chained_load(
'''
ADVANCE_SENTINEL = "def _stream_advance("

PLANS_OLD = """        self._stream_nonfinal_jobs: set[int] = set()
"""
PLANS_NEW = """        self._stream_nonfinal_jobs: set[int] = set()
        # LOCAL [glm53-offload-streaming]: req_id -> [remaining batches, waits].
        self._stream_plans: dict = {}
"""
PLANS_SENTINEL = "_stream_plans: dict = {}"

DRIVE_OLD = """        self._update_req_states(scheduler_output)
        schedule_end_context = ScheduleEndContext(
"""
DRIVE_NEW = """        self._update_req_states(scheduler_output)
        # LOCAL [glm53-offload-streaming]: drive in-progress streamed restores.
        # Ahead of on_schedule_end() so promotions initiated here are flushed by
        # _flush_pending_promotions() in this same step, and every step rather
        # than only on job completion, since a batch may be waiting on a
        # promotion that lands between jobs.
        for _stream_req_id in list(self._stream_plans):
            self._stream_advance(_stream_req_id)

        schedule_end_context = ScheduleEndContext(
"""
DRIVE_SENTINEL = "drive in-progress streamed restores"

SCHED2B_EDITS = [
    ("sched:plans", PLANS_OLD, PLANS_NEW, PLANS_SENTINEL),
    ("sched:advance", ADVANCE_OLD, ADVANCE_NEW, ADVANCE_SENTINEL),
    ("sched:sequential", SEQ_OLD, SEQ_NEW, SEQ_SENTINEL),
    ("sched:drive", DRIVE_OLD, DRIVE_NEW, DRIVE_SENTINEL),
]

COMMON_EDITS = [("common:metadata-field", META_OLD, META_NEW, META_SENTINEL)]
WORKER_EDITS = [
    ("worker:nonfinal-set", INIT_OLD, INIT_NEW, INIT_SENTINEL),
    ("worker:register", REG_OLD, REG_NEW, REG_SENTINEL),
    ("worker:withhold", FIN_OLD, FIN_NEW, FIN_SENTINEL),
    ("worker:clear", CLEAR_OLD, CLEAR_NEW, CLEAR_SENTINEL),
]


def apply_edits(path: Path, edits) -> str:
    if not path.is_file():
        return f"MISSING {path} - not patched"
    src = path.read_text()
    applied, skipped = [], []
    for label, old, new, sentinel in edits:
        if sentinel in src:
            skipped.append(label)
            continue
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"{MARK} FAIL {label}: anchor matched {n} times (expected 1). "
                f"Upstream drifted; refusing to patch."
            )
        src = src.replace(old, new)
        applied.append(label)
    if applied:
        compile(src, str(path), "exec")
        path.write_text(src)
    parts = []
    if applied:
        parts.append("applied " + ",".join(applied))
    if skipped:
        parts.append("already " + ",".join(skipped))
    return "; ".join(parts) or "nothing to do"


def main() -> int:
    if os.environ.get("GLM53_OFFLOAD_STREAM_RESTORE", "0") != "1":
        print(f"{MARK} disabled (GLM53_OFFLOAD_STREAM_RESTORE != 1); vLLM pristine")
        return 0
    print(f"{MARK} {apply_edits(COMMON, COMMON_EDITS)}")
    print(f"{MARK} {apply_edits(WORKER, WORKER_EDITS)}")
    print(f"{MARK} {apply_edits(SCHED, SCHED_EDITS + SCHED2B_EDITS)}")
    print(f"{MARK} {apply_edits(MANAGER, MANAGER_EDITS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
