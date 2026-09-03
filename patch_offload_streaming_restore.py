#!/usr/bin/env python3
"""W30: chained restore -- step 1 (primitive) + step 2a (per-group batching).

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

WHAT IS DELIBERATELY NOT HERE
-----------------------------
The batching itself (splitting `keys_to_load` by chunk range, promoting batch
k+1 from the FS tier, polling until resident, then issuing its GPU load). That
needs a per-request state machine driven from `build_connector_meta`, because
`TieringOffloadingManager.prepare_load` requires keys "already confirmed HIT by
lookup() earlier this step" and increments ref_cnt to pin them -- so a later
batch must be promoted and awaited first. Landing that on top of an unvalidated
primitive is how the earlier silent-corruption bugs happened.

Idempotent, sentinel-guarded, fails closed on drift. Gated on
GLM53_OFFLOAD_STREAM_RESTORE=1; vLLM is untouched otherwise. The field defaults
to None so metadata built by an unpatched scheduler behaves exactly as before.
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
    print(f"{MARK} {apply_edits(SCHED, SCHED_EDITS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
