#!/usr/bin/env python3
"""W30 step 1: let a restore span MULTIPLE load jobs (the chaining primitive).

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
