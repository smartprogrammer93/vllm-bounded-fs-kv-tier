#!/usr/bin/env python3
"""Back the CPU staging region with a sparse file on local NVMe.

Ported from MiaAI PR#58 (overlay/patch_kv_offload_groups.py, its `sor_edits`
and the cudaHostRegister skip), re-expressed in this kit's overlay style. Their
group-exclusion half is NOT taken: we already have
overlay/patch_offload_group_filter.py, which does the same job and additionally
carries the region-copies fix (rows sized for world_size while every worker uses
slot 0) that halved the row from 51.81 to 25.91 MiB. PR#58 does not have that.

WHY THIS REPLACES THE STREAMED RESTORE. vLLM admits a restore only when every
group's chunks are resident in the primary tier TOGETHER -- PR#58 states the
sizing rule directly: cpu_bytes_to_use >= row_bytes * (restore_tokens/3584 + 8),
and "undersized, stores still succeed and metrics still climb while no restore
ever serves", which is exactly the signature measured here (store count rising,
load count frozen). Our step-2b streamed restore tried to dodge that invariant
by reporting HIT and making chunks resident batch by batch; it stalls at N>=6.
Backing the region with NVMe instead makes the tier large enough to satisfy the
invariant, so the restore window stops being bounded by RAM without fighting the
scheduler.

SIZING. Do not take a row size on trust -- read it off your own boot. vLLM sets
num_blocks = cpu_bytes_to_use // aligned_kv_bytes_per_chunk and then creates a
region of exactly num_blocks * aligned_kv_bytes_per_chunk, so the "Created mmap
file ... (N GB)" line divided by num_blocks gives you the real row. On this
stack it is 54,329,344 B (51.81 MiB), confirmed twice: 32 GiB -> 632 rows and
260 GB -> 4785 rows, both exact.

The rule that actually matters is not a byte count, it is a comparison: THE TIER
MUST BE LARGER THAN THE GPU KV POOL. Both are LRU over the same stream of
blocks, so a tier smaller than the pool holds a strict subset of what the pool
already has -- the GPU answers first and the tier can never be the thing that
serves. Sizing it below the pool is not a smaller cache, it is no cache. The
file is sparse, so over-sizing costs nothing until written.

Gated on GLM53_OFFLOAD_MMAP_DIR being set; unset leaves vLLM's /dev/shm
behaviour untouched. Idempotent, sentinel-guarded, fails closed on drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SOR = VLLM / "v1/kv_offload/cpu/shared_offload_region.py"
GWK = VLLM / "v1/kv_offload/cpu/gpu_worker.py"
MARK = "[glm53-offload-mmap]"

# 1. put the region file on NVMe instead of /dev/shm
PATH_OLD = '''        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"'''
PATH_NEW = '''        # LOCAL [glm53-offload-mmap]: a sparse file on this node's own NVMe, so
        # the staging tier's capacity is disk rather than RAM. Each rank keeps
        # its own file, which is also why no cross-node mirroring is needed.
        _mmap_dir = os.environ.get("GLM53_OFFLOAD_MMAP_DIR")
        self._glm53_disk_backed = bool(_mmap_dir)
        if _mmap_dir:
            os.makedirs(_mmap_dir, exist_ok=True)
        else:
            _mmap_dir = "/dev/shm"
        self.mmap_path = f"{_mmap_dir}/vllm_offload_{engine_id}.mmap"'''
PATH_SENTINEL = "[glm53-offload-mmap]: a sparse file"

# 2. the /dev/shm free-space check is meaningless for a sparse file on NVMe
SHM_OLD = '''                check_shm_free_space(self.total_size_bytes)
                os.ftruncate(self.fd, self.total_size_bytes)'''
SHM_NEW = '''                if not self._glm53_disk_backed:  # LOCAL [glm53-offload-mmap]
                    check_shm_free_space(self.total_size_bytes)
                os.ftruncate(self.fd, self.total_size_bytes)'''
SHM_SENTINEL = "if not self._glm53_disk_backed:"

# 3. never pre-fault: writing every page would materialise the whole file
POP_OLD = '''        populate_write_fn = _get_populate_write_fn(self.mmap_obj)

        if rank is not None:'''
POP_NEW = '''        # LOCAL [glm53-offload-mmap]: pre-faulting a sparse NVMe file would
        # materialise the entire region on disk at boot.
        if self._glm53_disk_backed:
            populate_write_fn = lambda *_a, **_k: None
            _glm53_skip_populate = True
        else:
            populate_write_fn = _get_populate_write_fn(self.mmap_obj)
            _glm53_skip_populate = False

        if _glm53_skip_populate:
            pass
        elif rank is not None:'''
POP_SENTINEL = "_glm53_skip_populate"

SOR_EDITS = [
    ("sor:mmap-path", PATH_OLD, PATH_NEW, PATH_SENTINEL),
    ("sor:skip-shm-check", SHM_OLD, SHM_NEW, SHM_SENTINEL),
    ("sor:skip-populate", POP_OLD, POP_NEW, POP_SENTINEL),
]

# 4. cudaHostRegister on a disk-backed mapping is wrong and slow
GWK_OLD = '''    if not current_platform.is_cuda_alike():'''
GWK_NEW = '''    import os as _os  # LOCAL [glm53-offload-mmap]
    if _os.environ.get("GLM53_OFFLOAD_MMAP_DIR"):
        logger.info("%s disk-backed staging region: skipping cudaHostRegister",
                    "[glm53-offload-mmap]")
        return
    if not current_platform.is_cuda_alike():'''
GWK_SENTINEL = "skipping cudaHostRegister"

GWK_EDITS = [("gwk:skip-hostregister", GWK_OLD, GWK_NEW, GWK_SENTINEL)]


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
                f"Upstream drifted; refusing to patch.")
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
    if not os.environ.get("GLM53_OFFLOAD_MMAP_DIR"):
        print(f"{MARK} disabled (GLM53_OFFLOAD_MMAP_DIR unset); vLLM pristine")
        return 0
    print(f"{MARK} {apply_edits(SOR, SOR_EDITS)}")
    print(f"{MARK} {apply_edits(GWK, GWK_EDITS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
