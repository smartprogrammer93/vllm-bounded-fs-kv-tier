#!/usr/bin/env python3
"""Size-bounded filesystem secondary tier for vLLM KV offloading.

WHY THIS EXISTS
---------------
vLLM's ``FileSystemTierManager`` (vllm/v1/kv_offload/tiering/fs/manager.py) has
**no capacity limit and no eviction**: it writes one ``.bin`` per block and never
unlinks anything. Audited on vllm 0.1.dev20051+g487ecf187 -- it implements
lookup/submit_store/submit_load/get_finished_jobs/take_events/drain_jobs/
on_new_request/on_request_finished/on_schedule_end/shutdown, does NOT override
``touch()`` (so there is no recency tracking at all), and neither the ``obj``
tier nor the ``SecondaryTierManager`` ABC define capacity/evict/quota.

If ``root_dir`` sits on a filesystem you cannot afford to fill (a root volume,
for example), unbounded growth is an outage risk worse than the cache miss it
avoids.

This subclass adds a hard byte cap with LRU eviction. It is loaded as an
OUT-OF-TREE tier via the officially supported extension point
(``secondary_tiers: [{"type": ..., "module_path": ...}]``), so vLLM itself is
NOT patched and this survives engine upgrades.

WHY EVICTION NEEDS NO LOCKING
-----------------------------
Deleting a file out from under an in-flight load is SAFE by construction: the
parent's load task catches ``OSError`` (FileNotFoundError included), records how
many blocks succeeded, and ``get_finished_jobs`` calls
``self._lookup_manager.mark_miss(failed)``. The affected blocks simply become a
cache miss and are recomputed. No corruption, no request failure. So eviction
does not have to coordinate with readers.

ACCOUNTING
----------
``batch_store_block`` writes exactly one file of ``self._block_size`` BYTES per
key, so usage is ``len(tracked_files) * block_size`` -- no stat() per file.

CONFIG
------
    "secondary_tiers": [{
        "type": "BoundedFileSystemTierManager",
        "module_path": "vllm_bounded_fs_tier",
        "root_dir": "/kvoffload",
        "max_bytes": 214748364800,        # 200 GiB; also accepts "200GiB"/"200GB"
        "evict_to_ratio": 0.9,            # evict down to 90% of the cap
        "n_read_threads": 8,
        "n_write_threads": 8
    }]
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

logger = init_logger(__name__)

_SUFFIXES = {
    "K": 1000, "KB": 1000, "KIB": 1024,
    "M": 1000**2, "MB": 1000**2, "MIB": 1024**2,
    "G": 1000**3, "GB": 1000**3, "GIB": 1024**3,
    "T": 1000**4, "TB": 1000**4, "TIB": 1024**4,
}


def parse_bytes(value: Any) -> int:
    """Accept an int, a bare numeric string, or '200GiB'/'200GB'/'500MB'."""
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace("_", "")
    m = re.fullmatch(r"(?i)\s*([0-9]*\.?[0-9]+)\s*([a-z]*)\s*", text)
    if not m:
        raise ValueError(f"max_bytes: cannot parse {value!r}")
    num, suffix = float(m.group(1)), m.group(2).upper()
    if not suffix:
        return int(num)
    if suffix not in _SUFFIXES:
        raise ValueError(f"max_bytes: unknown unit {suffix!r} in {value!r}")
    return int(num * _SUFFIXES[suffix])


class BoundedFileSystemTierManager(FileSystemTierManager):
    """FileSystemTierManager + hard byte cap + LRU eviction."""

    def __init__(
        self,
        *args: Any,
        max_bytes: Any,
        evict_to_ratio: float = 0.9,
        **kwargs: Any,
    ) -> None:
        # Capture root_dir before delegating: the factory may pass it either
        # positionally (4th, after offloading_spec/primary_kv_view/tier_type)
        # or by keyword, and we must not depend on get_run_config()'s shape.
        root_dir = kwargs.get("root_dir")
        if root_dir is None and len(args) >= 4:
            root_dir = args[3]
        self._root_dir = root_dir

        super().__init__(*args, **kwargs)

        self._max_bytes = parse_bytes(max_bytes)
        if self._max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive (got {max_bytes!r})")
        if not 0.0 < evict_to_ratio <= 1.0:
            raise ValueError(f"evict_to_ratio must be in (0, 1] (got {evict_to_ratio})")
        self._evict_to = int(self._max_bytes * evict_to_ratio)

        # path -> None, in LRU order (oldest first). Source of truth for usage.
        self._tracked: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self._evicted_total = 0

        self._adopt_existing_files()
        logger.info(
            "[bounded-fs] cap=%.1f GiB evict_to=%.1f GiB block=%d B "
            "adopted=%d files (%.1f GiB) root=%s",
            self._max_bytes / 1024**3,
            self._evict_to / 1024**3,
            self._block_size,
            len(self._tracked),
            self._used_bytes() / 1024**3,
            self._root_dir,
        )
        # A restart can inherit an over-cap directory; trim before serving.
        self._evict_if_needed()

    # ---------------------------------------------------------------- helpers
    def _used_bytes(self) -> int:
        return len(self._tracked) * self._block_size

    def _adopt_existing_files(self) -> None:
        """Take ownership of blocks left by a previous run, oldest-first.

        Without this a restart would see usage 0 and grow past the cap on top
        of whatever is already on disk.
        """
        root = self._root_dir
        if not root or not os.path.isdir(root):
            return
        found: list[tuple[float, str]] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(".bin"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    found.append((os.stat(path).st_mtime, path))
                except OSError:
                    continue
        found.sort()
        for _mtime, path in found:
            self._tracked[path] = None

    def _evict_if_needed(self) -> None:
        used = self._used_bytes()
        if used <= self._max_bytes:
            return
        freed = 0
        removed = 0
        with self._lock:
            while self._used_bytes() > self._evict_to and self._tracked:
                path, _ = self._tracked.popitem(last=False)  # oldest
                try:
                    os.unlink(path)
                    freed += self._block_size
                except FileNotFoundError:
                    pass  # already gone; dropping it from the map is the fix
                except OSError as exc:
                    logger.warning("[bounded-fs] unlink %s: %s", path, exc)
                removed += 1
        self._evicted_total += removed
        logger.info(
            "[bounded-fs] evicted %d blocks (%.2f GiB); usage %.1f -> %.1f GiB "
            "of %.1f GiB cap (evicted_total=%d)",
            removed,
            freed / 1024**3,
            used / 1024**3,
            self._used_bytes() / 1024**3,
            self._max_bytes / 1024**3,
            self._evicted_total,
        )

    # ------------------------------------------------------------- overrides
    def submit_store(self, job_metadata: TransferJob) -> None:
        """Track the files this job will create, then delegate.

        Tracking at submit (not completion) deliberately over-counts a failed
        store; the correction is self-healing, because eviction tolerates
        FileNotFoundError and drops the entry.
        """
        super().submit_store(job_metadata)
        with self._lock:
            for key in job_metadata.keys:
                path = self.file_mapper.get_file_name(key)
                self._tracked.pop(path, None)
                self._tracked[path] = None  # newest

    def touch(self, keys: Collection[Any], req_context: Any) -> None:
        """Mark blocks recently used. The base class is a no-op, so without
        this the LRU would degrade to FIFO."""
        with self._lock:
            for key in keys:
                path = self.file_mapper.get_file_name(key)
                if path in self._tracked:
                    self._tracked.move_to_end(path)
        super().touch(keys, req_context)

    def get_finished_jobs(self) -> Iterable[JobResult]:
        results = list(super().get_finished_jobs())
        self._evict_if_needed()
        return results
