#!/usr/bin/env python3
"""Out-of-tree KV-cache offload tier for vLLM: bounded disk + per-rank cascade.

Two managers, both loaded through the supported extension point
(``secondary_tiers: [{"type": ..., "module_path": ...}]``), so vLLM itself is
never patched and this survives engine upgrades:

``BoundedFileSystemTierManager``
    vLLM's ``FileSystemTierManager`` has **no capacity limit and no eviction**:
    it writes one ``.bin`` per block and never unlinks anything. If ``root_dir``
    sits on a filesystem you cannot afford to fill, unbounded growth is an
    outage worse than the cache miss it avoids. This adds a hard byte cap with
    LRU eviction (``max_bytes``, ``evict_to_ratio``), recency tracking via a
    ``touch()`` override the parent leaves as a no-op, and adoption of blocks
    left by a previous run so a restart cannot grow past the cap on top of
    whatever is already on disk.

``PeerMirroredFileSystemTierManager``
    Makes the disk tier **correct when the tensor-parallel group spans hosts**,
    and much cheaper. vLLM builds every secondary tier scheduler-side over a
    memoryview of the *local* node's CPU primary tier, and that tier is a
    per-node ``/dev/shm`` region -- so on a multi-host TP group only rank 0's
    shard ever reaches disk, while the scheduler still issues the resulting load
    to every worker. The remote rank then copies stale bytes into its GPU blocks
    and the model emits garbage with no error anywhere. This manager drives a
    ``peer_kv_agent.py`` in each remote rank's container and withholds every
    ``JobResult`` until all peers acknowledge, which is what orders disk -> CPU
    before CPU -> GPU on all ranks. A peer failure fails the job, so
    ``mark_miss`` turns it into a recompute: a wasted prefill, never a silently
    half-restored prefix. It also stores only the worker's own slot of each
    region row, halving disk.

WHY EVICTION NEEDS NO LOCKING
-----------------------------
Deleting a file out from under an in-flight load is SAFE by construction: the
parent's load task catches ``OSError`` (``FileNotFoundError`` included), records
how many blocks succeeded, and ``get_finished_jobs`` calls
``self._lookup_manager.mark_miss(failed)``. The affected blocks simply become a
cache miss and are recomputed. No corruption, no request failure.

ENVIRONMENT
-----------
``GLM53_OFFLOAD_PEERS``            ``host:port[,host:port]`` in rank order from 1.
                                   Unset => plain single-node behaviour.
``GLM53_OFFLOAD_PEER_TIMEOUT``     seconds, default 120.
``GLM53_OFFLOAD_RANK_LOCAL_ROWS``  ``1`` (default) stores only this worker's slot.
``GLM53_OFFLOAD_LOCAL_SLOT``       which slot this worker owns; the LOCAL device
                                   index, so ``0`` with one GPU per node.
``GLM53_OFFLOAD_PEER_ROOT``        peers' cache root, default ``/kvoffload``.
``PYTHONHASHSEED``                 must be fixed and identical everywhere, or
                                   vLLM seeds its block-content hash chain with
                                   random bytes and identical tokens produce
                                   different filenames on every restart.

See README.md for the measured numbers and UPSTREAM_ISSUE.md for the five
upstream defects this works around.
"""

from __future__ import annotations

import json
import os
import re
import sys
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


_pclog = init_logger("vllm.peer_cascade")


def _plog(level, msg, *args):
    """Log under the vllm.* hierarchy AND to stderr.

    A logger named after this module sits outside vLLM's configured logger tree
    and every record is dropped, which hid a wire-protocol bug for a whole
    engine boot. Never let a cascade failure be invisible.
    """
    try:
        getattr(_pclog, level)(msg, *args)
    except Exception:
        pass
    try:
        print("[peer-cascade] " + (msg % args if args else msg),
              file=sys.stderr, flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-rank cascade for a tensor-parallel group that spans hosts.
# See peer_kv_agent.py for the full explanation of the vLLM limitation.
# ---------------------------------------------------------------------------


class _Peer:
    """One remote rank's cascade agent."""

    def __init__(self, addr: str, rank: int, timeout: float):
        host, _, port = addr.rpartition(":")
        self.addr = addr
        self.host = host
        self.port = int(port)
        self.rank = rank
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sock = None
        self._rf = None
        self._wf = None
        self._hello = None

    def set_hello(self, hello: dict) -> None:
        self._hello = hello

    def _connect_locked(self):
        import socket as _socket

        if self._sock is not None:
            return
        s = _socket.create_connection((self.host, self.port), timeout=self.timeout)
        s.settimeout(self.timeout)
        self._sock = s
        self._rf = s.makefile("rb")
        self._wf = s.makefile("wb")
        _plog("info", "connected to rank %d at %s", self.rank, self.addr)
        if self._hello is not None:
            resp = self._rpc_locked(dict(self._hello, op="hello", rank=self.rank))
            if not resp.get("ok"):
                raise RuntimeError("peer %s hello failed: %r" % (self.addr, resp))
            _plog("info", "rank %d attached %s", self.rank, resp.get("mmap"))

    def _rpc_locked(self, msg: dict) -> dict:
        self._wf.write((json.dumps(msg) + "\n").encode())
        self._wf.flush()
        line = self._rf.readline()
        if not line:
            raise RuntimeError("peer %s closed the connection" % self.addr)
        return json.loads(line)

    def rpc(self, msg: dict) -> dict:
        with self._lock:
            try:
                self._connect_locked()
                return self._rpc_locked(msg)
            except Exception as e:
                self.close_locked()
                return {"ok": False, "job_id": msg.get("job_id"),
                        "num_succeeded": 0, "err": "%s: %s" % (type(e).__name__, e)}

    def close_locked(self) -> None:
        for f in (self._rf, self._wf):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = self._rf = self._wf = None


class PeerMirroredFileSystemTierManager(BoundedFileSystemTierManager):
    """Bounded FS tier that also cascades every remote rank's own shard.

    vLLM constructs secondary tiers scheduler-side over the LOCAL node's CPU
    region only, so on a multi-host TP group the remote ranks' shards never
    reach disk and a promotion silently restores half a prefix. This subclass
    drives an agent on each remote rank so all shards move together, and holds
    each JobResult until every peer confirms.
    """

    def __init__(self, *args, **kwargs):
        peers_env = kwargs.pop("peers", None) or os.environ.get(
            "GLM53_OFFLOAD_PEERS", "")
        self._peer_timeout = float(kwargs.pop("peer_timeout", 0) or
                                   os.environ.get("GLM53_OFFLOAD_PEER_TIMEOUT", "120"))

        # MUST be initialised BEFORE super().__init__(): the bounded parent's
        # constructor calls _evict_if_needed() to trim an over-cap directory
        # inherited from a previous run, and our override of that method reads
        # these attributes. An empty peer list makes it fall through to the
        # parent, which is the correct behaviour during construction.
        self._peers: list = []
        self._pool_exec = None
        self._pending: dict = {}
        self._held: list = []
        self._peer_lock = threading.Lock()
        self._load_keys_mirror: dict = {}
        # 0 => "not decided yet"; _used_bytes() falls back to the row size.
        # The parent's constructor calls _used_bytes() (via its startup log and
        # _evict_if_needed), so this must exist before super().__init__().
        self._on_disk_block_bytes = 0

        super().__init__(*args, **kwargs)

        # W27b geometry: a region row spans every rank
        # (_worker_offset = rank * cpu_page_size), but a rank's worker only ever
        # touches its own sub-slot.
        _spec = getattr(self, "_offloading_spec", None)
        _par = getattr(getattr(_spec, "config", None), "parallel", None)
        self._page_size = int(getattr(_spec, "cpu_page_size_per_worker", 0) or 0)
        self._rank = int(getattr(_par, "rank", 0) or 0)
        self._world = int(getattr(_par, "world_size", 1) or 1)
        if (os.environ.get("GLM53_OFFLOAD_RANK_LOCAL_ROWS", "1") == "1"
                and self._world > 1
                and 0 < self._page_size < self._block_size):
            self._install_rank_local_io()

        addrs = [a.strip() for a in peers_env.split(",") if a.strip()]
        for i, a in enumerate(addrs):
            self._peers.append(_Peer(a, rank=i + 1, timeout=self._peer_timeout))

        if not self._peers:
            _plog("info", "no peers configured; single-node behaviour")
            return

        mapper = self.file_mapper
        hello = {
            "basename": os.path.basename(mapper.base_path),
            "block_size": int(self._block_size),
            "total_size": int(self._primary_kv_view.nbytes),
            "root_dir": os.environ.get("GLM53_OFFLOAD_PEER_ROOT", "/kvoffload"),
            "o_direct": bool(getattr(self, "_use_o_direct", False)),
            # The one configurable bound (max_bytes) governs every rank's copy.
            "max_bytes": int(self._max_bytes),
            # W27b: 0 => peer writes whole rows (legacy), >0 => its own slice.
            "page_size": int(self._page_size) if self._on_disk_block_bytes
            != self._block_size else 0,
            "local_slot": int(os.environ.get("GLM53_OFFLOAD_LOCAL_SLOT", "0")),
        }
        for p in self._peers:
            p.set_hello(hello)
        from concurrent.futures import ThreadPoolExecutor

        self._pool_exec = ThreadPoolExecutor(
            max_workers=max(2, 2 * len(self._peers)),
            thread_name_prefix="peer-cascade")
        _plog("info",
              "%d peer rank(s) %s; basename=%s block_size=%d region=%d bytes "
              "cap=%.2f GiB local_slot_page=%s",
              len(self._peers), [p.addr for p in self._peers],
              hello["basename"], hello["block_size"], hello["total_size"],
              hello["max_bytes"] / 1024 ** 3, hello["page_size"] or "off")

    # -- W27b: store only this rank's sub-slot of each row ----------------
    def _install_rank_local_io(self) -> None:
        """Halve offload storage by writing only this rank's slice of a row.

        Each rank's cascade otherwise writes the ENTIRE row, including the other
        rank's half, which is inert on this host -- and the peer writes the same
        row from its side, so every byte lands on disk twice. Narrowing the
        offsets and length the parent passes to the io helpers fixes that while
        reusing all of the parent's job bookkeeping, thread pool and partial-load
        failure handling untouched.

        Offsets stay O_DIRECT-aligned: cpu_page_size_per_worker is itself a
        multiple of the page size, so base and length are too.
        """
        import vllm.v1.kv_offload.tiering.fs.manager as _fsm

        if getattr(_fsm, "_glm53_rank_local_io", None) is not None:
            self._on_disk_block_bytes = _fsm._glm53_rank_local_io[1]
            return
        # MEASURED, do not "fix" this to rank * page: with one worker per node
        # each node has its OWN region, so the worker's slot index within that
        # region is its LOCAL rank -- 0 on every host. Dumping both regions
        # showed slot 0 populated and slot 1 all-zero on BOTH ranks, i.e. the
        # upper half of every row is dead padding here (cpu/spec.py sizes the row
        # by world_size even when the region is not shared across those ranks).
        # Using rank * page made the peer store and restore zeros -> silent
        # corruption (empty completions), which is how this was found.
        page = self._page_size
        base = int(os.environ.get("GLM53_OFFLOAD_LOCAL_SLOT", "0")) * page
        _orig_store = _fsm.batch_store_block
        _orig_load = _fsm.batch_load_block

        def _store(paths, view, offsets, block_size, use_o_direct=True):
            return _orig_store(
                paths, view, [o + base for o in offsets], page, use_o_direct)

        def _load(paths, view, offsets, block_size, use_o_direct=True):
            # Preserves the OSError.num_succeeded contract by forwarding as-is.
            return _orig_load(
                paths, view, [o + base for o in offsets], page, use_o_direct)

        _fsm.batch_store_block = _store
        _fsm.batch_load_block = _load
        _fsm._glm53_rank_local_io = (base, page)
        self._on_disk_block_bytes = page
        _plog("info",
              "W27b local-slot rows ON: rank %d/%d writes [%d,%d) of each row; "
              "on-disk block %.2f -> %.2f MiB (%.2fx less disk)",
              self._rank, self._world, base, base + page,
              self._block_size / 1024 ** 2, page / 1024 ** 2,
              self._block_size / float(page))

    def _used_bytes(self) -> int:
        """Account for what is actually on disk, so max_bytes stays truthful
        (and the cap holds ~2x more blocks once rank-local rows are on).

        Falls back to the full row size until the rank-local decision is made,
        because the parent's constructor calls this before that point.
        """
        return len(self._tracked) * (
            self._on_disk_block_bytes or self._block_size)

    # -- job mirroring ----------------------------------------------------
    def _mirror(self, op: str, job_metadata) -> None:
        if not self._peers:
            return
        job_id = job_metadata.job_id
        keys = [bytes(k).hex() for k in job_metadata.keys]
        block_ids = [int(b) for b in job_metadata.block_ids]
        with self._peer_lock:
            self._pending[job_id] = {"want": len(self._peers), "got": 0,
                                     "ok": True, "err": None}
        msg = {"op": op, "job_id": job_id, "keys": keys, "block_ids": block_ids}

        def run(peer=None):
            resp = peer.rpc(msg)
            with self._peer_lock:
                st = self._pending.get(job_id)
                if st is None:
                    return
                st["got"] += 1
                if not resp.get("ok"):
                    st["ok"] = False
                    st["err"] = resp.get("err")
                    _plog("warning", "rank %d %s job %s FAILED: %s",
                          peer.rank, op, job_id, resp.get("err"))

        for p in self._peers:
            self._pool_exec.submit(run, peer=p)

    def submit_store(self, job_metadata) -> None:
        self._mirror("store", job_metadata)
        super().submit_store(job_metadata)

    def submit_load(self, job_metadata) -> None:
        # Start the peers first so their disk reads overlap ours.
        self._mirror("load", job_metadata)
        with self._peer_lock:
            self._load_keys_mirror[job_metadata.job_id] = list(job_metadata.keys)
        super().submit_load(job_metadata)

    def get_finished_jobs(self):
        parent = list(super().get_finished_jobs())
        if not self._peers:
            return parent

        released = []
        with self._peer_lock:
            self._held.extend(parent)
            still_held = []
            for res in self._held:
                st = self._pending.get(res.job_id)
                if st is None:
                    released.append(res)
                    continue
                if st["got"] < st["want"]:
                    still_held.append(res)
                    continue
                self._pending.pop(res.job_id, None)
                keys = self._load_keys_mirror.pop(res.job_id, None)
                if st["ok"]:
                    released.append(res)
                else:
                    # Fail closed: a promotion that did not complete on every
                    # rank must become a MISS, never a half-restored prefix.
                    if keys:
                        try:
                            self._lookup_manager.mark_miss(keys)
                        except Exception:
                            logger.exception("peer-cascade: mark_miss failed")
                    released.append(
                        JobResult(job_id=res.job_id, success=False,
                                  successful_keys=None))
            self._held = still_held
        return released

    # -- eviction mirroring ----------------------------------------------
    def _evict_if_needed(self) -> None:
        """Mirror the bounded tier's LRU deletions to every peer.

        The cap is enforced once, here, by the parent class; the peers just
        delete the same blocks from their own shard so a single configurable
        max_bytes bounds the whole cluster. Cheap: the set diff only runs on
        the steps that actually evict (_used_bytes() is O(1)).
        """
        if not self._peers or self._used_bytes() <= self._max_bytes:
            super()._evict_if_needed()
            return
        before = set(self._tracked)
        super()._evict_if_needed()
        removed = before - set(self._tracked)
        if removed:
            self._mirror_evict(removed)

    def _mirror_evict(self, paths) -> None:
        own = "_r%d/" % self.file_mapper.rank
        for peer in self._peers:
            mapped = [x.replace(own, "_r%d/" % peer.rank, 1) for x in paths]
            self._pool_exec.submit(
                peer.rpc, {"op": "evict", "paths": mapped})
        _plog("info", "mirrored eviction of %d blocks to %d peer(s)",
              len(paths), len(self._peers))

    def shutdown(self) -> None:
        try:
            if self._pool_exec is not None:
                self._pool_exec.shutdown(wait=False)
            for p in self._peers:
                with p._lock:
                    p.close_locked()
        finally:
            super().shutdown()
