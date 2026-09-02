#!/usr/bin/env python3
"""Per-rank KV cascade agent — makes vLLM's disk offload tier correct on a
tensor-parallel group that spans hosts.

WHY THIS EXISTS
---------------
vLLM builds every offloading secondary tier scheduler-side, over a memoryview
of the *local* node's CPU primary tier:

    # tiering/spec.py
    primary_kv_view = primary_tier.get_kv_memoryview()
    tier = SecondaryTierFactory.create_secondary_tier(tier_config, primary_kv_view, self)

and that CPU tier is a per-node `/dev/shm/vllm_offload_<engine_id>.mmap` whose
docstring states the assumption outright: "Single mmap-backed memory region
shared across all workers for a vLLM instance. Workers coordinate via the
filesystem." Each block row is subdivided by rank
(`_worker_offset = rank * cpu_page_size`); the scheduler's `rank=None` view
spans every rank's sub-slot.

That holds on one node. When the TP group spans hosts (`--nnodes 2`, one
`vllm serve` per node) each node creates its OWN region and only its own rank's
sub-slot is ever written there. The scheduler's disk tier then cascades and
promotes only the head's region, while still issuing the resulting load to every
worker — so the remote rank copies stale bytes into its GPU blocks. The request
is marked as having a valid cached prefix and the model emits garbage, with no
error anywhere.

WHAT THIS DOES
--------------
Runs inside the remote rank's container and performs byte-identical cascade and
promotion against *that* rank's own region and its own rank-namespaced files, so
every rank's shard reaches and returns from disk. The head's tier drives it and
does not report a job complete until this agent has acknowledged it, which keeps
disk -> CPU ordered before CPU -> GPU.

`FileMapper.base_path` deliberately excludes rank ("rank lives outside the
hash"), so `_r0` and `_r1` are siblings under one namespace and cannot collide.

Protocol: newline-delimited JSON over TCP.
  -> {"op":"hello","basename":..,"rank":N,"block_size":B,"total_size":S,
      "root_dir":..,"o_direct":bool}
  <- {"ok":true,"mmap":"/dev/shm/..."}
  -> {"op":"store"|"load","job_id":J,"keys":[hex,..],"block_ids":[..]}
  <- {"ok":bool,"job_id":J,"num_succeeded":N,"err":str|null}
  -> {"op":"evict","paths":[abs,..]}         # mirror the head LRU deletions
  <- {"ok":true,"deleted":N}

A failure here is reported honestly so the head can turn the job into a cache
MISS (recompute) instead of a silent wrong answer.
"""
from __future__ import annotations

import glob
import json
import mmap
import os
import socketserver
import sys
import threading
import time as _time
import traceback

LOG_PREFIX = "[peer-kv-agent]"
_SWEEPER_STARTED = False

try:
    from vllm.v1.kv_offload.tiering.fs.io import batch_load_block, batch_store_block
    _HAVE_VLLM_IO = True
except Exception:  # pragma: no cover - fallback keeps the agent usable
    _HAVE_VLLM_IO = False


def log(msg: str) -> None:
    print("%s %s" % (LOG_PREFIX, msg), flush=True)


def _py_store(paths, view, offsets, block_size, o_direct):
    for path, off in zip(paths, offsets):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp%d" % os.getpid()
        with open(tmp, "wb") as f:
            f.write(view[off:off + block_size])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def _py_load(paths, view, offsets, block_size, o_direct):
    for i, (path, off) in enumerate(zip(paths, offsets)):
        with open(path, "rb") as f:
            data = f.read(block_size)
        if len(data) != block_size:
            err = OSError("short read %d != %d for %s" % (len(data), block_size, path))
            err.num_succeeded = i
            raise err
        view[off:off + block_size] = data


class Region:
    """Lazily attaches to this rank's CPU offload region."""

    def __init__(self):
        self._lock = threading.Lock()
        self.mm = None
        self.view = None
        self.path = None
        self.cfg = None

    def configure(self, cfg: dict) -> str:
        with self._lock:
            self.cfg = cfg
            if self.view is not None:
                return self.path
            want = int(cfg["total_size"])
            candidates = []
            for p in glob.glob("/dev/shm/vllm_offload_*.mmap"):
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size == want:
                    candidates.append((st.st_mtime, p))
            if not candidates:
                sizes = {p: os.path.getsize(p)
                         for p in glob.glob("/dev/shm/vllm_offload_*.mmap")}
                raise RuntimeError(
                    "no /dev/shm offload region of exactly %d bytes; found %r"
                    % (want, sizes))
            candidates.sort()
            self.path = candidates[-1][1]
            fd = os.open(self.path, os.O_RDWR)
            try:
                self.mm = mmap.mmap(fd, want, flags=mmap.MAP_SHARED,
                                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fd)
            self.view = memoryview(self.mm)
            log("attached region %s (%d bytes, rank=%s, block_size=%s, "
                "rank_local_page=%s)"
                % (self.path, want, cfg["rank"], cfg["block_size"],
                   cfg.get("page_size") or "off"))
            return self.path

    def path_for(self, key_hex: str, group_idx: int) -> str:
        c = self.cfg
        return ("%s/%s_r%d/%s/%s_g%d/%s.bin"
                % (c["root_dir"], c["basename"], c["rank"],
                   key_hex[:3], key_hex[3:5], group_idx, key_hex))


REGION = Region()


def _sweep_once() -> int:
    """Trim this rank's shard to the same configurable cap the head enforces.

    Eviction is normally mirrored from the head, which owns the LRU order. This
    is only a backstop so a dropped message cannot leak disk forever; it deletes
    oldest-first by mtime, the same policy the head's bounded tier uses.
    """
    cfg = REGION.cfg
    if not cfg:
        return 0
    cap = int(cfg.get("max_bytes") or 0)
    root = cfg.get("root_dir")
    if cap <= 0 or not root or not os.path.isdir(root):
        return 0
    found = []
    total = 0
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            if not name.endswith(".bin"):
                continue
            fp = os.path.join(dirpath, name)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            found.append((st.st_mtime, fp, st.st_size))
            total += st.st_size
    if total <= cap:
        return 0
    found.sort()
    target = int(cap * 0.9)
    removed = 0
    for _mt, fp, size in found:
        if total <= target:
            break
        try:
            os.unlink(fp)
            total -= size
            removed += 1
        except OSError:
            pass
    if removed:
        log("backstop sweep removed %d blocks; usage now %.2f GiB of %.2f GiB cap"
            % (removed, total / 1024 ** 3, cap / 1024 ** 3))
    return removed


def _sweeper():
    interval = float(os.environ.get("PEER_KV_SWEEP_SECONDS", "60"))
    while True:
        try:
            _sweep_once()
        except Exception:
            traceback.print_exc()
        _time.sleep(interval)


def split_key(key_hex: str):
    """OffloadKey = block_hash || group_idx (4 bytes, big-endian)."""
    raw = bytes.fromhex(key_hex)
    return raw[:-4].hex(), int.from_bytes(raw[-4:], "big", signed=False)


def handle(msg: dict) -> dict:
    op = msg.get("op")
    if op == "hello":
        p = REGION.configure(msg)
        global _SWEEPER_STARTED
        if not _SWEEPER_STARTED:
            _SWEEPER_STARTED = True
            threading.Thread(target=_sweeper, daemon=True,
                             name="peer-kv-sweeper").start()
            log("backstop sweeper armed (cap=%.2f GiB)"
                % (int(msg.get("max_bytes") or 0) / 1024 ** 3))
        return {"ok": True, "mmap": p, "vllm_io": _HAVE_VLLM_IO}

    if op == "evict":
        root = (REGION.cfg or {}).get("root_dir", "/kvoffload")
        n = 0
        for path in msg.get("paths", []):
            # Never unlink outside the configured cache root.
            if not os.path.abspath(path).startswith(os.path.abspath(root) + os.sep):
                log("refusing to unlink outside %s: %r" % (root, path))
                continue
            try:
                os.unlink(path)
                n += 1
            except OSError:
                pass
        return {"ok": True, "deleted": n}

    if op not in ("store", "load"):
        return {"ok": False, "err": "unknown op %r" % op}

    cfg = REGION.cfg
    if cfg is None or REGION.view is None:
        return {"ok": False, "job_id": msg.get("job_id"), "num_succeeded": 0,
                "err": "region not configured"}

    block_size = int(cfg["block_size"])
    o_direct = bool(cfg.get("o_direct", False))
    # W27b: a row spans every rank, but this rank's worker only touches its own
    # sub-slot -- so store/load just that slice instead of the whole row, which
    # otherwise puts the head's inert half on this host's disk as well.
    page = int(cfg.get("page_size") or 0)
    if page > 0:
        span = page
        # Slot 0, NOT rank * page. Each node has its own region with a single
        # worker, so that worker's slot index within it is its LOCAL rank (0).
        # Verified by dumping both regions: slot 0 populated, slot 1 all-zero on
        # both hosts. Using rank * page here stored and restored zeros for this
        # rank -- a silent-corruption bug (empty completions).
        base = int(cfg.get("local_slot", 0)) * page
    else:
        span = block_size
        base = 0
    paths, offsets = [], []
    for key_hex, bid in zip(msg["keys"], msg["block_ids"]):
        h, g = split_key(key_hex)
        paths.append(REGION.path_for(h, g))
        offsets.append(int(bid) * block_size + base)

    if op == "store":
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)

    fn = ((batch_store_block if op == "store" else batch_load_block)
          if _HAVE_VLLM_IO else (_py_store if op == "store" else _py_load))
    try:
        fn(paths, REGION.view, offsets, span, o_direct)
    except OSError as e:
        return {"ok": False, "job_id": msg.get("job_id"),
                "num_succeeded": int(getattr(e, "num_succeeded", 0)),
                "err": "%s: %s" % (type(e).__name__, e)}
    except Exception as e:
        return {"ok": False, "job_id": msg.get("job_id"), "num_succeeded": 0,
                "err": "%s: %s" % (type(e).__name__, e)}
    return {"ok": True, "job_id": msg.get("job_id"),
            "num_succeeded": len(paths), "err": None}


class Handler(socketserver.StreamRequestHandler):
    # A stalled peer must not wedge the engine: the head applies its own
    # timeout, and we keep the socket blocking-but-bounded here.
    timeout = 600

    def handle(self):
        log("connection from %s" % (self.client_address,))
        while True:
            try:
                line = self.rfile.readline()
            except Exception:
                break
            if not line:
                break
            try:
                msg = json.loads(line)
            except Exception as e:
                resp = {"ok": False, "err": "bad json: %s" % e}
            else:
                try:
                    resp = handle(msg)
                except Exception as e:
                    log("ERROR handling %s: %s" % (msg.get("op"), e))
                    traceback.print_exc()
                    resp = {"ok": False, "job_id": msg.get("job_id"),
                            "num_succeeded": 0, "err": str(e)}
            try:
                self.wfile.write((json.dumps(resp) + "\n").encode())
                self.wfile.flush()
            except Exception:
                break
        log("connection closed %s" % (self.client_address,))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    host = os.environ.get("PEER_KV_AGENT_HOST", "0.0.0.0")
    port = int(os.environ.get("PEER_KV_AGENT_PORT", "8799"))
    log("starting on %s:%d (vllm io=%s)" % (host, port, _HAVE_VLLM_IO))
    with Server((host, port), Handler) as srv:
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
