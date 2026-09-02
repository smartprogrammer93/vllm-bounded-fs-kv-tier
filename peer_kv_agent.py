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
  -> {"op":"evict","paths_rel":[..]}         # mirror the head's LRU deletions
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
import traceback

LOG_PREFIX = "[peer-kv-agent]"

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
            log("attached region %s (%d bytes, rank=%s, block_size=%s)"
                % (self.path, want, cfg["rank"], cfg["block_size"]))
            return self.path

    def path_for(self, key_hex: str, group_idx: int) -> str:
        c = self.cfg
        return ("%s/%s_r%d/%s/%s_g%d/%s.bin"
                % (c["root_dir"], c["basename"], c["rank"],
                   key_hex[:3], key_hex[3:5], group_idx, key_hex))


REGION = Region()


def split_key(key_hex: str):
    """OffloadKey = block_hash || group_idx (4 bytes, big-endian)."""
    raw = bytes.fromhex(key_hex)
    return raw[:-4].hex(), int.from_bytes(raw[-4:], "big", signed=False)


def handle(msg: dict) -> dict:
    op = msg.get("op")
    if op == "hello":
        p = REGION.configure(msg)
        return {"ok": True, "mmap": p, "vllm_io": _HAVE_VLLM_IO}

    if op == "evict":
        n = 0
        for rel in msg.get("paths_rel", []):
            try:
                os.unlink(os.path.join(REGION.cfg["root_dir"], rel))
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
    paths, offsets = [], []
    for key_hex, bid in zip(msg["keys"], msg["block_ids"]):
        h, g = split_key(key_hex)
        paths.append(REGION.path_for(h, g))
        offsets.append(int(bid) * block_size)

    if op == "store":
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)

    fn = ((batch_store_block if op == "store" else batch_load_block)
          if _HAVE_VLLM_IO else (_py_store if op == "store" else _py_load))
    try:
        fn(paths, REGION.view, offsets, block_size, o_direct)
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
