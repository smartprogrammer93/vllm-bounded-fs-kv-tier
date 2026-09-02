"""Verify the per-rank cascade against the real vLLM classes.

Covers the upstream facts the fix depends on (so an engine upgrade that moves
them fails here rather than silently corrupting KV), the wire framing, and the
agent's pure logic.

Run inside a container/venv that has vLLM installed. No GPU, no CUDA context.
"""
import importlib.util
import inspect
import io
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

print("=== 1. upstream premises the fix relies on ===")
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering import spec as tiering_spec
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.base import make_offload_key, get_offload_group_idx

reg_src = inspect.getsource(SharedOffloadRegion)
check("CPU region is per-node and rank-subdivided (the whole reason the "
      "disk tier needs a per-rank cascade)",
      "_worker_offset" in reg_src and "rank * cpu_page_size" in reg_src)
check("CPU region path is keyed only by engine_id (so each host has its own)",
      "/dev/shm/vllm_offload_" in reg_src)

spec_src = inspect.getsource(tiering_spec)
check("secondary tiers are built over the PRIMARY tier's local memoryview",
      "primary_tier.get_kv_memoryview()" in spec_src)
check("secondary tiers are built in get_manager (scheduler side), not get_worker",
      "def get_manager" in spec_src)

fs_src = inspect.getsource(FileSystemTierManager)
for attr in ("_primary_kv_view", "_block_size", "_lookup_manager",
             "file_mapper", "_use_o_direct"):
    check(f"parent still exposes {attr}", attr in fs_src)
check("parent load failure path calls mark_miss (our failure policy reuses it)",
      "mark_miss" in fs_src)
check("parent submit_store/submit_load take job_metadata with keys+block_ids",
      "job_metadata.keys" in fs_src and "job_metadata.block_ids" in fs_src)

print("=== 2. FileMapper layout our agent reproduces byte-for-byte ===")
fm_src = inspect.getsource(FileMapper)
check("base_path excludes rank, so _r0/_r1 are siblings",
      "rank lives outside the hash" in fm_src or "_r{self.rank}" in fm_src)
mapper = FileMapper(root_dir="/kvoffload", model_name="m", tokens_per_hash=64,
                    blocks_per_file=1, tp_size=2, pp_size=1, pcp_size=1,
                    dcp_size=1, rank=0, dtype="torch.bfloat16",
                    kv_cache_groups=[{"tokens_per_block": 3328,
                                      "layer_names": ["a"]}])
mapper1 = FileMapper(root_dir="/kvoffload", model_name="m", tokens_per_hash=64,
                     blocks_per_file=1, tp_size=2, pp_size=1, pcp_size=1,
                     dcp_size=1, rank=1, dtype="torch.bfloat16",
                     kv_cache_groups=[{"tokens_per_block": 3328,
                                       "layer_names": ["a"]}])
check("rank does not change base_path", mapper.base_path == mapper1.base_path)

key = make_offload_key(bytes(range(32)), 3)
agent = load(os.path.join(ROOT, "peer_kv_agent.py"), "peer_kv_agent_t")
agent.REGION.cfg = {"root_dir": "/kvoffload",
                    "basename": os.path.basename(mapper.base_path),
                    "rank": 1, "block_size": 4096, "max_bytes": 0}
h, g = agent.split_key(bytes(key).hex())
check("agent recovers the group index from the key", g == 3, str(g))
check("agent path == FileMapper path for the same key and rank",
      agent.REGION.path_for(h, g) == mapper1.get_file_name(key),
      f"{agent.REGION.path_for(h, g)} vs {mapper1.get_file_name(key)}")

print("=== 3. hashes_per_chunk must stay >= 1 (why edit 5 clamps it) ===")
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    scheduler as off_sched, events as off_events)
sch_src = inspect.getsource(off_sched)
check("update_offload_keys still steps islice by hashes_per_chunk "
      "(0 would raise, and it runs for EVERY group)",
      "islice(" in sch_src and "hashes_per_chunk," in sch_src)
ev_src = inspect.getsource(off_events)
check("events.py still divides by hashes_per_chunk",
      "// hashes_per_chunk" in ev_src or "// group_config.hashes_per_chunk" in ev_src)
check("events.py still asserts hashes_per_chunk > 0",
      "assert hashes_per_chunk > 0" in ev_src)

print("=== 4. wire framing (regression guard) ===")
peer_mod = load(os.path.join(ROOT, "vllm_bounded_fs_tier_peer.py"),
                "vllm_bounded_fs_tier_peer_t")
_Peer = peer_mod._Peer
p = _Peer("127.0.0.1:1", rank=1, timeout=1)


class _FakeSock:
    def __init__(self):
        self.sent = io.BytesIO()

    def makefile(self, mode):
        return self.sent if "w" in mode else io.BytesIO(b'{"ok": true}\n')

    def settimeout(self, _):
        pass

    def close(self):
        pass


fake = _FakeSock()
p._sock = fake
p._wf = fake.sent
p._rf = io.BytesIO(b'{"ok": true, "job_id": 7}\n')
resp = p._rpc_locked({"op": "store", "job_id": 7})
raw = fake.sent.getvalue()
check("frame ends with a REAL newline, not a literal backslash-n "
      "(this exact bug made the agent's readline() block forever)",
      raw.endswith(b"\n") and b"\\n" not in raw, repr(raw[-12:]))
check("frame is valid JSON", json.loads(raw.decode().strip())["job_id"] == 7)
check("response is parsed back", resp.get("ok") is True)

print("=== 5. failure policy is fail-closed ===")
src = inspect.getsource(peer_mod.PeerMirroredFileSystemTierManager)
check("a peer failure marks a miss instead of releasing a partial restore",
      "mark_miss" in src)
check("results are withheld until every peer acknowledges", "_held" in src)
check("failed jobs are released as success=False",
      "success=False" in src)
check("eviction is mirrored so one max_bytes bounds every rank",
      "_mirror_evict" in src and "_r%d/" in src)

print("=== 5b. construction order (regression guard) ===")
init_src = inspect.getsource(peer_mod.PeerMirroredFileSystemTierManager.__init__)
i_peers = init_src.find("self._peers")
i_super = init_src.find("super().__init__")
check("_peers is assigned BEFORE super().__init__ -- the bounded parent's "
      "constructor calls _evict_if_needed(), which our override reads it in",
      i_peers != -1 and i_super != -1 and i_peers < i_super,
      f"peers@{i_peers} super@{i_super}")
for attr in ("_pool_exec", "_pending", "_held", "_peer_lock",
             "_load_keys_mirror"):
    idx = init_src.find("self." + attr)
    check(f"{attr} assigned before super().__init__",
          idx != -1 and idx < i_super, f"{attr}@{idx}")
bounded_init = inspect.getsource(peer_mod.BoundedFileSystemTierManager.__init__)
check("the parent constructor really does call _evict_if_needed "
      "(this is why the ordering matters)",
      "_evict_if_needed" in bounded_init)

print("=== 6. agent: eviction confinement and backstop sweep ===")
with tempfile.TemporaryDirectory() as td:
    agent.REGION.cfg = {"root_dir": td, "basename": "b", "rank": 1,
                        "block_size": 1024, "max_bytes": 4096}
    outside = os.path.join(tempfile.gettempdir(), "peer_cascade_outside.bin")
    with open(outside, "wb") as f:
        f.write(b"x")
    r = agent.handle({"op": "evict", "paths": [outside]})
    check("evict refuses to unlink outside root_dir",
          os.path.exists(outside) and r["deleted"] == 0)
    os.unlink(outside)

    inside = os.path.join(td, "sub", "a.bin")
    os.makedirs(os.path.dirname(inside), exist_ok=True)
    with open(inside, "wb") as f:
        f.write(b"y")
    r = agent.handle({"op": "evict", "paths": [inside]})
    check("evict removes a path inside root_dir",
          not os.path.exists(inside) and r["deleted"] == 1)

    # 8 x 1 KiB against a 4 KiB cap -> trim to <= 90% of cap, oldest first
    paths = []
    for i in range(8):
        fp = os.path.join(td, "g", f"{i:02d}.bin")
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "wb") as f:
            f.write(b"z" * 1024)
        os.utime(fp, (1000 + i, 1000 + i))
        paths.append(fp)
    removed = agent._sweep_once()
    left = [q for q in paths if os.path.exists(q)]
    total = sum(os.path.getsize(q) for q in left)
    check("backstop sweep trims to the cap", total <= 4096, f"{total} bytes")
    check("backstop sweep evicts oldest-first",
          all(not os.path.exists(q) for q in paths[:len(paths) - len(left)]),
          f"{len(left)} left")
    check("backstop sweep reports what it removed", removed > 0, str(removed))
    check("sweep is a no-op under the cap", agent._sweep_once() == 0)

print()
if fails:
    print(f"FAILED ({len(fails)}): " + ", ".join(fails))
    sys.exit(1)
print("ALL PEER-CASCADE CHECKS PASSED")
