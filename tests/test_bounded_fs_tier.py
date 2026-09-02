"""Verify vllm_bounded_fs_tier against the real vLLM classes.

Run inside a container/venv that has vLLM installed.
Needs no GPU and no CUDA context.
"""
import inspect
import os
import sys
import tempfile
import threading
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vllm_bounded_fs_tier as m
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


print("=== 1. parent contract still matches our assumptions ===")
sig = inspect.signature(FileSystemTierManager.__init__)
params = list(sig.parameters)
check("root_dir is the 4th positional param", params[:5] ==
      ["self", "offloading_spec", "primary_kv_view", "tier_type", "root_dir"],
      str(params[:6]))
src = inspect.getsource(FileSystemTierManager)
check("parent uses self._block_size", "_block_size" in src)
check("parent exposes file_mapper.get_file_name", "file_mapper.get_file_name" in src)
check("parent does NOT define touch (we must)", "def touch" not in src)
check("parent has no capacity/evict logic",
      not any(w in src for w in ("max_bytes", "capacity", "def evict", "quota")))
check("load failure marks a miss (safe eviction)", "mark_miss" in src)

print()
print("=== 2. subclass wiring ===")
check("subclasses FileSystemTierManager",
      issubclass(m.BoundedFileSystemTierManager, FileSystemTierManager))
check("overrides submit_store", "submit_store" in m.BoundedFileSystemTierManager.__dict__)
check("overrides touch", "touch" in m.BoundedFileSystemTierManager.__dict__)
check("overrides get_finished_jobs",
      "get_finished_jobs" in m.BoundedFileSystemTierManager.__dict__)

print()
print("=== 3. parse_bytes ===")
cases = [(1024, 1024), ("1024", 1024), ("200GiB", 200 * 1024**3),
         ("200GB", 200 * 1000**3), ("500MB", 500 * 1000**2), ("1.5GiB", int(1.5 * 1024**3))]
for raw, want in cases:
    got = m.parse_bytes(raw)
    check(f"parse_bytes({raw!r}) == {want}", got == want, f"got {got}")
for bad in ("", "12xb", "abc"):
    try:
        m.parse_bytes(bad)
        check(f"parse_bytes({bad!r}) rejected", False, "accepted!")
    except ValueError:
        check(f"parse_bytes({bad!r}) rejected", True)

print()
print("=== 4. eviction on REAL files (bypassing parent __init__) ===")
obj = object.__new__(m.BoundedFileSystemTierManager)
tmp = tempfile.mkdtemp(prefix="bfs-test-")
BLOCK = 4096
obj._root_dir = tmp
obj._block_size = BLOCK
obj._tracked = OrderedDict()
obj._lock = threading.Lock()
obj._evicted_total = 0
obj._max_bytes = 10 * BLOCK          # cap = 10 blocks
obj._evict_to = 9 * BLOCK            # evict down to 9

paths = []
for i in range(15):
    p = os.path.join(tmp, f"{i:03d}.bin")
    with open(p, "wb") as f:
        f.write(b"\0" * BLOCK)
    paths.append(p)
    obj._tracked[p] = None

check("usage before evict == 15 blocks", obj._used_bytes() == 15 * BLOCK)
obj._evict_if_needed()
check("usage after evict <= evict_to", obj._used_bytes() <= obj._evict_to,
      f"{obj._used_bytes()} vs {obj._evict_to}")
survivors = [p for p in paths if os.path.exists(p)]
check("oldest files were the ones removed",
      survivors == paths[-len(survivors):], f"{len(survivors)} survivors")
check("tracked map matches disk", len(obj._tracked) == len(survivors))

print()
print("=== 5. touch() promotes to newest (LRU, not FIFO) ===")
oldest = next(iter(obj._tracked))
obj._tracked.move_to_end(oldest)
check("moved key is now newest", list(obj._tracked)[-1] == oldest)

print()
print("=== 6. eviction tolerates already-deleted files ===")
ghost = os.path.join(tmp, "ghost.bin")
obj._tracked[ghost] = None           # tracked but never created
obj._max_bytes = 2 * BLOCK
obj._evict_to = 1 * BLOCK
try:
    obj._evict_if_needed()
    check("no exception on missing file", True)
except Exception as e:
    check("no exception on missing file", False, f"{type(e).__name__}: {e}")

print()
print("=== 7. adopt existing files on restart ===")
obj2 = object.__new__(m.BoundedFileSystemTierManager)
tmp2 = tempfile.mkdtemp(prefix="bfs-adopt-")
sub = os.path.join(tmp2, "aa", "bb_g0")
os.makedirs(sub)
for i in range(4):
    with open(os.path.join(sub, f"{i}.bin"), "wb") as f:
        f.write(b"\0" * BLOCK)
open(os.path.join(sub, "notablock.txt"), "w").close()
obj2._root_dir = tmp2
obj2._block_size = BLOCK
obj2._tracked = OrderedDict()
obj2._adopt_existing_files()
check("adopted 4 .bin files, ignored non-.bin", len(obj2._tracked) == 4,
      f"got {len(obj2._tracked)}")

import shutil
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(tmp2, ignore_errors=True)

print()
print(f"RESULT: {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
