#!/usr/bin/env python3
"""Offline test of the streamed-restore batch split.

A restore batch is (offload_keys, gpu_block_ids, group_sizes, block_indices).
One group's batch can be far larger than the CPU tier -- at 3584 tokens per
offload block a 1M-token restore is ~280 blocks -- so it must be split, and the
split must preserve the worker's contract exactly:

  * sum(group_sizes) == len(block_ids)            (GPULoadStoreSpec asserts it)
  * block_indices[gi] is the LOGICAL index of the first GPU block in the group,
    which is how the worker knows to skip part of the first offload block
  * concatenating the sub-batches reproduces the original, in order

Chunk j of a group's slice covers GPU blocks [j*bpc - off, (j+1)*bpc - off)
relative to the slice start, where bpc = blocks_per_chunk and
off = block_indices[gi] % bpc. Getting `off` wrong silently shifts every block
by less than one chunk -- KV that loads without error and is wrong. So this is
arithmetic worth testing rather than debugging live.
"""
from pathlib import Path

import ast
import importlib.util
import os
import sys
import types

PRISTINE = Path(os.environ.get("PRISTINE", "/tmp/pristine/vllm"))
PATCH = Path(os.environ.get(
    "PATCH", "/home/ahmad/glm53-exl3/overlay/patch_offload_streaming_restore.py"))
GROUPS = Path(os.environ.get(
    "GROUPS_PATCH",
    "/home/ahmad/glm53-exl3/overlay/patch_offload_group_filter.py"))

fails = []


def check(name, cond, detail=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", name,
                           (" -- " + detail) if detail else ""))
    if not cond:
        fails.append(name)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_split():
    """Pull _stream_split_batches out of the PATCHED scheduler source.

    The point is to test the function that actually ships, not a copy of it in
    this file -- a copy drifts silently, and the arithmetic here is exactly the
    kind that fails without raising.
    """
    m = _load("sr", PATCH)
    g = _load("gf", GROUPS)
    path = PRISTINE / m.SCHED.relative_to(m.VLLM)
    src = path.read_text()
    for label, old, new, sentinel in g.EDITS + m.SCHED_EDITS + m.SCHED2B_EDITS:
        if sentinel in src:
            continue
        assert src.count(old) == 1, "%s matched %d" % (label, src.count(old))
        src = src.replace(old, new)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_stream_split_batches")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<patch>", "exec"), ns)
    raw = ns["_stream_split_batches"]

    def split(batches, limit, bpc):
        stub = types.SimpleNamespace(
            _stream_batch_blocks=lambda: limit,
            config=types.SimpleNamespace(blocks_per_chunk=bpc),
        )
        return raw(stub, batches)

    return split


split = _extract_split()


def make_batch(ngroups, gi, n_keys, first_logical, bpc, n_blocks=None):
    """A group's batch as the scheduler builds it.

    first_logical is num_locally_computed_gpu_blocks; the slice runs from there
    to the end of the last (possibly partial) chunk it covers.
    """
    off = first_logical % bpc
    if n_blocks is None:
        n_blocks = n_keys * bpc - off
    keys = ["k%d" % j for j in range(n_keys)]
    blocks = list(range(1000 + first_logical, 1000 + first_logical + n_blocks))
    gs = [0] * ngroups
    bi = [0] * ngroups
    gs[gi] = n_blocks
    bi[gi] = first_logical
    return keys, blocks, gs, bi


CASES = [
    ("chunk-aligned start", 7, 0, 20, 0, 56),
    ("unaligned start", 7, 2, 20, 13, 56),
    ("unaligned, small chunks", 7, 3, 17, 5, 8),
    ("single key", 7, 1, 1, 0, 56),
    ("exactly at the limit", 7, 4, 4, 3, 56),
    ("last group", 7, 6, 33, 7, 4),
]

print("=== 1. the split preserves the batch exactly ===")
for name, ng, gi, nk, first, bpc in CASES:
    keys, blocks, gs, bi = make_batch(ng, gi, nk, first, bpc)
    for limit in (1, 2, 3, 4, 7, nk, nk + 5):
        subs = split([(keys, blocks, gs, bi)], limit, bpc)
        rk = [k for s in subs for k in s[0]]
        rb = [b for s in subs for b in s[1]]
        if rk != keys or rb != blocks:
            check("%s @limit=%d: concatenation is the original" % (name, limit),
                  False, "keys %d/%d blocks %d/%d"
                  % (len(rk), len(keys), len(rb), len(blocks)))
            break
    else:
        check("%s: every limit reproduces the original" % name, True)

print("=== 2. every sub-batch satisfies GPULoadStoreSpec's assert ===")
bad = []
for name, ng, gi, nk, first, bpc in CASES:
    keys, blocks, gs, bi = make_batch(ng, gi, nk, first, bpc)
    for limit in (1, 2, 3, 5, 8):
        for s in split([(keys, blocks, gs, bi)], limit, bpc):
            if sum(s[2]) != len(s[1]):
                bad.append("%s@%d: sum(gs)=%d len=%d"
                           % (name, limit, sum(s[2]), len(s[1])))
            if len(s[3]) != len(s[2]):
                bad.append("%s@%d: len mismatch" % (name, limit))
check("sum(group_sizes) == len(block_ids) everywhere", not bad, str(bad[:3]))

print("=== 3. block_indices names the first block's LOGICAL index ===")
# This is the field that makes the worker skip the right part of the first
# offload block. Off by less than a chunk = silently wrong KV.
bad = []
for name, ng, gi, nk, first, bpc in CASES:
    keys, blocks, gs, bi = make_batch(ng, gi, nk, first, bpc)
    for limit in (1, 2, 3, 5):
        for s in split([(keys, blocks, gs, bi)], limit, bpc):
            # blocks were numbered 1000 + logical index
            want = s[1][0] - 1000
            if s[3][gi] != want:
                bad.append("%s@%d: block_indices=%d want %d"
                           % (name, limit, s[3][gi], want))
check("block_indices matches the first block's logical index", not bad,
      str(bad[:3]))

print("=== 4. sub-batches after the first are chunk-aligned ===")
# Only the first sub-batch may begin mid-chunk; every later one starts exactly
# on a chunk boundary, which is why its block_indices needs no correction.
bad = []
for name, ng, gi, nk, first, bpc in CASES:
    keys, blocks, gs, bi = make_batch(ng, gi, nk, first, bpc)
    for limit in (1, 2, 3, 5):
        subs = split([(keys, blocks, gs, bi)], limit, bpc)
        for s in subs[1:]:
            if s[3][gi] % bpc != 0:
                bad.append("%s@%d: %d %% %d = %d"
                           % (name, limit, s[3][gi], bpc, s[3][gi] % bpc))
check("later sub-batches start on a chunk boundary", not bad, str(bad[:3]))

print("=== 5. no sub-batch exceeds the limit ===")
bad = []
for name, ng, gi, nk, first, bpc in CASES:
    keys, blocks, gs, bi = make_batch(ng, gi, nk, first, bpc)
    for limit in (1, 2, 3, 5, 8):
        for s in split([(keys, blocks, gs, bi)], limit, bpc):
            if len(s[0]) > limit:
                bad.append("%s@%d: %d keys" % (name, limit, len(s[0])))
check("every sub-batch has at most `limit` offload keys", not bad, str(bad[:3]))

print("=== 6. a partial tail chunk survives the split ===")
# The scheduler appends a boundary key for a partial final chunk, so the last
# chunk can cover fewer than bpc GPU blocks.
bpc = 56
keys, blocks, gs, bi = make_batch(7, 0, 10, 0, bpc, n_blocks=9 * bpc + 17)
for limit in (1, 3, 4, 9, 10):
    subs = split([(keys, blocks, gs, bi)], limit, bpc)
    rb = [b for s in subs for b in s[1]]
    ok = rb == blocks and all(sum(s[2]) == len(s[1]) for s in subs)
    check("partial tail preserved @limit=%d" % limit, ok,
          "%d of %d blocks" % (len(rb), len(blocks)))

print("=== 7. other groups stay empty in every sub-batch ===")
# A non-zero size in another group's slot would make the worker read the wrong
# tensor entirely.
bad = []
for name, ng, gi, nk, first, bpc in CASES:
    keys, blocks, gs, bi = make_batch(ng, gi, nk, first, bpc)
    for s in split([(keys, blocks, gs, bi)], 3, bpc):
        if any(v for i, v in enumerate(s[2]) if i != gi):
            bad.append(name)
        if any(v for i, v in enumerate(s[3]) if i != gi):
            bad.append(name + " (indices)")
check("only the batch's own group is non-zero", not bad, str(set(bad)))

print("=== 8. multi-group input is split independently ===")
bpc = 56
b0 = make_batch(7, 0, 12, 0, bpc)
b1 = make_batch(7, 2, 12, 13, bpc)
subs = split([b0, b1], 5, bpc)
check("all sub-batches accounted for", len(subs) == 3 + 3, str(len(subs)))
check("group 0's keys stay with group 0",
      [k for s in subs if s[2][0] for k in s[0]] == b0[0])
check("group 2's keys stay with group 2",
      [k for s in subs if s[2][2] for k in s[0]] == b1[0])

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL SPLIT TESTS PASSED")
