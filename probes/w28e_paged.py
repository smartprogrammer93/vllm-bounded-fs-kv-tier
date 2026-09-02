#!/usr/bin/env python3
"""Managed vs device bandwidth under a PAGED-ATTENTION-like access pattern.

The elementwise benchmark (0.91x) streams a contiguous buffer, which is the best
case for managed memory. Decode instead gathers scattered fixed-size blocks out
of a large KV pool, which is where page-table pressure would show up. This
measures that: random block gather, same kernels, one pool per allocator.

Run per arm so the allocator can be swapped before the first CUDA allocation:
    python3 w28e_paged.py device|managed
"""
import os, sys, time
import torch

ARM = sys.argv[1]
MB = 1 << 20
POOL_MB = int(os.environ.get("POOL_MB", "2048"))
BLOCK = int(os.environ.get("BLOCK_KB", "64")) * 1024   # bytes per gathered block
ITERS = int(os.environ.get("ITERS", "30"))

if ARM == "managed":
    alloc = torch.cuda.memory.CUDAPluggableAllocator(
        "/tmp/managed_alloc.so", "managed_malloc", "managed_free")
    torch.cuda.memory.change_current_allocator(alloc)

nblocks = (POOL_MB * MB) // BLOCK
pool = torch.zeros(nblocks, BLOCK, dtype=torch.uint8, device="cuda")
# gather ~1/8 of the pool per iteration, in scattered order, like attention
# reading the pages of one sequence out of a shared pool
k = max(1, nblocks // 8)
idx = torch.randperm(nblocks, device="cuda")[:k]
torch.cuda.synchronize()

for _ in range(4):
    out = pool.index_select(0, idx)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(ITERS):
    out = pool.index_select(0, idx)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
moved = k * BLOCK * 2 * ITERS          # read + write
print("%s %.1f GB/s  (pool %d MiB, %d blocks of %d KiB, gather %d)"
      % (ARM, moved / 1e9 / dt, POOL_MB, nblocks, BLOCK // 1024, k), flush=True)
