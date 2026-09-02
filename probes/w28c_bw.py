#!/usr/bin/env python3
"""Bandwidth of the SAME kernels over a cudaMalloc pool vs a cudaMallocManaged
pool, using torch's CUDAPluggableAllocator -- the exact mechanism vLLM would use
to move its KV pool to host-addressable memory.

The allocator must be swapped before the first CUDA allocation, so each arm runs
as its own process: `python3 w28c_bw.py device|managed`.
"""
import os, sys, time

ARM = sys.argv[1]
MB = 1 << 20
SIZE = int(os.environ.get("W28C_MB", "512")) * MB
ITERS = int(os.environ.get("W28C_ITERS", "40"))

import torch
if ARM == "managed":
    alloc = torch.cuda.memory.CUDAPluggableAllocator(
        "/tmp/managed_alloc.so", "managed_malloc", "managed_free")
    torch.cuda.memory.change_current_allocator(alloc)

t = torch.zeros(SIZE, dtype=torch.uint8, device="cuda")
assert t.is_cuda, "not a CUDA tensor"
torch.cuda.synchronize()
for _ in range(5):
    t.add_(1)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(ITERS):
    t.add_(1)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
gbps = (t.numel() * 2 * ITERS) / 1e9 / dt
print("%s %.1f" % (ARM, gbps), flush=True)

# For the managed arm, prove the pool really is host-addressable: the whole
# point of the exercise.
if ARM == "managed":
    import ctypes
    try:
        buf = (ctypes.c_ubyte * 64).from_address(t.data_ptr())
        _ = bytes(buf)
        path = "/tmp/w28c_iov.bin"
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        n = os.pwritev(fd, [memoryview((ctypes.c_ubyte * MB).from_address(t.data_ptr()))], 0)
        os.close(fd); os.unlink(path)
        print("managed_host_addressable 1 pwritev_bytes %d" % n, flush=True)
    except BaseException as e:
        print("managed_host_addressable 0 %s: %s" % (type(e).__name__, e), flush=True)
