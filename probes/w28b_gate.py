#!/usr/bin/env python3
"""W28b gate: is there an allocation type that is BOTH host-addressable (so
pwritev/preadv work on it) AND usable as GPU memory?

W28a showed plain cudaMalloc memory is not host-dereferenceable on GB10, which
killed "gather KV straight to disk". But that is an allocation-POLICY fact, not a
Linux limitation: the kernel simply refuses an address with no mapping in this
process. CUDA has allocation types that ARE mapped host-side:

  * cudaMallocManaged  -- unified/managed, migrates on access
  * cudaHostAlloc      -- pinned host memory, GPU-accessible
  * plain malloc       -- on Grace-class parts with ATS, GPU-accessible directly

If one of those is host-addressable for iovec I/O *and* the GPU can read it, then
allocating the vLLM KV pool through a pluggable allocator would remove the CPU
staging tier, the uniform-row padding and the restore-window cap together.

This gate answers only the addressability half. The performance half -- do the
EXL3/DFlash attention kernels keep full speed against such memory -- is a
separate, larger question and is NOT answered here.

Every probe runs in a forked child: dereferencing unmapped device memory is a
segfault, not an exception.
"""
import ctypes
import ctypes.util
import os
import sys

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok))
    print("  %-4s %s%s" % ("PASS" if ok else "FAIL", name,
                           (" -- " + detail) if detail else ""), flush=True)


def in_child(fn):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.write(w, ("OK " + str(fn() or "ok")).encode()[:4000])
            os._exit(0)
        except BaseException as e:
            os.write(w, ("ERR %s: %s" % (type(e).__name__, e)).encode()[:4000])
            os._exit(1)
    os.close(w)
    out = b""
    while True:
        c = os.read(r, 4096)
        if not c:
            break
        out += c
    os.close(r)
    _, st = os.waitpid(pid, 0)
    if os.WIFSIGNALED(st):
        return False, "child died on signal %d (not host-dereferenceable)" % os.WTERMSIG(st)
    return os.WEXITSTATUS(st) == 0, out.decode() or "no output"


def cudart():
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    p = ctypes.util.find_library("cudart")
    if p:
        return ctypes.CDLL(p)
    raise RuntimeError("libcudart not found")


N = 1 << 20  # 1 MiB
PATTERN = bytes((i * 7 + 3) % 251 for i in range(N))


def _iov_write(ptr, n, odirect=False):
    """Write n bytes at host address ptr to a file via pwritev; return content."""
    path = "/tmp/w28b_%d.bin" % os.getpid()
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    if odirect:
        flags |= getattr(os, "O_DIRECT", 0)
    fd = os.open(path, flags, 0o600)
    try:
        wrote = os.pwritev(fd, [memoryview((ctypes.c_ubyte * n).from_address(ptr))], 0)
    finally:
        os.close(fd)
    data = open(path, "rb").read()
    os.unlink(path)
    if wrote != n:
        raise AssertionError("pwritev wrote %d/%d" % (wrote, n))
    return data


def probe_managed():
    """cudaMallocManaged: host writes, pwritev reads, GPU sees the same bytes."""
    rt = cudart()
    ptr = ctypes.c_void_p()
    # cudaMemAttachGlobal = 1
    rc = rt.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(N), ctypes.c_uint(1))
    if rc != 0:
        raise AssertionError("cudaMallocManaged rc=%d" % rc)
    ctypes.memmove(ptr, PATTERN, N)              # CPU write -- must not fault
    data = _iov_write(ptr.value, N)              # kernel must accept the address
    if data != PATTERN:
        raise AssertionError("pwritev content mismatch")
    # GPU-side visibility: copy device->host through the runtime and compare
    back = (ctypes.c_ubyte * N)()
    rc = rt.cudaMemcpy(ctypes.byref(back), ptr, ctypes.c_size_t(N), ctypes.c_int(3))
    rt.cudaDeviceSynchronize()
    if rc != 0 or bytes(back) != PATTERN:
        raise AssertionError("GPU-side view mismatch (rc=%d)" % rc)
    rt.cudaFree(ptr)
    return "host write + pwritev + GPU-visible, %d bytes" % N


def probe_managed_odirect():
    rt = cudart()
    ptr = ctypes.c_void_p()
    if rt.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(N), ctypes.c_uint(1)) != 0:
        raise AssertionError("alloc failed")
    ctypes.memmove(ptr, PATTERN, N)
    if ptr.value % 4096:
        rt.cudaFree(ptr)
        return "SKIP: managed pointer 0x%x not 4096-aligned" % ptr.value
    data = _iov_write(ptr.value, N, odirect=True)
    rt.cudaFree(ptr)
    if data != PATTERN:
        raise AssertionError("O_DIRECT content mismatch")
    return "O_DIRECT pwritev from managed memory works"


def probe_pinned():
    """torch pinned host memory: host-addressable by construction; confirm the
    GPU can consume it and that iovec I/O works."""
    import torch
    t = torch.empty(N, dtype=torch.uint8, pin_memory=True)
    ctypes.memmove(t.data_ptr(), PATTERN, N)
    data = _iov_write(t.data_ptr(), N)
    if data != PATTERN:
        raise AssertionError("pwritev content mismatch")
    d = torch.empty(N, dtype=torch.uint8, device="cuda")
    d.copy_(t, non_blocking=False)
    torch.cuda.synchronize()
    if bytes(d.cpu().numpy()) != PATTERN:
        raise AssertionError("GPU copy mismatch")
    return "pinned host memory: pwritev + GPU consumes it"


def probe_ats_system_memory():
    """Can the GPU address plain malloc'd memory (ATS)? If so the KV pool could
    in principle live in ordinary system memory, which is trivially iovec-able."""
    rt = cudart()
    host = (ctypes.c_ubyte * N).from_buffer_copy(PATTERN)
    src = ctypes.addressof(host)
    dst = ctypes.c_void_p()
    if rt.cudaMalloc(ctypes.byref(dst), ctypes.c_size_t(N)) != 0:
        raise AssertionError("cudaMalloc failed")
    # cudaMemcpyDeviceToDevice = 3: forces the copy engine to treat the plain
    # host pointer as device-addressable, which only works under ATS.
    rc = rt.cudaMemcpy(dst, ctypes.c_void_p(src), ctypes.c_size_t(N), ctypes.c_int(3))
    rt.cudaDeviceSynchronize()
    rt.cudaFree(dst)
    if rc != 0:
        raise AssertionError("DeviceToDevice from system memory rc=%d "
                             "(no ATS for malloc'd memory)" % rc)
    return "GPU addressed plain malloc'd memory directly (ATS active)"


def probe_pluggable_allocator():
    """Does this torch expose CUDAPluggableAllocator? That is the realistic
    mechanism for making vLLM allocate its KV pool as managed memory."""
    import torch
    a = getattr(torch.cuda.memory, "CUDAPluggableAllocator", None)
    if a is None:
        raise AssertionError("torch.cuda.memory.CUDAPluggableAllocator missing")
    has_cumem = hasattr(torch.cuda.memory, "CUDAPluggableAllocator")
    return "CUDAPluggableAllocator present (%s); a managed-memory allocator " \
           "could be installed for the KV pool" % has_cumem


print("=== W28b gate: host-addressable allocations that the GPU can also use ===")
for name, fn in [
    ("cudaMallocManaged is host-addressable + iovec-able + GPU-visible", probe_managed),
    ("O_DIRECT pwritev from managed memory", probe_managed_odirect),
    ("pinned host memory is iovec-able and GPU-consumable", probe_pinned),
    ("GPU can address plain system memory (ATS)", probe_ats_system_memory),
    ("torch exposes CUDAPluggableAllocator", probe_pluggable_allocator),
]:
    ok, msg = in_child(fn)
    report(name, ok, msg.strip())

managed_ok = RESULTS[0][1]
print()
if managed_ok:
    print("VERDICT: a host-addressable GPU-usable allocation EXISTS on this hardware.")
    print("         W28 is not dead on addressability -- it is blocked on making vLLM")
    print("         allocate the KV pool that way, and on whether the attention kernels")
    print("         keep full speed against it. That is the next gate, not this one.")
else:
    print("VERDICT: no host-addressable GPU-usable allocation found; CPU tier is structural.")
sys.exit(0 if managed_ok else 1)
