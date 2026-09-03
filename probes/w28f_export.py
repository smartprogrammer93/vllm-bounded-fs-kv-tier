#!/usr/bin/env python3
"""W28f: can we get a CPU-READABLE VIEW of device memory, for streaming only?

This is a different proposal from W28. W28 tried to make the KV POOL itself
host-addressable, which destroyed attention: every host-mapped allocation is
reached by the GPU through host page tables, and a scattered gather ran ~150x
slower. But offloading KV to disk is a SEQUENTIAL STREAM, and streaming was never
the problem (0.91x).

So: keep the pool as ordinary device memory (attention unaffected, full 190+
GB/s gathers) and obtain a second, CPU-readable mapping of the same physical
pages used ONLY for the bulk copy out. If that exists, the CPU staging tier and
its whole copy disappear, and none of the gather penalty applies.

cuMemSetAccess with a HOST location was already refused (CUDA_ERROR_NOT_SUPPORTED).
Untested export routes, both of which hand back a file descriptor:

  cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)
      -> dma-buf fd for an arbitrary device range (used by GDS/RDMA).
         If the exporter implements mmap, the CPU can read it.
  cuMemExportToShareableHandle(..., CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
      -> fd for a VMM allocation, intended for IPC import. Try mmap/pread anyway.

A dma-buf fd would also be directly usable by O_DIRECT / sendfile paths even if
mmap is refused, so the fd itself is worth having.
"""
import ctypes
import ctypes.util
import mmap as _mmap
import os
import sys

CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

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
        return False, "child died on signal %d" % os.WTERMSIG(st)
    return os.WEXITSTATUS(st) == 0, out.decode() or "no output"


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAllocationProp(ctypes.Structure):
    class _F(ctypes.Structure):
        _fields_ = [("compressionType", ctypes.c_ubyte),
                    ("gpuDirectRDMACapable", ctypes.c_ubyte),
                    ("usage", ctypes.c_ushort),
                    ("reserved", ctypes.c_ubyte * 4)]
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("location", CUmemLocation), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", _F)]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


def drv():
    for n in ("libcuda.so", "libcuda.so.1"):
        try:
            return ctypes.CDLL(n)
        except OSError:
            pass
    return ctypes.CDLL(ctypes.util.find_library("cuda"))


def errname(d, rc):
    s = ctypes.c_char_p()
    try:
        d.cuGetErrorName(ctypes.c_int(rc), ctypes.byref(s))
        return s.value.decode() if s.value else str(rc)
    except Exception:
        return str(rc)


def chk(d, rc, what):
    if rc != 0:
        raise RuntimeError("%s -> %s" % (what, errname(d, rc)))


PATTERN = bytes((i * 31 + 11) % 251 for i in range(1 << 16))


def probe_dmabuf_from_torch_tensor():
    """The realistic case: a device range allocated by torch (as the KV pool is),
    exported as a dma-buf fd, then read from the CPU."""
    import torch
    d = drv()
    t = torch.frombuffer(bytearray(PATTERN), dtype=torch.uint8).cuda()
    torch.cuda.synchronize()
    size = t.numel()
    fd = ctypes.c_int(-1)
    rc = d.cuMemGetHandleForAddressRange(
        ctypes.byref(fd), ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(size),
        ctypes.c_int(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD), ctypes.c_ulonglong(0))
    if rc != 0:
        raise RuntimeError("cuMemGetHandleForAddressRange -> %s" % errname(d, rc))
    if fd.value < 0:
        raise RuntimeError("no fd returned")
    # Got a dma-buf fd. Can the CPU map it?
    try:
        mm = _mmap.mmap(fd.value, size, prot=_mmap.PROT_READ)
    except OSError as e:
        os.close(fd.value)
        raise RuntimeError("got dma-buf fd %d but mmap failed: %s "
                           "(fd may still be usable for O_DIRECT/sendfile)"
                           % (fd.value, e))
    got = mm[:64]
    mm.close()
    os.close(fd.value)
    if got != PATTERN[:64]:
        raise AssertionError("mapped but contents wrong")
    return "dma-buf fd mmap'd and readable -- device memory is CPU-visible for streaming"


def probe_dmabuf_vmm():
    """Same, for a VMM allocation rather than a torch one."""
    d = drv()
    chk(d, d.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    chk(d, d.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    chk(d, d.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "ctxRetain")
    chk(d, d.cuCtxSetCurrent(ctx), "ctxSetCurrent")
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev.value
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    gran = ctypes.c_size_t()
    chk(d, d.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0),
        "granularity")
    size = gran.value
    h = ctypes.c_ulonglong()
    chk(d, d.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(size), ctypes.byref(prop), 0),
        "cuMemCreate(exportable)")
    ptr = ctypes.c_void_p()
    chk(d, d.cuMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(size),
                                 ctypes.c_size_t(0), ctypes.c_void_p(0),
                                 ctypes.c_ulonglong(0)), "reserve")
    chk(d, d.cuMemMap(ptr, ctypes.c_size_t(size), ctypes.c_size_t(0), h,
                      ctypes.c_ulonglong(0)), "cuMemMap")
    desc = CUmemAccessDesc()
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = dev.value
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    chk(d, d.cuMemSetAccess(ptr, ctypes.c_size_t(size), ctypes.byref(desc),
                            ctypes.c_size_t(1)), "cuMemSetAccess(device)")
    fd = ctypes.c_int(-1)
    rc = d.cuMemGetHandleForAddressRange(
        ctypes.byref(fd), ptr, ctypes.c_size_t(size),
        ctypes.c_int(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD), ctypes.c_ulonglong(0))
    if rc != 0:
        raise RuntimeError("dma-buf export of VMM range -> %s" % errname(d, rc))
    try:
        mm = _mmap.mmap(fd.value, size, prot=_mmap.PROT_READ)
        mm.close()
    except OSError as e:
        os.close(fd.value)
        raise RuntimeError("dma-buf fd %d obtained, mmap failed: %s" % (fd.value, e))
    os.close(fd.value)
    return "VMM range exported as dma-buf and mmap'd (size %d)" % size


def probe_shareable_handle_mmap():
    """cuMemExportToShareableHandle gives an IPC fd; see whether it is mmap-able
    or readable, which would be an equally good stream source."""
    d = drv()
    chk(d, d.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    chk(d, d.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    chk(d, d.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "ctxRetain")
    chk(d, d.cuCtxSetCurrent(ctx), "ctxSetCurrent")
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev.value
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    gran = ctypes.c_size_t()
    chk(d, d.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0),
        "granularity")
    size = gran.value
    h = ctypes.c_ulonglong()
    chk(d, d.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(size), ctypes.byref(prop), 0),
        "cuMemCreate")
    fd = ctypes.c_int(-1)
    chk(d, d.cuMemExportToShareableHandle(
        ctypes.byref(fd), h, ctypes.c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
        ctypes.c_ulonglong(0)), "cuMemExportToShareableHandle")
    try:
        mm = _mmap.mmap(fd.value, size, prot=_mmap.PROT_READ)
        mm.close()
        os.close(fd.value)
        return "shareable-handle fd is mmap-able (size %d)" % size
    except OSError as e:
        try:
            data = os.pread(fd.value, 64, 0)
            os.close(fd.value)
            return "not mmap-able (%s) but pread returned %d bytes" % (e, len(data))
        except OSError as e2:
            os.close(fd.value)
            raise RuntimeError("fd obtained but neither mmap (%s) nor pread (%s) works"
                               % (e, e2))


print("=== W28f: a CPU-readable VIEW of device memory, for streaming only ===")
_RAW = ""
for name, fn in [
    ("torch device tensor -> dma-buf fd -> CPU mmap", probe_dmabuf_from_torch_tensor),
    ("VMM device range -> dma-buf fd -> CPU mmap", probe_dmabuf_vmm),
    ("VMM shareable-handle fd is mmap/pread-able", probe_shareable_handle_mmap),
]:
    ok, msg = in_child(fn)
    _RAW += msg
    report(name, ok, msg.strip())

print()
# An OOM means the probe never ran; it must not be reported as a refusal.
_details = [d for _, d in [(n, o) for n, o in RESULTS]]
_oom = "OUT_OF_MEMORY" in _RAW or "out of memory" in _RAW
if _oom and not any(ok for _, ok in RESULTS):
    print("VERDICT: INCONCLUSIVE -- every probe failed with CUDA OOM, so none of")
    print("         them actually ran. Free device memory and re-run.")
elif any(ok for _, ok in RESULTS):
    print("VERDICT: a CPU-readable view of DEVICE memory exists. The pool can stay")
    print("         device-resident (attention unaffected) while offload streams")
    print("         raw bytes straight out of it -- no CPU staging tier.")
else:
    print("VERDICT: no CPU-readable view of device memory. Every export route is")
    print("         refused, so the staging copy is the only way to get the bytes.")
sys.exit(0)
