#!/usr/bin/env python3
"""W28d gate: can DEVICE memory be made host-addressable without paying the
managed-memory penalty?

Context: cudaMalloc memory is not host-dereferenceable here (W28a), while
cudaMallocManaged is, at ~0.91x bandwidth (W28c). But managed is not "cudaMalloc
plus extras" -- they are sibling driver paths with different MAPPING POLICY. A
third path exists: the VMM driver API allocates a physical handle and then lets
you grant access per location.

    cuMemCreate        -> physical allocation (location = DEVICE)
    cuMemAddressReserve-> VA range
    cuMemMap           -> bind VA to the handle
    cuMemSetAccess     -> grant RW to a list of locations

If CU_MEM_LOCATION_TYPE_HOST is accepted as an access location for a DEVICE
allocation, we get device-resident memory that the CPU can also address -- full
bandwidth AND iovec-able, which would beat managed memory outright.

This matters because torch already uses this API for
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, so a KV pool could plausibly be
allocated this way without a bespoke allocator.

Each probe runs in a forked child: a wrong mapping is a segfault, not an
exception.
"""
import ctypes
import ctypes.util
import os
import sys

# --- CUDA driver API constants ------------------------------------------------
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_LOCATION_TYPE_HOST = 2
CU_MEM_LOCATION_TYPE_HOST_NUMA = 3
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAllocationProp(ctypes.Structure):
    class _AllocFlags(ctypes.Structure):
        _fields_ = [("compressionType", ctypes.c_ubyte),
                    ("gpuDirectRDMACapable", ctypes.c_ubyte),
                    ("usage", ctypes.c_ushort),
                    ("reserved", ctypes.c_ubyte * 4)]
    _fields_ = [("type", ctypes.c_int),
                ("requestedHandleTypes", ctypes.c_int),
                ("location", CUmemLocation),
                ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", _AllocFlags)]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


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


def drv():
    for n in ("libcuda.so", "libcuda.so.1"):
        try:
            return ctypes.CDLL(n)
        except OSError:
            pass
    p = ctypes.util.find_library("cuda")
    if p:
        return ctypes.CDLL(p)
    raise RuntimeError("libcuda not found")


def err(d, rc, what):
    if rc == 0:
        return
    s = ctypes.c_char_p()
    try:
        d.cuGetErrorName(ctypes.c_int(rc), ctypes.byref(s))
        name = s.value.decode() if s.value else "?"
    except Exception:
        name = "?"
    raise RuntimeError("%s -> %d (%s)" % (what, rc, name))


def _init(d):
    err(d, d.cuInit(ctypes.c_uint(0)), "cuInit")
    dev = ctypes.c_int()
    err(d, d.cuDeviceGet(ctypes.byref(dev), ctypes.c_int(0)), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    # retain the primary context so this coexists with anything else running
    err(d, d.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    err(d, d.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    return d, dev.value


def _granularity(d, dev, loc_type):
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = loc_type
    prop.location.id = dev if loc_type == CU_MEM_LOCATION_TYPE_DEVICE else 0
    gran = ctypes.c_size_t()
    # CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
    err(d, d.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop),
                                           ctypes.c_int(0)),
        "cuMemGetAllocationGranularity")
    return prop, gran.value


def _map_and_grant(d, dev, prop, size, access_locations):
    handle = ctypes.c_ulonglong()
    err(d, d.cuMemCreate(ctypes.byref(handle), ctypes.c_size_t(size),
                         ctypes.byref(prop), ctypes.c_ulonglong(0)), "cuMemCreate")
    ptr = ctypes.c_void_p()
    err(d, d.cuMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(size),
                                 ctypes.c_size_t(0), ctypes.c_void_p(0),
                                 ctypes.c_ulonglong(0)), "cuMemAddressReserve")
    err(d, d.cuMemMap(ptr, ctypes.c_size_t(size), ctypes.c_size_t(0),
                      handle, ctypes.c_ulonglong(0)), "cuMemMap")
    n = len(access_locations)
    descs = (CUmemAccessDesc * n)()
    for i, (t, i_id) in enumerate(access_locations):
        descs[i].location.type = t
        descs[i].location.id = i_id
        descs[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    err(d, d.cuMemSetAccess(ptr, ctypes.c_size_t(size), descs, ctypes.c_size_t(n)),
        "cuMemSetAccess%s" % ([t for t, _ in access_locations],))
    return handle, ptr


def probe_vmm_device_only():
    """Baseline: VMM device allocation with DEVICE access only. Expected to be
    allocatable, and NOT host-readable."""
    d, dev = _init(drv())
    prop, gran = _granularity(d, dev, CU_MEM_LOCATION_TYPE_DEVICE)
    size = max(gran, 1 << 21)
    size = ((size + gran - 1) // gran) * gran
    _, ptr = _map_and_grant(d, dev, prop, size,
                            [(CU_MEM_LOCATION_TYPE_DEVICE, dev)])
    return "allocated+mapped %d bytes (granularity %d) at 0x%x" % (size, gran, ptr.value)


def probe_vmm_host_access():
    """THE question: grant the HOST access to a DEVICE allocation, then read it
    from the CPU and hand it to pwritev."""
    d, dev = _init(drv())
    prop, gran = _granularity(d, dev, CU_MEM_LOCATION_TYPE_DEVICE)
    size = max(gran, 1 << 21)
    size = ((size + gran - 1) // gran) * gran
    _, ptr = _map_and_grant(
        d, dev, prop, size,
        [(CU_MEM_LOCATION_TYPE_DEVICE, dev), (CU_MEM_LOCATION_TYPE_HOST, 0)])
    # CPU write then CPU read-back
    pattern = bytes((i * 13 + 7) % 251 for i in range(4096))
    ctypes.memmove(ptr, pattern, len(pattern))
    got = bytes((ctypes.c_ubyte * len(pattern)).from_address(ptr.value))
    if got != pattern:
        raise AssertionError("host read-back mismatch")
    path = "/tmp/w28d_vmm.bin"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        n = os.pwritev(fd, [memoryview((ctypes.c_ubyte * len(pattern))
                                       .from_address(ptr.value))], 0)
    finally:
        os.close(fd)
    data = open(path, "rb").read()
    os.unlink(path)
    if n != len(pattern) or data != pattern:
        raise AssertionError("pwritev mismatch")
    return ("DEVICE memory with HOST access: CPU read/write AND pwritev work "
            "(%d bytes, granularity %d)" % (size, gran))


def probe_vmm_host_numa():
    """Some drivers expose host access only via the HOST_NUMA location type."""
    d, dev = _init(drv())
    prop, gran = _granularity(d, dev, CU_MEM_LOCATION_TYPE_DEVICE)
    size = max(gran, 1 << 21)
    size = ((size + gran - 1) // gran) * gran
    _, ptr = _map_and_grant(
        d, dev, prop, size,
        [(CU_MEM_LOCATION_TYPE_DEVICE, dev), (CU_MEM_LOCATION_TYPE_HOST_NUMA, 0)])
    ctypes.memmove(ptr, b"\xa5" * 256, 256)
    got = bytes((ctypes.c_ubyte * 8).from_address(ptr.value))
    if got != b"\xa5" * 8:
        raise AssertionError("host read-back mismatch")
    return "HOST_NUMA access on a DEVICE allocation works"


def probe_torch_expandable():
    """torch's expandable_segments uses this same VMM API. If a tensor allocated
    that way were host-addressable, no custom allocator would be needed at all."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    t = torch.arange(4096, dtype=torch.uint8, device="cuda") % 251
    torch.cuda.synchronize()
    buf = (ctypes.c_ubyte * 4096).from_address(t.data_ptr())
    got = bytes(buf)
    if got != bytes((i % 251) for i in range(4096)):
        raise AssertionError("readable but wrong")
    return "expandable_segments tensor is host-readable"


print("=== W28d: can DEVICE memory be granted host access via the VMM API? ===")
for name, fn in [
    ("VMM device allocation works at all", probe_vmm_device_only),
    ("DEVICE alloc + HOST access -> CPU read + pwritev", probe_vmm_host_access),
    ("DEVICE alloc + HOST_NUMA access", probe_vmm_host_numa),
    ("torch expandable_segments tensor is host-readable", probe_torch_expandable),
]:
    ok, msg = in_child(fn)
    report(name, ok, msg.strip())

print()
if RESULTS[1][1] or RESULTS[2][1]:
    print("VERDICT: device memory CAN be host-addressable -- full bandwidth AND")
    print("         iovec-able, which beats managed memory (0.91x). Use this.")
else:
    print("VERDICT: the driver refuses host access to device allocations here;")
    print("         cudaMallocManaged (0.91x) remains the only host-addressable route.")
sys.exit(0)
