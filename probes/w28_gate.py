#!/usr/bin/env python3
"""W28 gate: can a CPU process read a CUDA tensor's bytes on GB10, and can those
bytes be handed straight to pwritev/preadv?

If yes, the CPU offload tier can be removed entirely: a KV block is scattered
across per-layer tensors, but pwritev takes an iovec, so the gather can go
directly from the KV pages into one contiguous file with no staging region --
which also removes the restore-window cap (today tier size == max restorable
prefix) and the uniform-row padding.

If no, the CPU tier is structural on this hardware and W28 is dead.

Every probe runs in a FORKED CHILD: dereferencing device memory that is not
host-mapped is a segfault, not an exception, and that must be contained and
reported rather than taking the parent down.
"""
import ctypes
import os
import sys

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok))
    print("  %-4s %s%s" % ("PASS" if ok else "FAIL", name,
                           (" -- " + detail) if detail else ""), flush=True)


def in_child(fn):
    """Run fn() in a forked child. Returns (ok, message).

    A segfault shows up as a signal in the exit status, which is exactly the
    answer we are looking for, so it must not be allowed to kill the parent.
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            msg = fn() or "ok"
            os.write(w, ("OK " + str(msg)).encode()[:4000])
            os._exit(0)
        except BaseException as e:
            os.write(w, ("ERR %s: %s" % (type(e).__name__, e)).encode()[:4000])
            os._exit(1)
    os.close(w)
    out = b""
    while True:
        chunk = os.read(r, 4096)
        if not chunk:
            break
        out += chunk
    os.close(r)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        return False, "child died on signal %d (memory not host-dereferenceable)" % os.WTERMSIG(status)
    return os.WEXITSTATUS(status) == 0, out.decode() or "no output"


def probe_env():
    import torch
    p = torch.cuda.get_device_properties(0)
    return "torch %s | %s | cc %d.%d | unified=%s" % (
        torch.__version__, p.name, p.major, p.minor,
        getattr(p, "is_integrated", "?"))


def probe_host_read():
    """Write a known pattern on the GPU, then read it from the CPU by address."""
    import torch
    n = 4096
    t = torch.arange(n, dtype=torch.uint8, device="cuda") % 251
    torch.cuda.synchronize()
    expect = bytes((i % 251) for i in range(n))
    buf = (ctypes.c_ubyte * n).from_address(t.data_ptr())
    got = bytes(buf)
    if got != expect:
        first = next((i for i in range(n) if got[i] != expect[i]), -1)
        raise AssertionError("readable but WRONG at byte %d (%r vs %r)"
                             % (first, got[first:first + 8], expect[first:first + 8]))
    return "read %d bytes at 0x%x, contents match" % (n, t.data_ptr())


def probe_pwritev():
    """Gather several scattered device ranges into one file in one syscall --
    this is the actual W28 store primitive."""
    import torch
    n, frags = 4096, 4
    t = torch.arange(n, dtype=torch.uint8, device="cuda") % 251
    torch.cuda.synchronize()
    base = t.data_ptr()
    step = n // frags
    iov = [memoryview((ctypes.c_ubyte * step).from_address(base + i * step))
           for i in range(frags)]
    path = "/tmp/w28_gate_test.bin"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        wrote = os.pwritev(fd, iov, 0)
    finally:
        os.close(fd)
    data = open(path, "rb").read()
    os.unlink(path)
    expect = bytes((i % 251) for i in range(n))
    if wrote != n or data != expect:
        raise AssertionError("pwritev wrote %d/%d, content match=%s"
                             % (wrote, n, data == expect))
    return "pwritev gathered %d fragments (%d bytes) straight from device memory" % (frags, n)


def probe_preadv():
    """Scatter a file back into device memory -- the W28 restore primitive."""
    import torch
    n, frags = 4096, 4
    t = torch.zeros(n, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    payload = bytes((i % 251) for i in range(n))
    path = "/tmp/w28_gate_test2.bin"
    with open(path, "wb") as f:
        f.write(payload)
    base = t.data_ptr()
    step = n // frags
    iov = [memoryview((ctypes.c_ubyte * step).from_address(base + i * step))
           for i in range(frags)]
    fd = os.open(path, os.O_RDONLY)
    try:
        read = os.preadv(fd, iov, 0)
    finally:
        os.close(fd)
        os.unlink(path)
    torch.cuda.synchronize()
    back = bytes(t.cpu().numpy())
    if read != n or back != payload:
        raise AssertionError("preadv read %d/%d, GPU-side match=%s"
                             % (read, n, back == payload))
    return "preadv scattered %d bytes into device memory, GPU sees it" % n


def probe_odirect():
    """O_DIRECT needs an aligned buffer; device pointers are page-aligned in
    practice, but the offload path uses O_DIRECT when the filesystem allows it,
    so confirm it works rather than assuming."""
    import torch
    n = 1 << 20
    t = torch.arange(n, dtype=torch.uint8, device="cuda") % 251
    torch.cuda.synchronize()
    ptr = t.data_ptr()
    if ptr % 4096:
        return "SKIP: device pointer 0x%x is not 4096-aligned" % ptr
    path = "/tmp/w28_gate_odirect.bin"
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_DIRECT", 0)
    fd = os.open(path, flags, 0o600)
    try:
        wrote = os.pwritev(fd, [memoryview((ctypes.c_ubyte * n).from_address(ptr))], 0)
    finally:
        os.close(fd)
    ok = wrote == n and open(path, "rb").read() == bytes((i % 251) for i in range(n))
    os.unlink(path)
    if not ok:
        raise AssertionError("O_DIRECT pwritev wrote %d/%d or content mismatch" % (wrote, n))
    return "O_DIRECT pwritev of %d bytes straight from device memory" % n


print("=== W28 gate: is device memory host-addressable for scatter/gather I/O? ===")
for name, fn in [("environment", probe_env),
                 ("CPU can read CUDA tensor bytes by address", probe_host_read),
                 ("pwritev gathers device memory into a file", probe_pwritev),
                 ("preadv scatters a file into device memory", probe_preadv),
                 ("O_DIRECT pwritev from device memory", probe_odirect)]:
    ok, msg = in_child(fn)
    report(name, ok, msg.strip())

core = [ok for n, ok in RESULTS[1:4]]
print()
if all(core):
    print("VERDICT: W28 IS VIABLE -- KV can go straight to disk, no CPU tier.")
elif not RESULTS[1][1]:
    print("VERDICT: W28 IS DEAD -- device memory is not host-dereferenceable here; "
          "the CPU staging tier is structural.")
else:
    print("VERDICT: PARTIAL -- host reads work but the syscall path does not; "
          "a userspace copy would still be needed.")
sys.exit(0 if all(core) else 1)
