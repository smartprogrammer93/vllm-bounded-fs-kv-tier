#!/usr/bin/env python3
"""Assemble vllm_bounded_fs_tier_peer.py = pristine bounded tier + peer cascade.

The cascade lives in peer_mirror_tier.py as a quoted source block so the file
stays a normal Python module; extracting it as raw text means escape sequences
must be repaired here (a literal backslash-n once shipped as the wire framing
and blocked the agent's readline for a whole engine boot).
"""
import ast
import sys

REPS = [
    ('logger.info("peer-cascade: connected to rank %d at %s", self.rank, self.addr)',
     '_plog("info", "connected to rank %d at %s", self.rank, self.addr)'),
    ('''logger.info("peer-cascade: rank %d attached %s", self.rank,
                        resp.get("mmap"))''',
     '_plog("info", "rank %d attached %s", self.rank, resp.get("mmap"))'),
    ('logger.info("peer-cascade: no peers configured; single-node behaviour")',
     '_plog("info", "no peers configured; single-node behaviour")'),
    ('''logger.warning(
                        "peer-cascade: rank %d %s job %s FAILED: %s",
                        peer.rank, op, job_id, resp.get("err"))''',
     '''_plog("warning", "rank %d %s job %s FAILED: %s",
                          peer.rank, op, job_id, resp.get("err"))'''),
    ('''logger.info(
            "peer-cascade: %d peer rank(s) %s; basename=%s block_size=%d "
            "region=%d bytes",
            len(self._peers), [p.addr for p in self._peers], hello["basename"],
            hello["block_size"], hello["total_size"])''',
     '''_plog("info",
              "%d peer rank(s) %s; basename=%s block_size=%d region=%d bytes "
              "cap=%.2f GiB local_slot_page=%s",
              len(self._peers), [p.addr for p in self._peers],
              hello["basename"], hello["block_size"], hello["total_size"],
              hello["max_bytes"] / 1024 ** 3, hello["page_size"] or "off")'''),
]

SHIM = '''

_pclog = init_logger("vllm.peer_cascade")


def _plog(level, msg, *args):
    """Log under the vllm.* hierarchy AND to stderr.

    A logger named after this module sits outside vLLM's configured logger tree
    and every record is dropped, which hid a wire-protocol bug for a whole
    engine boot. Never let a cascade failure be invisible.
    """
    try:
        getattr(_pclog, level)(msg, *args)
    except Exception:
        pass
    try:
        print("[peer-cascade] " + (msg % args if args else msg),
              file=sys.stderr, flush=True)
    except Exception:
        pass
'''

PRE_SUPER = ("_peers", "_pool_exec", "_pending", "_held", "_peer_lock",
             "_load_keys_mirror", "_on_disk_block_bytes")


def main() -> int:
    src = open("bounded_pristine.py").read()
    peer = open("peer_mirror_tier.py").read()
    start = peer.index("PEER_MIRROR_SRC = '''") + len("PEER_MIRROR_SRC = '''")
    block = peer[start:peer.rindex("'''")]

    n = block.count('"\\\\n"')
    block = block.replace('"\\\\n"', '"\\n"')
    assert n == 1 and '"\\\\n"' not in block, f"wire framing: {n}"
    block = block.replace('int(len(self._primary_kv_view))',
                          'int(self._primary_kv_view.nbytes)')
    block = block.replace('    @override\n', '')
    block = SHIM + block
    for a, b in REPS:
        assert a in block, a[:60]
        block = block.replace(a, b)

    out = src.rstrip('\n') + '\n' + block
    for need in ('\nimport json\n', '\nimport threading\n'):
        if need not in out:
            out = out.replace('\nimport os\n', need + 'import os\n', 1)
    if '\nimport sys\n' not in out:
        out = out.replace('\nimport threading\n', '\nimport sys\nimport threading\n', 1)

    ast.parse(out)
    cls = out[out.index('class PeerMirroredFileSystemTierManager'):]
    init = cls[cls.index('def __init__'):]
    init = init[:init.index('\n    def ', 5)]
    i_super = init.index('super().__init__(*args, **kwargs)')
    for a in PRE_SUPER:
        assert init.index('self.' + a) < i_super, f"{a} must precede super()"
    assert 'or self._block_size' in cls, "_used_bytes needs its sentinel fallback"
    assert 'GLM53_OFFLOAD_LOCAL_SLOT' in cls, "local slot knob missing"

    open("vllm_bounded_fs_tier_peer.py", "w").write(out)
    print(f"built {len(out)} bytes; {len(PRE_SUPER)} pre-super attrs verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
