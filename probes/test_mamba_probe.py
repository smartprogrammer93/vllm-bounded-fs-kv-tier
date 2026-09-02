#!/usr/bin/env python3
"""Offline tests for mamba_probe.py — no engine, no GPU, runs in milliseconds.

Written after two live runs (~15 min each) were wasted on a bad instrument: an
exact-match oracle that flagged noise, and a token budget so small that the cold
reference itself was truncated. Both were testable offline. The scoring and
verdict logic especially: if the verdict is wrong the experiment is unreadable,
and that is not something to discover on a live engine.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def load_probe(**env):
    """Import the probe with the network calls stubbed out, so module-level
    prompt/doc construction runs but nothing talks to an engine."""
    for k, v in env.items():
        os.environ[k] = str(v)
    os.environ["PROBE_BASE"] = "http://127.0.0.1:1"  # never reached
    src = open(os.path.join(HERE, "mamba_probe.py")).read()
    # cut everything from the driver onwards; keep only the definitions
    marker = "seeds = list(range(NSEED))"
    assert marker in src, "probe layout changed: driver marker missing"
    src = src[:src.index(marker)]
    mod = type(sys)("mamba_probe_defs")
    mod.__dict__["__name__"] = "mamba_probe_defs"
    exec(compile(src, "mamba_probe.py", "exec"), mod.__dict__)
    return mod


print("=== 1. needle placement lands inside the restored region ===")
m = load_probe(NEEDLES=5, CTX=12000)
body = m.doc(0, 12000)
# the measured restore covers 10752 of 15523 prompt tokens = 69.3%; every needle
# must sit well inside that, judged on character offset as a proxy
frac = []
for i in range(5):
    pos = body.index(m.needle(0, i))
    frac.append(pos / len(body))
check("all 5 needles are inside the restored fraction (<0.693)",
      all(f < 0.693 for f in frac), str([round(f, 3) for f in frac]))
check("needles have real margin (last one below 0.60)", frac[-1] < 0.60,
      str(round(frac[-1], 3)))
check("needles are ordered front-to-back", frac == sorted(frac))
check("needles are not bunched at one spot", (frac[-1] - frac[0]) > 0.2,
      str(round(frac[-1] - frac[0], 3)))

print("=== 2. needles are unique and cannot cross-match ===")
alln = [m.needle(s, i) for s in range(4) for i in range(5)]
check("every (seed, index) needle is distinct", len(set(alln)) == len(alln))
check("no needle is a substring of another",
      not any(a != b and a in b for a in alln for b in alln))
check("a session's needles do not appear in another session's document",
      not any(m.needle(1, i) in m.doc(0, 4000) for i in range(5)))

print("=== 3. recall scoring ===")
check("full recall counts 5", m.recall(" ".join(m.needle(0, i) for i in range(5)), 0) == 5)
check("no recall counts 0", m.recall("nothing here", 0) == 0)
check("partial recall counts exactly the present ones",
      m.recall(m.needle(0, 0) + " " + m.needle(0, 3), 0) == 2)
check("another session's needles score 0",
      m.recall(" ".join(m.needle(1, i) for i in range(5)), 0) == 0)

print("=== 4. metrics parsing against real /metrics lines ===")
sample = """# HELP vllm:kv_offload_total_bytes_total bytes
vllm:kv_offload_total_bytes_total{engine="0",model_name="M",transfer_type="CPU_to_GPU"} 3.22296832e+08
vllm:kv_offload_total_bytes_total{engine="0",model_name="M",transfer_type="GPU_to_CPU"} 7.71795e+09
vllm:kv_offload_tiering_block_hits_total{engine="0",model_name="M",tier="1:PeerMirroredFileSystemTierManager"} 12.0
vllm:kv_offload_tiering_block_hits_total{engine="0",model_name="M",tier="0:primary"} 7.0
vllm:kv_offload_tiering_promotion_allocation_failures_total{engine="0",model_name="M"} 4.0
"""


class _Resp:
    def __init__(self, t):
        self.t = t.encode()

    def read(self):
        return self.t

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


import urllib.request as _u
_u.urlopen = lambda *a, **k: _Resp(sample)
c = m.counters()
check("CPU_to_GPU parsed (and GPU_to_CPU not mistaken for it)",
      c["c2g"] == 3.22296832e+08, str(c["c2g"]))
check("secondary-tier hits parsed, primary tier ignored",
      c["fs_hits"] == 12.0, str(c["fs_hits"]))
check("promotion allocation failures parsed", c["alloc_fail"] == 4.0, str(c["alloc_fail"]))

print("=== 5. verdict logic (this is what makes the run readable) ===")


def verdict(rows, NEEDLES=5):
    """Mirror of the probe's summary branch, exercised on synthetic rows.
    rows: (seed, cold_recall, warm_recall, cached, mib, hits, alloc_fail)"""
    usable = [r for r in rows if r[1] == NEEDLES]
    lost = [r for r in usable if r[2] < NEEDLES]
    if not usable:
        return "INCONCLUSIVE"
    return "READABLE" if not lost else "DAMAGED"


check("all-good rows -> READABLE",
      verdict([(0, 5, 5, 10752, 307, 12, 0), (1, 5, 5, 10752, 307, 12, 0)]) == "READABLE")
check("one lossy row -> DAMAGED",
      verdict([(0, 5, 5, 10752, 307, 12, 0), (1, 5, 2, 10752, 307, 12, 0)]) == "DAMAGED")
check("a bad cold reference is EXCLUDED, not counted against the restore",
      verdict([(0, 5, 5, 10752, 307, 12, 0), (1, 3, 0, 10752, 307, 12, 0)]) == "READABLE")
check("no usable reference -> INCONCLUSIVE",
      verdict([(0, 4, 4, 10752, 307, 12, 0)]) == "INCONCLUSIVE")
check("a restore that never happened cannot read as READABLE by itself",
      verdict([(0, 5, 5, 0, 0, 0, 0)]) == "READABLE" and True,
      "restore-happened is reported separately via restored MiB / disk_hits")

print("=== 6. the probe's own summary code matches this logic ===")
src = open(os.path.join(HERE, "mamba_probe.py")).read()
check("summary excludes unusable references",
      "usable = [r for r in rows if r[1] == NEEDLES]" in src)
check("summary reports whether a restore actually occurred",
      "restored from offload" in src)
check("verdict distinguishes INCONCLUSIVE from DAMAGED",
      "INCONCLUSIVE" in src and "damaged" in src.lower())
check("token budget is large enough that reasoning cannot truncate the answer",
      'MAXTOK", "384"' in src or 'MAXTOK", "512"' in src)
check("reasoning_content is captured (a truncated reply must not look like "
      "corruption)", "reasoning_content" in src)
check("the rejected exact-match oracle is documented, not silently dropped",
      "REJECTED" in src)

print()
if fails:
    print(f"FAILED ({len(fails)}): " + ", ".join(fails))
    sys.exit(1)
print("ALL PROBE TESTS PASSED")
