#!/usr/bin/env python3
"""Offline tests for w31_concurrent_probe.py.

A concurrency probe has two failure modes that make its output a lie rather than
a number, and both are offline-detectable:

  * needle strings that collide between sessions, so a cross-session leak reads
    as a clean restore (or vice versa);
  * a thread harness that loses or overwrites results, so a hung request
    silently disappears instead of being counted as the failure it is.

Both are checked here. A live iteration costs 15+ minutes.
"""
import importlib.util
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, cond, detail=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", name,
                           (" -- " + detail) if detail else ""))
    if not cond:
        fails.append(name)


spec = importlib.util.spec_from_file_location(
    "w31", os.path.join(HERE, "w31_concurrent_probe.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print("=== 1. needles are unique across every session pair ===")
allns = [(s, n) for s in range(16) for n in m.needles(s)]
strs = [n for _, n in allns]
check("no duplicates across 16 sessions", len(set(strs)) == len(strs))
sub = [n for n in strs if any(n != o and n in o for o in strs)]
check("no needle is a substring of another", not sub, str(sub[:3]))

print("=== 2. a cross-session leak is detected, a clean restore is not ===")
seeds = list(range(8))
hits, others = m.score(3, ", ".join(m.needles(3)), "", seeds)
check("clean full restore: 5/5, no leak", len(hits) == 5 and not others)
hits, others = m.score(3, ", ".join(m.needles(3)) + " " + m.needles(5)[2], "", seeds)
check("leak from session 5 is flagged", others == [m.needles(5)[2]], str(others))
hits, others = m.score(3, ", ".join(m.needles(3)[:3]), "", seeds)
check("truncated restore scores 3/5, no false leak",
      len(hits) == 3 and not others)

print("=== 3. truncation and leakage are distinguishable ===")
# Both must not be conflated: one is a perf/alignment issue, the other is a bug.
t_hits, t_oth = m.score(1, ", ".join(m.needles(1)[:2]), "", seeds)
l_hits, l_oth = m.score(1, ", ".join(m.needles(1)) + m.needles(2)[0], "", seeds)
check("truncation shows as low recall only", len(t_hits) < 5 and not t_oth)
check("leakage shows as full recall plus a foreign needle",
      len(l_hits) == 5 and len(l_oth) == 1)

print("=== 4. needle depths survive at the sizes this probe uses ===")
for ctx in (120000, 250000):
    d = m.doc(0, ctx)
    ns = m.needles(0)
    present = all(n in d for n in ns)
    pos = [d.index(n) / len(d) for n in ns if n in d]
    ordered = pos == sorted(pos)
    near = all(abs(p - w) < 0.08 for p, w in zip(pos, m.DEPTHS))
    check("ctx=%d: 5 needles, in order, at depth" % ctx,
          present and ordered and near,
          "%d found" % len(pos))

print("=== 5. the thread harness keeps every result, including failures ===")
calls = []


def fake_ask(text, timeout=None):
    # third agent 'hangs' -> the harness must record it as an error, not drop it
    i = len(calls)
    calls.append(i)
    time.sleep(0.05)
    if i == 2:
        return None, None, None, "", "", "timeout: hung restore"
    return 1.0 + i, 100, 200, ", ".join(m.needles(i)), "", None


real = m.ask
m.ask = fake_ask
try:
    sd = list(range(5))
    res, wall = m.fire_concurrently(sd, {s: "doc%d" % s for s in sd})
    check("all %d results present" % len(sd), len(res) == len(sd), str(sorted(res)))
    check("the hung one is recorded as an error, not lost",
          res[2][5] is not None and "timeout" in res[2][5], str(res[2][5]))
    check("requests really overlapped (wall << sum of latencies)",
          wall < 0.05 * len(sd), "wall=%.3fs" % wall)
finally:
    m.ask = real

print("=== 6. report() calls a hung or leaking run a FAILURE ===")
m.ask = fake_ask
try:
    sd = list(range(3))
    calls.clear()
    res, wall = m.fire_concurrently(sd, {s: "d" % () if False else "d" for s in sd})
    z = {"c2g": 0.0, "jobs": 0.0, "disk_hits": 0.0, "alloc_fail": 0.0}
    ok = m.report("t", sd, res, wall, z, dict(z), {s: 9.0 for s in sd})
    check("a run containing a hung request does not pass", ok is False)
finally:
    m.ask = real

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL W31 PROBE TESTS PASSED")
