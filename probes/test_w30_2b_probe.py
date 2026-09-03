#!/usr/bin/env python3
"""Offline tests for w30_2b_probe.py -- the instrument, not the engine.

The probe's whole claim rests on needle placement and scoring. If needles are
misplaced, collide between sessions, or the document overflows MAX_MODEL_LEN,
the live run wastes ~15 minutes and produces a conclusion that is wrong in a
direction that is hard to see. All of it is checkable offline in milliseconds.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, cond, detail=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", name,
                           (" -- " + detail) if detail else ""))
    if not cond:
        fails.append(name)


spec = importlib.util.spec_from_file_location(
    "p", os.path.join(HERE, "w30_2b_probe.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print("=== 1. every needle is actually in the document, in order ===")
d = m.doc(0, 11000)
ns = m.needles(0)
check("all 5 needles present", all(n in d for n in ns),
      str([n for n in ns if n not in d]))
pos = [d.index(n) for n in ns if n in d]
check("needles appear in depth order", pos == sorted(pos), str(pos))

print("=== 2. needles sit near their intended depths ===")
# A needle that drifts to the wrong depth would not distinguish a partial
# restore from a full one, which is this probe's entire purpose.
for i, want in enumerate(m.DEPTHS):
    got = d.index(ns[i]) / len(d)
    check("needle %d near %.0f%% depth" % (i, want * 100),
          abs(got - want) < 0.08, "at %.1f%%" % (got * 100))

print("=== 3. sessions cannot be confused with each other ===")
allns = [n for s in range(6) for n in m.needles(s)]
check("no duplicate needle strings across sessions",
      len(set(allns)) == len(allns))
bad = [(a, b) for a in allns for b in allns if a != b and a in b]
check("no needle is a substring of another", not bad, str(bad[:3]))

print("=== 4. contamination is detected, and only when real ===")
seeds = [0, 1, 2]
hits, others = m.score(0, ", ".join(m.needles(0)), "", seeds)
check("clean full recall scores 5/5 with no contamination",
      len(hits) == 5 and not others, "%d/%s" % (len(hits), others))
hits, others = m.score(0, ", ".join(m.needles(0)[:2]), "", seeds)
check("a partial restore scores below 5/5", len(hits) == 2, str(len(hits)))
hits, others = m.score(0, m.needles(1)[0], "", seeds)
check("another session's needle is flagged", others == [m.needles(1)[0]],
      str(others))
hits, others = m.score(0, "", "".join(m.needles(0)), seeds)
check("needles recalled in reasoning_content still count", len(hits) == 5)

print("=== 5. the document fits MAX_MODEL_LEN ===")
# doc() measured at ~1.29 prompt tokens per unit of CTX (12000 -> 15,523), and
# the test config runs MAX_MODEL_LEN=16384. Overflow returns HTTP 400 and the
# probe aborts several minutes in.
est = m.CTX * 1.29 + 120
check("estimated prompt tokens under 16384", est < 16384, "~%.0f tokens" % est)
check("default CTX leaves >5%% headroom", est < 16384 * 0.95,
      "~%.0f of 15565" % est)

print("=== 6. degenerate output is not scored as success ===")
hits, others = m.score(0, "", "", seeds)
check("empty answer scores 0/5", not hits and not others)

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL W30-2B PROBE TESTS PASSED")
