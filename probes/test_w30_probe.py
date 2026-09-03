#!/usr/bin/env python3
"""Offline tests for w30_probe.py -- the analysis, not the engine.

The probe's conclusion rests entirely on a least-squares fit and a metrics
parse. Both are testable in milliseconds, and a live iteration costs ~15 minutes.
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


spec = importlib.util.spec_from_file_location("w30", os.path.join(HERE, "w30_probe.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)          # __main__ guard keeps the driver from running

print("=== 1. the linear fit recovers a known line ===")
a, b, r2 = m.fit([(x, 5.0 + 2.0 * x) for x in (1, 2, 3, 4, 10)])
check("recovers intercept", abs(a - 5.0) < 1e-6, "a=%.6f" % a)
check("recovers slope", abs(b - 2.0) < 1e-6, "b=%.6f" % b)
check("r2 == 1 for a perfect line", abs(r2 - 1.0) < 1e-9, "r2=%.6f" % r2)

a, b, r2 = m.fit([(x, 7.0) for x in (1, 2, 3, 4)])
check("flat data -> zero slope, intercept = level",
      abs(b) < 1e-9 and abs(a - 7.0) < 1e-9, "a=%.3f b=%.3g" % (a, b))

a, b, r2 = m.fit([(1, 10.0), (2, 12.0), (3, 13.5), (4, 16.5)])
check("noisy data still gives a sane slope and 0<r2<=1",
      1.0 < b < 3.0 and 0.0 < r2 <= 1.0, "b=%.3f r2=%.3f" % (b, r2))
check("a single point cannot produce a fit", m.fit([(1, 1.0)]) == (0.0, 0.0, 0.0))

print("=== 2. an all-fixed-cost line is distinguishable from an all-marginal one ===")
a_fixed, b_fixed, _ = m.fit([(x, 100.0 + 0.001 * x) for x in (10, 100, 1000)])
a_marg, b_marg, _ = m.fit([(x, 0.5 * x) for x in (10, 100, 1000)])
check("dominant fixed cost shows as large a, ~0 b",
      a_fixed > 50 and b_fixed < 0.01, "a=%.1f b=%.4f" % (a_fixed, b_fixed))
check("dominant marginal cost shows as ~0 a, large b",
      abs(a_marg) < 1e-6 and b_marg > 0.1, "a=%.3g b=%.3f" % (a_marg, b_marg))

print("=== 3. metrics parsing picks the right counters ===")
sample = """# HELP x
vllm:kv_offload_load_time_total{engine="0",model_name="M"} 0.006199
vllm:kv_offload_load_bytes_total{engine="0",model_name="M"} 3.49403e+08
vllm:kv_offload_load_size_count{engine="0",model_name="M"} 2.0
vllm:kv_offload_store_time_total{engine="0",model_name="M"} 9.9
vllm:kv_offload_total_bytes_total{engine="0",model_name="M",transfer_type="GPU_to_CPU"} 7.7e+09
"""


class _R:
    def __init__(self, t):
        self.t = t.encode()

    def read(self):
        return self.t

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


import urllib.request as _u
_u.urlopen = lambda *a, **k: _R(sample)
c = m.counters()
check("load_time parsed", abs(c["load_time"] - 0.006199) < 1e-9, str(c["load_time"]))
check("load_bytes parsed", c["load_bytes"] == 3.49403e+08, str(c["load_bytes"]))
check("load_count parsed", c["load_count"] == 2.0, str(c["load_count"]))
check("store_time NOT mistaken for load_time", c["load_time"] != 9.9)
check("GPU_to_CPU total NOT mistaken for load_bytes", c["load_bytes"] != 7.7e+09)

print("=== 4. the streaming projection is arithmetically right ===")
a_ms, b_ms, step = 2.0, 0.5, 30.0
restore, tier = 1400.0, 64.0
k = max(1, int(restore / tier + 0.999))
streamed = k * (a_ms + step) + b_ms * restore
single = a_ms + b_ms * restore
check("job count rounds up", k == 22, "k=%d" % k)
check("streaming adds only the per-job round trips",
      abs((streamed - single) - ((k - 1) * a_ms + k * step)) < 1e-9,
      "delta=%.1f ms" % (streamed - single))
check("marginal term is identical in both",
      abs((streamed - k * (a_ms + step)) - (single - a_ms)) < 1e-9)

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL W30 PROBE TESTS PASSED")
