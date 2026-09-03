#!/usr/bin/env python3
"""W30 probe: what would a STREAMING restore cost?

Today the CPU tier must hold every block of a restore before the single
CPU->GPU load, so tier size == maximum restorable prefix (measured: 9 blocks ->
cached=0, 39 -> 25k tokens, 79 -> 50k). If instead a restore were issued as K
successive load jobs, a small fixed buffer could serve an arbitrarily large
restore -- but each extra job costs at least one engine step of latency.

So the question is the shape of restore cost:

    restore_time(blocks) = a + b * blocks

`a` is what a streamed restore pays K times over; `b` is unavoidable either way.
If a << total, streaming is nearly free and the tier can shrink to megabytes.

Method: restore prefixes of different lengths and regress. Prompt length is
varied so the number of restored chunks varies; the offload transfer time comes
from vLLM's own counters (kv_offload_load_time_total / load_bytes / load_size),
not from wall clock, so it excludes the prefill of the unrestored tail.

Also samples decode inter-token latency, since that is the unit of the per-job
round trip a streamed restore would pay.
"""
import json
import os
import random
import sys
import time
import urllib.request

MODEL = "GLM-5.3-Flash-EXL3"
BASE = os.environ.get("PROBE_BASE", "http://127.0.0.1:8000")
KEY = os.environ.get("VLLM_API_KEY", "")
NFILL = int(os.environ.get("NFILL", "8"))
# doc() yields ~1.29 prompt tokens per unit (measured 12000 -> 15,523 tokens),
# so keep every request under MAX_MODEL_LEN/1.29 or the server returns HTTP 400.
SIZES = [int(x) for x in os.environ.get(
    "SIZES", "3400,6200,9000,11800").split(",")]
FILL_SIZE = int(os.environ.get("FILL_SIZE", "11800"))
W = ["alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge",
     "ionic", "joule", "kelvin", "lumen", "matrix", "nadir", "orbit",
     "prism", "quartz", "rotor", "sigma", "torus", "umbra", "vector"]
Q = "Reply with the single word OK."


def doc(seed, n_tokens):
    r = random.Random(seed)
    return "[doc %d]\n%s\n[end]" % (seed, " ".join(
        "%s%d" % (r.choice(W), r.randint(0, 999))
        for _ in range(int(n_tokens / 2.4))))


def hdr():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def counters():
    """vLLM's own offload counters -- transfer time and volume, not wall clock."""
    req = urllib.request.Request(BASE + "/metrics", headers=hdr())
    o = {"load_time": 0.0, "load_bytes": 0.0, "load_count": 0.0}
    with urllib.request.urlopen(req, timeout=60) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#"):
                continue
            try:
                v = float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
            if line.startswith("vllm:kv_offload_load_time_total"):
                o["load_time"] = v
            elif line.startswith("vllm:kv_offload_load_bytes_total"):
                o["load_bytes"] = v
            elif line.startswith("vllm:kv_offload_load_size_count"):
                o["load_count"] = v
    return o


def ask(text, max_tokens=8, want_itl=False):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": text + "\n\n" + Q}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr())
    t0 = time.perf_counter()
    ttft, usage, stamps = None, {}, []
    try:
        _r = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %s from the server: %s"
                         % (e.code, e.read().decode()[:300]))
    with _r as r:
        for raw in r:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data: "):
                continue
            c = s[6:]
            if c == "[DONE]":
                break
            try:
                o = json.loads(c)
            except json.JSONDecodeError:
                continue
            if o.get("usage"):
                usage = o["usage"]
            for ch in (o.get("choices") or []):
                if (ch.get("delta") or {}).get("content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    stamps.append(now)
    det = usage.get("prompt_tokens_details") or {}
    itl = None
    if want_itl and len(stamps) > 3:
        gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
        gaps.sort()
        itl = gaps[len(gaps) // 2]          # median, robust to scheduling noise
    return ttft, det.get("cached_tokens"), usage.get("prompt_tokens"), itl


def fit(points):
    """Least-squares a + b*x. Returns (a, b, r2)."""
    n = len(points)
    if n < 2:
        return 0.0, 0.0, 0.0
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    if den == 0:
        return sy / n, 0.0, 0.0
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    mean = sy / n
    ss_tot = sum((p[1] - mean) ** 2 for p in points)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in points)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return a, b, r2


if __name__ == "__main__":
    print("=== W30: fixed vs marginal cost of an offload restore ===", flush=True)
    print("--- decode inter-token latency (the per-job round trip unit) ---",
          flush=True)
    _, _, _, itl = ask(doc(555, 2000), max_tokens=40, want_itl=True)
    print("  median ITL: %s" % ("%.1f ms" % (itl * 1000) if itl else "n/a"),
          flush=True)

    rows = []
    for i, size in enumerate(SIZES):
        d = doc(100 + i, size)
        ask(d)                                    # cold: populate + offload
        time.sleep(3)
        for f in range(NFILL):                    # evict
            ask(doc(9000 + i * 100 + f, FILL_SIZE))
        time.sleep(6)
        b = counters()
        ttft, cached, prompt, _ = ask(d)
        time.sleep(4)
        a = counters()
        dt = a["load_time"] - b["load_time"]
        by = a["load_bytes"] - b["load_bytes"]
        cnt = a["load_count"] - b["load_count"]
        print("  prompt=%-6s cached=%-7s restored=%8.1f MiB  jobs=%-3.0f "
              "transfer=%7.2f ms  TTFT=%s"
              % (prompt, cached, by / 2 ** 20, cnt, dt * 1000,
                 "%.2fs" % ttft if ttft else "n/a"), flush=True)
        if by > 0:
            rows.append((by / 2 ** 20, dt * 1000))

    print("--- fit: transfer_ms = a + b * MiB ---", flush=True)
    if len(rows) >= 2:
        a_ms, b_ms, r2 = fit(rows)
        print("  a (fixed per job) = %.3f ms" % a_ms, flush=True)
        print("  b (marginal)      = %.4f ms/MiB  -> %.1f GB/s" % (
            b_ms, (1 / b_ms) * 1000 / 1024 if b_ms > 0 else 0), flush=True)
        print("  r2                = %.3f" % r2, flush=True)
        step = (itl * 1000) if itl else 30.0
        for restore_mib, tier_mib in ((1400, 64), (1400, 256), (350, 64)):
            k = max(1, int(restore_mib / tier_mib + 0.999))
            streamed = k * (a_ms + step) + b_ms * restore_mib
            single = a_ms + b_ms * restore_mib
            print("  restore %4d MiB via a %3d MiB buffer -> %2d jobs: "
                  "%.0f ms vs %.0f ms single-shot (+%.0f ms)"
                  % (restore_mib, tier_mib, k, streamed, single,
                     streamed - single), flush=True)
        print("  (compare: a full cold prefill of that prefix is seconds)",
              flush=True)
    else:
        print("  INCONCLUSIVE: fewer than 2 restores actually happened",
              flush=True)
    print("W30_PROBE_DONE", flush=True)
