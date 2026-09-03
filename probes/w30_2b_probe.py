#!/usr/bin/env python3
"""W30 step 2b: is restore size still bounded by the CPU tier?

Control, measured earlier this session with the eager (unstreamed) path:
a ~9-block primary tier restored NOTHING -- cached_tokens was 0, because
TieringOffloadingManager.lookup() promotes during the prefix scan and returns
MISS once the tier is full, truncating the scan to zero.

So with the tier set to 256 MiB (~9 blocks of 25.91 MiB), any cached_tokens > 0
on a revisit is already proof that tier size no longer caps the restore.

Five needles per document, at 10/30/50/70/90% depth. That is the point of this
probe rather than the single-needle one: a streamed restore that lands only its
first batches would recall the early needles and silently drop the late ones,
which a needle at one depth cannot distinguish from a full restore. Recall must
be 5/5, and a needle from another session must never appear.

Reports the offload job count per restore (load_size_count), which is how the
batching itself is observed: >1 job per restore means the chain ran.
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
CTX = int(os.environ.get("CTX", "11000"))
NSEED = int(os.environ.get("NSEED", "3"))
NFILL = int(os.environ.get("NFILL", "5"))
TRIALS = int(os.environ.get("TRIALS", "2"))
DEPTHS = (0.10, 0.30, 0.50, 0.70, 0.90)
W = ["alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge",
     "ionic", "joule", "kelvin", "lumen", "matrix", "nadir", "orbit",
     "prism", "quartz", "rotor", "sigma", "torus", "umbra", "vector"]
Q = ("List the five access codes recorded in this document, in the order they "
     "appear, separated by commas. Output only the codes.")


def needles(seed):
    return ["ZULU-%d%02d-QX" % (seed, i) for i in range(len(DEPTHS))]


def doc(seed, n_tokens):
    r = random.Random(seed)
    words = ["%s%d" % (r.choice(W), r.randint(0, 999))
             for _ in range(int(n_tokens / 2.4))]
    ns = needles(seed)
    # insert from the back so earlier indices stay valid
    for i in reversed(range(len(DEPTHS))):
        at = int(len(words) * DEPTHS[i])
        words.insert(at, "Access code %d is %s." % (i, ns[i]))
    return "[document %d]\n%s\n[end document]" % (seed, " ".join(words))


def hdr():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def counters():
    """Offload counters straight from vLLM, not wall clock."""
    req = urllib.request.Request(BASE + "/metrics", headers=hdr())
    o = {"c2g": 0.0, "jobs": 0.0, "load_time": 0.0, "disk_hits": 0.0,
         "disk_bytes": 0.0, "promo_fail": 0.0}
    with urllib.request.urlopen(req, timeout=60) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#"):
                continue
            try:
                v = float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
            if line.startswith("vllm:kv_offload_total_bytes_total") \
                    and "CPU_to_GPU" in line:
                o["c2g"] = v
            elif line.startswith("vllm:kv_offload_load_size_count"):
                o["jobs"] = v
            elif line.startswith("vllm:kv_offload_load_time_total"):
                o["load_time"] = v
            elif line.startswith("vllm:kv_offload_tiering_block_hits_total"):
                # Secondary tiers are labelled tier="<idx>:<ClassName>"; the
                # primary is tier="0:primary". Matching on tier="0" reported
                # zero disk hits for a run that actually took 155 of them.
                if 'tier="0:primary"' not in line:
                    o["disk_hits"] += v
            elif line.startswith("vllm:kv_offload_tiering_read_bytes_total"):
                o["disk_bytes"] = v
            elif line.startswith("vllm:kv_offload_allocation_failure_total"):
                o["promo_fail"] = v
    return o


def ask(text, label, quiet=False):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": text + "\n\n" + Q}],
            "temperature": 0, "max_tokens": 384, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr())
    t0 = time.perf_counter()
    ttft, usage, out, think = None, {}, [], []
    try:
        _r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %s: %s" % (e.code, e.read().decode()[:300]))
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
                d = ch.get("delta") or {}
                if d.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    out.append(d["content"])
                if d.get("reasoning_content"):
                    think.append(d["reasoning_content"])
    det = usage.get("prompt_tokens_details") or {}
    ans = "".join(out)
    if not quiet:
        print("  %-16s TTFT=%7s prompt=%-6s cached=%-6s" % (
            label, ("%.2fs" % ttft) if ttft else "n/a",
            usage.get("prompt_tokens"), det.get("cached_tokens")), flush=True)
    return ttft, det.get("cached_tokens"), ans, "".join(think)


def score(seed, answer, thinking, seeds):
    """Needle recall for `seed`, plus contamination from any other session."""
    hay = answer + "\n" + thinking
    mine = needles(seed)
    hits = [n for n in mine if n in hay]
    others = [n for o in seeds if o != seed for n in needles(o) if n in hay]
    return hits, others


if __name__ == "__main__":
    seeds = list(range(NSEED))
    docs = {s: doc(s, CTX) for s in seeds}
    print("=== W30 2b: %d sessions x %d tokens, %d fillers, %d trial(s) ==="
          % (NSEED, CTX, NFILL, TRIALS), flush=True)
    print("    control: a ~9-block tier restored cached=0 before this change",
          flush=True)

    print("=== 1. seed the sessions (cold) ===", flush=True)
    cold = {}
    for s in seeds:
        t, k, a, th = ask(docs[s], "session %d" % s)
        hits, bad = score(s, a, th, seeds)
        cold[s] = t
        print("       cold recall %d/5%s" % (
            len(hits), "  CONTAMINATED: %s" % bad if bad else ""), flush=True)
        if len(hits) < len(DEPTHS):
            print("       WARNING: cold prefill itself missed a needle -- the "
                  "oracle is weaker than the test", flush=True)
    time.sleep(5)

    rows, bad_any = [], False
    for trial in range(TRIALS):
        print("=== 2.%d evict with %d fillers ===" % (trial, NFILL), flush=True)
        for i in range(1, NFILL + 1):
            ask(doc(5000 + trial * 100 + i, CTX), "filler %d" % i, quiet=True)
        time.sleep(8)

        print("=== 3.%d revisit each evicted session ===" % trial, flush=True)
        for s in seeds:
            b = counters()
            t, k, a, th = ask(docs[s], "trial %d sess %d" % (trial, s))
            time.sleep(3)
            c = counters()
            hits, others = score(s, a, th, seeds)
            degenerate = len(a.strip()) < 3
            ok = len(hits) == len(DEPTHS) and not others and not degenerate
            bad_any = bad_any or not ok
            print("       recall %d/5  jobs=%.0f  restored=%.0f MiB  "
                  "from_disk=+%.0f blk/%.0f MiB  promo_retry=+%.0f%s%s" % (
                      len(hits), c["jobs"] - b["jobs"],
                      (c["c2g"] - b["c2g"]) / 2 ** 20,
                      c["disk_hits"] - b["disk_hits"],
                      (c["disk_bytes"] - b["disk_bytes"]) / 2 ** 20,
                      c["promo_fail"] - b["promo_fail"],
                      "  CONTAMINATED: %s" % others if others else "",
                      "  DEGENERATE OUTPUT" if degenerate else ""), flush=True)
            if not hits:
                print("       answer was: %r" % a[:160], flush=True)
            rows.append((trial, s, cold[s], t, k, len(hits),
                         c["jobs"] - b["jobs"],
                         (c["c2g"] - b["c2g"]) / 2 ** 20, ok))

    print("=== 4. summary ===", flush=True)
    print("  %-6s %-4s %-10s %-10s %-8s %-8s %-6s %-9s %s" % (
        "trial", "sess", "cold TTFT", "warm TTFT", "speedup", "cached",
        "recall", "jobs", "restored"), flush=True)
    for tr, s, tc, tw, k, h, jobs, mib, ok in rows:
        sp = ("%.1fx" % (tc / tw)) if (tc and tw) else "n/a"
        print("  %-6s %-4s %-10s %-10s %-8s %-8s %-6s %-9.0f %.0f MiB%s" % (
            tr, s, ("%.2fs" % tc) if tc else "n/a",
            ("%.2fs" % tw) if tw else "n/a", sp, k, "%d/5" % h, jobs, mib,
            "" if ok else "   <-- FAIL"), flush=True)
    restored = [r for r in rows if (r[4] or 0) > 0]
    print("  restores that happened at all: %d/%d" % (len(restored), len(rows)),
          flush=True)
    if restored:
        print("  mean jobs per restore: %.1f (>1 means the chain ran)"
              % (sum(r[6] for r in restored) / len(restored)), flush=True)
    print("  VERDICT: %s" % ("PASS -- tier no longer bounds the restore"
                             if restored and not bad_any else
                             "FAIL -- see rows above"), flush=True)
    print("W30_2B_DONE", flush=True)
    sys.exit(0 if (restored and not bad_any) else 1)
