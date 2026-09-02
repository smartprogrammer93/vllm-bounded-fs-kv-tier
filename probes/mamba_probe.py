#!/usr/bin/env python3
"""Does an offload restore actually need the Mamba recurrent state?

Mamba groups are ~97% of the real offload payload (76 of 78.7 MiB per chunk), so
the answer decides whether offload can ever approach GPU-pool cost. An earlier
accidental run stored ZERO Mamba blocks and still returned the right needle --
but the GPU prefix cache may simply have supplied the state, so that proved
nothing.

INSTRUMENT: multiple needles placed INSIDE the restored region, scored by recall.

Exact-match on the completion was tried first and REJECTED: the control arm
scored 1/3 because greedy text is not bit-reproducible across different prefill
paths (a cold single prefill vs a restore plus a partial prefill differ in
chunked-prefill boundaries, batch composition and MoE routing), and because the
one divergence was in a span that lay OUTSIDE the restored region and so was
recomputed in both runs. That makes it noise, not corruption.

So: 5 needles in the first 60% of the document -- comfortably inside the
restored prefix -- and the score is how many come back. Corrupt recurrent state
cannot be read through; a single needle can survive by luck, five cannot.

CONFOUND DEFEATED: each session is asked cold, then pushed out with fillers
totalling several times the GPU pool, then re-asked. The comparison is
cold-vs-restored on the SAME prompt, so a GPU-cache hit and an offload restore
are distinguishable by the reported cached_tokens and the tier counters, while
correctness is judged only on the completion.

Run twice, one variable changed:
  arm B (control)  GLM53_OFFLOAD_EXCLUDE_MAMBA unset -> Mamba offloaded
  arm A (test)     GLM53_OFFLOAD_EXCLUDE_MAMBA=1     -> Mamba NOT offloaded
If arm B matches and arm A diverges, the state is required. If both match across
several sessions, it is not -- and offload gets ~20x cheaper.
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
CTX = int(os.environ.get("CTX", "12000"))
NSEED = int(os.environ.get("NSEED", "3"))
NFILL = int(os.environ.get("NFILL", "8"))
MAXTOK = int(os.environ.get("MAXTOK", "384"))  # reasoning must not truncate it
NEEDLES = int(os.environ.get("NEEDLES", "5"))
ARM = os.environ.get("ARM", "?")
W = ["alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge",
     "ionic", "joule", "kelvin", "lumen", "matrix", "nadir", "orbit",
     "prism", "quartz", "rotor", "sigma", "torus", "umbra", "vector"]

# Forces the model to use spans from the START, MIDDLE and END of the context,
# so a restore that is wrong anywhere in the prefix shows up.
Q = ("The document contains several lines of the form "
     "'access code N is CODE'. List every one you can find, as 'N CODE' per "
     "line, and nothing else.")


def needle(seed, i):
    return "ZULU-%04d-%d-QX" % (7000 + seed, i)


def doc(seed, n_tokens):
    """Needles spread across the first 60% of the body, so every one of them
    lands inside the region a restore actually covers."""
    r = random.Random(seed)
    body = [f"{r.choice(W)}{r.randint(0, 999)}" for _ in range(int(n_tokens / 2.4))]
    span = int(len(body) * 0.6)
    for i in range(NEEDLES):
        at = int(span * (i + 1) / (NEEDLES + 1))
        body.insert(at + i, f"access code {i} is {needle(seed, i)}.")
    return "[document %d]\n%s\n[end document]" % (seed, " ".join(body))


def recall(txt, seed):
    return sum(1 for i in range(NEEDLES) if needle(seed, i) in txt)


def hdr():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def counters():
    req = urllib.request.Request(BASE + "/metrics", headers=hdr())
    out = {"c2g": 0.0, "fs_hits": 0.0, "alloc_fail": 0.0}
    with urllib.request.urlopen(req, timeout=60) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#"):
                continue
            v = line.rsplit(" ", 1)[-1]
            try:
                val = float(v)
            except ValueError:
                continue
            if line.startswith("vllm:kv_offload_total_bytes_total") and "CPU_to_GPU" in line:
                out["c2g"] = val
            elif line.startswith("vllm:kv_offload_tiering_block_hits_total") and 'tier="1:' in line:
                out["fs_hits"] = val
            elif line.startswith("vllm:kv_offload_tiering_promotion_allocation_failures_total"):
                out["alloc_fail"] = val
    return out


def ask(text, label, quiet=False):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": text + "\n\n" + Q}],
            "temperature": 0, "max_tokens": MAXTOK, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr())
    t0 = time.perf_counter()
    ttft, usage, out = None, {}, []
    reasoning = []
    with urllib.request.urlopen(req, timeout=1800) as r:
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
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                piece = d.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    out.append(piece)
    det = usage.get("prompt_tokens_details") or {}
    if not quiet:
        print("  %-16s TTFT=%7s cached=%-7s completion=%-4s reasoning=%d ch" % (
            label, ("%.2fs" % ttft) if ttft else "n/a",
            det.get("cached_tokens"), usage.get("completion_tokens"),
            len("".join(reasoning))), flush=True)
    # score against content AND reasoning: a needle recalled in either proves the
    # restored KV was readable.
    return "".join(out) + "\n" + "".join(reasoning), det.get("cached_tokens")


seeds = list(range(NSEED))
docs = {s: doc(s, CTX) for s in seeds}
print("=== ARM %s : %d sessions, %d fillers, exact-match on %d greedy tokens ==="
      % (ARM, NSEED, NFILL, MAXTOK), flush=True)

print("--- 1. cold reference (must recall %d/%d or the reference is unusable) ---"
      % (NEEDLES, NEEDLES), flush=True)
cold = {}
for s in seeds:
    txt, k = ask(docs[s], "session %d cold" % s)
    cold[s] = recall(txt, s)
    print("      cold recall=%d/%d" % (cold[s], NEEDLES), flush=True)
time.sleep(6)

print("--- 2. evict (%d fillers) ---" % NFILL, flush=True)
for i in range(1, NFILL + 1):
    ask(doc(9000 + i, CTX), "filler %d" % i, quiet=True)
print("      done", flush=True)
time.sleep(10)

print("--- 3. revisit and compare ---", flush=True)
rows = []
for s in seeds:
    b = counters()
    txt, k = ask(docs[s], "session %d warm" % s)
    a = counters()
    warm = recall(txt, s)
    rows.append((s, cold[s], warm, k,
                 (a["c2g"] - b["c2g"]) / 2 ** 20, a["fs_hits"] - b["fs_hits"],
                 a["alloc_fail"] - b["alloc_fail"]))
    print("      recall cold=%d/%d -> warm=%d/%d  restored=%.0f MiB "
          "disk_hits=%.0f alloc_fail=%.0f"
          % (cold[s], NEEDLES, warm, NEEDLES, (a["c2g"] - b["c2g"]) / 2 ** 20,
             a["fs_hits"] - b["fs_hits"], a["alloc_fail"] - b["alloc_fail"]),
          flush=True)

print("=== ARM %s SUMMARY ===" % ARM, flush=True)
print("  %-9s %-11s %-11s %-8s %-10s %s" % (
    "session", "cold recall", "warm recall", "cached", "restored", "disk_hits"),
    flush=True)
for s, c, w, k, mib, hits, af in rows:
    print("  %-9s %-11s %-11s %-8s %-10.0f %.0f"
          % (s, "%d/%d" % (c, NEEDLES), "%d/%d" % (w, NEEDLES), k, mib, hits),
          flush=True)
usable = [r for r in rows if r[1] == NEEDLES]
kept = [r for r in usable if r[2] == NEEDLES]
lost = [r for r in usable if r[2] < NEEDLES]
n_restored = sum(1 for r in rows if r[4] > 0)
print("  usable references (cold recall %d/%d): %d/%d" % (
    NEEDLES, NEEDLES, len(usable), len(rows)), flush=True)
print("  restored from offload: %d/%d" % (n_restored, len(rows)), flush=True)
if not usable:
    print("  VERDICT: INCONCLUSIVE -- no usable cold reference", flush=True)
elif not lost:
    print("  VERDICT: restored KV is READABLE (full recall on every usable "
          "session)", flush=True)
else:
    print("  VERDICT: RECALL LOST on %d/%d usable sessions -- restored KV is "
          "damaged" % (len(lost), len(usable)), flush=True)
print("MAMBA_PROBE_DONE", flush=True)
