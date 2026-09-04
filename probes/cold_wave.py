#!/usr/bin/env python3
"""Concurrent COLD-prefill baseline: what does prefilling N sessions at once cost?

This is the measurement whose absence invalidated a published 7-8x throughput
figure. That number came from taking the SEQUENTIAL prefill rate and multiplying
by N, which assumes aggregate prefill throughput does not improve with
concurrency. On this server that assumption is suspect: max_num_batched_tokens
is 3584 with long_prefill_token_threshold 1792, and scheduler.py caps each long
prefill at the threshold per step -- so a lone prefill fills half the batch and
concurrent prefills can pack it.

So: fire N prefills AT ONCE and time the wave, against the same N restored
concurrently (already measured). Nothing derived, nothing multiplied.

CORRECTNESS. Every document is unique FROM TOKEN 0 (the seed drives the first
word), so no request can hit the GPU prefix cache or the offload tier. That is
asserted, not assumed: any response with cached_tokens > 0 invalidates its wave
and is reported as such rather than averaged in.
"""
import json, os, random, sys, threading, time, urllib.error, urllib.request

BASE = os.environ.get("PROBE_BASE", "http://127.0.0.1:8000")
KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = "GLM-5.3-Flash-EXL3"
CTX = int(os.environ.get("CTX", "46500"))          # ~60k actual, matches the restore waves
NAGENTS = [int(x) for x in os.environ.get("NAGENTS", "1,2,4,6,8").split(",")]
TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "900"))
W = ["alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge",
     "ionic", "joule", "kelvin", "lumen", "matrix", "nadir", "orbit",
     "prism", "quartz", "rotor", "sigma", "torus", "umbra", "vector"]


def doc(seed, n_tokens):
    """Unique from the FIRST word: seed picks the opening token."""
    r = random.Random(seed * 7919 + 13)
    head = "%s-%06d" % (W[seed % len(W)], seed * 31 + 7)
    body = " ".join(r.choice(W) for _ in range(int(n_tokens)))
    return head + " " + body


def ask(text):
    """Returns (ttft, total, prompt_tokens, cached_tokens, err). TTFT by stream."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": text + "\n\nReply with the single word: ok"}],
        "max_tokens": 2048, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    t0 = time.time()
    ttft = None
    ptok = ctok = None
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except ValueError:
                    continue
                if ttft is None and d.get("choices"):
                    delta = d["choices"][0].get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        ttft = time.time() - t0
                if d.get("usage"):
                    ptok = d["usage"].get("prompt_tokens")
                    det = d["usage"].get("prompt_tokens_details") or {}
                    ctok = det.get("cached_tokens")
        return ttft, time.time() - t0, ptok, ctok, None
    except Exception as e:
        return ttft, time.time() - t0, ptok, ctok, repr(e)[:120]


def running():
    try:
        req = urllib.request.Request(BASE + "/metrics",
                                     headers={"Authorization": "Bearer " + KEY})
        with urllib.request.urlopen(req, timeout=10) as r:
            for ln in r.read().decode().splitlines():
                if ln.startswith("vllm:num_requests_running{"):
                    return float(ln.rsplit(" ", 1)[1])
    except Exception:
        pass
    return -1.0


def wave(n, seed_base):
    res = [None] * n
    docs = [doc(seed_base + i, CTX) for i in range(n)]

    def go(i):
        res[i] = ask(docs[i])

    ths = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return time.time() - t0, res


def main():
    print("=== concurrent COLD prefill: waves of %s at CTX=%d ===" % (NAGENTS, CTX), flush=True)
    print("    foreign load at start: running=%.1f" % running(), flush=True)
    rows = []
    seed = int(time.time()) % 100000
    for n in NAGENTS:
        # Let the engine settle so a previous wave's decode does not bleed in.
        for _ in range(30):
            if running() <= 0:
                break
            time.sleep(2)
        fl = running()
        wall, res = wave(n, seed)
        seed += n
        ttfts = [r[0] for r in res if r and r[0] is not None]
        ptoks = [r[2] for r in res if r and r[2]]
        cached = [r[3] for r in res if r and r[3] is not None]
        errs = [r[4] for r in res if r and r[4]]
        dirty = [c for c in cached if c and c > 0]
        tot = sum(ptoks) if ptoks else 0
        print("  N=%-2d wall=%7.1fs  prompt_tot=%-8d  ttft max=%6.1fs min=%6.1fs  "
              "agg=%6.0f tok/s  cached=%s%s%s"
              % (n, wall, tot, max(ttfts) if ttfts else -1, min(ttfts) if ttfts else -1,
                 (tot / wall) if wall and tot else 0,
                 "0 (clean)" if not dirty else "%s <-- INVALID, cache hit" % dirty,
                 "  foreign_load=%.1f" % fl if fl > 0 else "",
                 "  ERR=%s" % errs[:1] if errs else ""), flush=True)
        rows.append((n, wall, tot, (tot / wall) if wall and tot else 0, not dirty and not errs))
    print("\n=== summary: aggregate cold-prefill throughput vs concurrency ===", flush=True)
    print("  %-4s %-10s %-12s %-12s %s" % ("N", "wall", "prompt_tok", "agg tok/s", "clean"), flush=True)
    for n, wall, tot, rate, ok in rows:
        print("  %-4d %-10.1f %-12d %-12.0f %s" % (n, wall, tot, rate, "yes" if ok else "NO"), flush=True)
    print("COLD_WAVE_DONE", flush=True)


if __name__ == "__main__":
    main()
