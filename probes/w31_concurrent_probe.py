#!/usr/bin/env python3
"""W31: what happens when N agents restore deep contexts AT THE SAME TIME?

Every result so far is a single sequential restore. The claim people care about
is multi-agent, and that is a different regime: N requests interleave their
batches through ONE shared CPU staging tier (79 blocks, 39 per batch), so only
~2 deep restores can hold a full batch at once.

Four things are measured, in order of how much they matter:

  1. CORRECTNESS under contention. Each session carries five needles at distinct
     depths AND a session-unique needle prefix, so both a truncated restore and a
     cross-session mix-up are visible. This is the one that would be a silent
     data bug rather than a slowdown.
  2. LIVENESS. A queued batch holds no reference, so another request's stores can
     evict its blocks from the disk tier mid-restore. Exposure grows with N. A
     stalled batch shows up as a request that never returns, so every request has
     a hard timeout and a timeout is a FAILURE, not a slow result.
  3. SCALING. Expect serialisation, not failure: promotions that cannot allocate
     retry once the in-flight batch's complete_load returns its blocks.
  4. Whether GLM53_OFFLOAD_STREAM_BATCH_BLOCKS actually buys concurrency.

Sequential baseline is measured in the same run, so the concurrent numbers are
compared against this machine on this day, not against an earlier session.
"""
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

MODEL = "GLM-5.3-Flash-EXL3"
BASE = os.environ.get("PROBE_BASE", "http://127.0.0.1:8000")
KEY = os.environ.get("VLLM_API_KEY", "")
# ~60k tokens. Using blocks = 2.0*chunks + 5.9 (measured), a 60k session is ~39
# offload blocks -- exactly the batch cap and half the 79-block tier. So ONE
# restore fits comfortably and TWO already oversubscribe it, which is the
# property under test. Going deeper (250k = 142 blocks) tests nothing extra
# about contention and costs ~4x more to seed.
CTX = int(os.environ.get("CTX", "46500"))
FILL_CTX = int(os.environ.get("FILL_CTX", "0")) or None
NAGENTS = [int(x) for x in os.environ.get("NAGENTS", "2,4,8").split(",")]
NFILL = int(os.environ.get("NFILL", "8"))
# A timeout MEANS "hung". A constant cannot mean that: 900 s was right at 250k
# idle and wrong at 950k, then 1133 s was right at 250k idle and wrong at 250k
# under load, because contention changed the RATE, not the size. So it is
# calibrated at runtime from what the engine is actually achieving. Set by
# calibrate_timeout() before seeding; this is only the fallback.
TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "0")) or 1800
DEPTHS = (0.10, 0.30, 0.50, 0.70, 0.90)
W = ["alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge",
     "ionic", "joule", "kelvin", "lumen", "matrix", "nadir", "orbit",
     "prism", "quartz", "rotor", "sigma", "torus", "umbra", "vector"]
Q = ("List the five access codes recorded in this document, in the order they "
     "appear, separated by commas. Output only the codes.")


def needles(seed):
    # Session-unique prefix AND per-depth index: a cross-session leak and a
    # truncated restore are then distinguishable from each other.
    return ["S%02dK%d-%04d" % (seed, i, 7000 + seed * 13 + i)
            for i in range(len(DEPTHS))]


def doc(seed, n_tokens):
    r = random.Random(seed)
    words = ["%s%d" % (r.choice(W), r.randint(0, 999))
             for _ in range(int(n_tokens / 2.4))]
    ns = needles(seed)
    for i in reversed(range(len(DEPTHS))):
        words.insert(int(len(words) * DEPTHS[i]),
                     "Access code %d is %s." % (i, ns[i]))
    return "[document %d]\n%s\n[end document]" % (seed, " ".join(words))


def hdr():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def counters():
    req = urllib.request.Request(BASE + "/metrics", headers=hdr())
    o = {"c2g": 0.0, "jobs": 0.0, "disk_hits": 0.0, "alloc_fail": 0.0}
    try:
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
                elif line.startswith("vllm:kv_offload_tiering_block_hits_total") \
                        and 'tier="0:primary"' not in line:
                    o["disk_hits"] += v
                elif line.startswith("vllm:kv_offload_allocation_failure_total"):
                    o["alloc_fail"] = v
    except Exception:
        pass
    return o


def ask(text, timeout=TIMEOUT):
    """Returns (ttft, cached, prompt_tokens, answer, thinking, error)."""
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
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return None, None, None, "", "", "HTTP %s: %s" % (e.code, e.read()[:120])
    except Exception as e:                       # timeout == a hung restore
        return None, None, None, "", "", "%s: %s" % (type(e).__name__, e)
    try:
        with r:
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
    except Exception as e:
        return ttft, None, None, "".join(out), "".join(think), \
               "stream broke: %s" % type(e).__name__
    det = usage.get("prompt_tokens_details") or {}
    return (ttft, det.get("cached_tokens"), usage.get("prompt_tokens"),
            "".join(out), "".join(think), None)


def score(seed, answer, thinking, seeds):
    hay = answer + "\n" + thinking
    hits = [n for n in needles(seed) if n in hay]
    others = [n for o in seeds if o != seed for n in needles(o) if n in hay]
    return hits, others


def fire_concurrently(seeds, docs):
    """All N revisits in flight at once -- the whole point of this probe."""
    res = {}
    lock = threading.Lock()

    def one(s):
        r = ask(docs[s])
        with lock:
            res[s] = r

    threads = [threading.Thread(target=one, args=(s,)) for s in seeds]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return res, time.perf_counter() - t0


def report(label, seeds, res, wall, before, after, cold):
    ok_all = True
    print("  --- %s: %d concurrent, wall %.1f s ---" % (label, len(seeds), wall),
          flush=True)
    for s in sorted(res):
        ttft, cached, prompt, a, th, err = res[s]
        if err:
            print("    sess %-2d FAILED: %s" % (s, err), flush=True)
            ok_all = False
            continue
        hits, others = score(s, a, th, seeds)
        degenerate = len(a.strip()) < 3
        ok = len(hits) == len(DEPTHS) and not others and not degenerate
        ok_all = ok_all and ok
        sp = ("%.1fx" % (cold[s] / ttft)) if (cold.get(s) and ttft) else "n/a"
        print("    sess %-2d TTFT=%7.2fs (%s vs cold) cached=%-8s recall=%d/5%s%s"
              % (s, ttft or -1, sp, cached, len(hits),
                 "  LEAKED %s" % others if others else "",
                 "  DEGENERATE" if degenerate else ""), flush=True)
    d = {k: after[k] - before[k] for k in before}
    print("    tier: jobs=+%.0f  CPU->GPU=+%.0f MiB  disk_hits=+%.0f  "
          "alloc_retries=+%.0f" % (d["jobs"], d["c2g"] / 2 ** 20,
                                   d["disk_hits"], d["alloc_fail"]), flush=True)
    return ok_all


def _gauge(name, substr=None):
    """One scalar out of /metrics, or None."""
    try:
        req = urllib.request.Request(BASE + "/metrics", headers=hdr())
        with urllib.request.urlopen(req, timeout=30) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#") or not line.startswith(name):
                    continue
                if substr and substr not in line:
                    continue
                return float(line.rsplit(" ", 1)[-1])
    except Exception:
        pass
    return None


def wait_for_quiet(need_quiet=60, max_wait=7200):
    """Block until the engine is idle enough to seed at full speed.

    Seeding a deep session against sustained multi-agent decode runs ~9x slower
    (measured: 1000 -> 107 tok/s aggregate), which turns a 10 minute setup into
    hours. Rather than fight the decode-priority gate, wait it out -- the owner
    keeps their latency and the experiment gets a clean setup.
    """
    t0 = time.time()
    quiet_since = None
    while time.time() - t0 < max_wait:
        running = _gauge("vllm:num_requests_running")
        if running is not None and running <= 1:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= need_quiet:
                print("  engine quiet for %ds; seeding now" % need_quiet,
                      flush=True)
                return True
        else:
            if quiet_since:
                print("  busy again (running=%s); resetting quiet timer"
                      % running, flush=True)
            quiet_since = None
        time.sleep(10)
    print("  no quiet window in %ds; proceeding anyway (it will be slow)"
          % max_wait, flush=True)
    return False


def calibrate_timeout(needed_tokens, sample_s=45):
    """Size the per-request timeout from measured prefill throughput.

    Samples vllm:prompt_tokens_total over a window and divides by the number of
    requests sharing the engine, so the timeout tracks the conditions the run
    will actually meet. Floored so an idle sample cannot produce an absurdly
    small one, and capped so a hang is still caught in bounded time.
    """
    a = _gauge("vllm:prompt_tokens_total")
    t0 = time.time()
    time.sleep(sample_s)
    b = _gauge("vllm:prompt_tokens_total")
    running = _gauge("vllm:num_requests_running") or 0
    elapsed = time.time() - t0
    rate = None
    # Reject a sample window that did not actually elapse: dividing a token
    # delta by ~0 seconds yields a nonsense rate, which collapses the timeout to
    # its floor -- reintroducing exactly the bug this function exists to fix.
    if a is not None and b is not None and b > a and elapsed >= sample_s * 0.5:
        rate = (b - a) / elapsed
    # An idle engine measures ~0 prefill; fall back to this machine's measured
    # idle capability rather than to infinity.
    effective = (rate or 800.0) / max(1.0, running)
    effective = max(effective, 40.0)
    # The floor must scale with the work, not be a constant. A 1800 s constant
    # is fine for 60k and far too short for 250k under contention, which needed
    # 2330 s -- that constant is what killed the previous run. 25 tok/s is the
    # worst per-request rate actually observed (107 aggregate over 4 requests).
    floor = needed_tokens / 25.0
    t = int(min(10800, max(1800, floor, (needed_tokens / effective) * 2.0)))
    print("  calibration: aggregate prefill %s tok/s, running=%s -> assuming "
          "%.0f tok/s for us -> timeout %ds for %d tokens"
          % ("%.0f" % rate if rate else "idle", running, effective, t,
             needed_tokens), flush=True)
    return t


def safety() -> tuple:
    """Is it still safe to raise concurrency?

    The 2026-09-02 incident was 16 concurrent deep sessions -> 1037 NVRM
    NV_ERR_NO_MEMORY in a BURST -> host swap-starved, OOM killer never fired,
    hard reset. Isolated NV_ERR_NO_MEMORY events are baseline on this node
    (roughly hourly under deep load), so the signal is a same-minute CLUSTER,
    not a count. MemFree is the number CUDA actually sees on GB10 -- page cache
    hides memory from it -- so both are reported.
    """
    import subprocess
    burst, total = 0, 0
    try:
        out = subprocess.run(
            ["journalctl", "-k", "--since", "-12min", "--no-pager"],
            capture_output=True, text=True, timeout=60).stdout
        stamps = [" ".join(l.split()[:3])[:16] for l in out.splitlines()
                  if "NV_ERR_NO_MEMORY" in l]
        total = len(stamps)
        for t in set(stamps):
            burst = max(burst, stamps.count(t))
    except Exception:
        pass
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k in ("MemFree", "MemAvailable", "SwapFree"):
                mem[k] = int(v.split()[0]) / 1048576.0
    except Exception:
        pass
    ok = burst < 3
    return ok, ("NVRM last 12min: %d total, largest same-minute burst %d | "
                "MemFree %.1f GiB, MemAvailable %.1f GiB, SwapFree %.1f GiB"
                % (total, burst, mem.get("MemFree", -1),
                   mem.get("MemAvailable", -1), mem.get("SwapFree", -1)))


if __name__ == "__main__":
    NSESS = int(os.environ.get("NSESS", "8"))
    ROUNDS = int(os.environ.get("ROUNDS", "1"))
    seeds = list(range(NSESS))
    print("=== W31: %d sessions x %d tokens; concurrency ramp %s; %d round(s) ==="
          % (NSESS, CTX, NAGENTS, ROUNDS), flush=True)
    ok, msg = safety()
    print("    pre-flight: %s" % msg, flush=True)
    if not ok:
        raise SystemExit("ABORT before starting: NVRM burst already present")

    docs = {s: doc(s, CTX) for s in seeds}

    print("=== waiting for a quiet window before seeding ===", flush=True)
    wait_for_quiet()
    globals()["TIMEOUT"] = calibrate_timeout(int(CTX * 1.29))

    # Seeding is also the eviction: NSESS x CTX deliberately exceeds the GPU
    # pool, so by the time the last session is seeded the first is already out
    # of it and can only come back from disk. That removes the ~29 min filler
    # phase each concurrency step would otherwise need.
    print("=== seed %d sessions (cold). This is also what evicts them. ==="
          % NSESS, flush=True)
    cold = {}
    for s in seeds:
        ttft, cached, prompt, a, th, err = ask(docs[s])
        if err:
            raise SystemExit("seeding session %d failed: %s" % (s, err))
        hits, _ = score(s, a, th, seeds)
        cold[s] = ttft
        print("  sess %-2d cold TTFT=%7.2fs prompt=%-7s cached=%-8s recall=%d/5%s"
              % (s, ttft or -1, prompt, cached, len(hits),
                 "   <-- ORACLE WEAK HERE" if len(hits) < 5 else ""), flush=True)
    time.sleep(5)

    # Always revisit the LEAST recently used sessions, so every round is a real
    # disk restore rather than a GPU prefix-cache hit.
    from collections import deque
    lru = deque(seeds)
    all_ok, rows = True, []
    for rnd in range(ROUNDS):
        for n in NAGENTS:
            if n > NSESS:
                continue
            ok, msg = safety()
            print("=== round %d, N=%d ===\n    %s" % (rnd, n, msg), flush=True)
            if not ok:
                print("    ABORT: NVRM burst detected, not raising concurrency"
                      " further", flush=True)
                all_ok = False
                break
            sub = [lru.popleft() for _ in range(n)]
            b = counters()
            res, wall = fire_concurrently(sub, docs)
            time.sleep(4)
            a2 = counters()
            good = report("round %d N=%d" % (rnd, n), sub, res, wall, b, a2, cold)
            all_ok = all_ok and good
            for x in sub:
                lru.append(x)                  # they are now most recently used
            rows.append((rnd, n, wall, good))
        else:
            continue
        break

    print("=== summary ===", flush=True)
    print("  %-6s %-4s %-10s %s" % ("round", "N", "wall", "clean"), flush=True)
    for rnd, n, wall, good in rows:
        print("  %-6d %-4d %-10.1f %s" % (rnd, n, wall, "yes" if good else "NO"),
              flush=True)
    ok, msg = safety()
    print("  post-flight: %s" % msg, flush=True)
    print("=== verdict: %s ===" % ("PASS" if all_ok else "FAIL"), flush=True)
    print("W31_DONE", flush=True)
    sys.exit(0 if all_ok else 1)
