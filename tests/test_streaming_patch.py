#!/usr/bin/env python3
"""Offline tests for patch_offload_streaming_restore.py.

The patch rewrites four vLLM source files at container start. Every failure mode
this has actually produced was mechanical and offline-detectable:

  * a sentinel that also appears in another edit's NEW text, so the second edit
    is silently skipped and the patch lands half-applied;
  * an anchor that matches zero or two places after an earlier edit rewrote the
    region, so the chain order is wrong;
  * inserted text at the wrong indentation, which for a class body means the
    method silently attaches to something else;
  * a second application that is not a no-op.

All four are caught here in milliseconds. A live iteration is ~15 minutes, so
this file is the cheap instrument for the expensive experiment.

Run with PRISTINE=<dir> pointing at a tree of untouched vLLM sources
(docker cp'd out of the image, not the running container, which is patched).
"""
import ast
import importlib.util
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _pick(env, *candidates):
    """Prefer $env, else the first candidate that exists, else the first.

    These default to the copies in THIS repo. An earlier version defaulted to
    the author's deployment paths, which made the tests unrunnable for anyone
    who cloned the repo -- found by running them in a clean clone of the pushed
    tree, which is the only way that class of mistake shows up.
    """
    v = os.environ.get(env)
    if v:
        return Path(v)
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])


PRISTINE = _pick("PRISTINE", "/tmp/pristine/vllm",
                 "/usr/local/lib/python3.12/dist-packages/vllm")
PATCH = _pick("PATCH", REPO / "patch_offload_streaming_restore.py")
GROUPS = _pick("GROUPS_PATCH", REPO / "patch_offload_nonshareable_groups.py")

if not PRISTINE.is_dir():
    print("SKIP: no vLLM source tree at %s" % PRISTINE)
    print("      This test applies the patch chain to real upstream sources.")
    print("      Point it at a copy:  PRISTINE=/path/to/site-packages/vllm")
    print("      Nothing is written back -- a read-only copy is fine.")
    raise SystemExit(0)

fails = []


def check(name, cond, detail=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", name,
                           (" -- " + detail) if detail else ""))
    if not cond:
        fails.append(name)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)     # __main__ guard keeps main() from running
    return mod


m = load("sr", PATCH)
# The streaming patch does NOT stand alone: its scheduler anchors quote lines
# that the group-filter patch introduces (the non-shareable-group skip). Both
# run at container start, group filter first, so the chain is modelled here in
# that order -- otherwise the test reports phantom anchor misses.
g = load("gf", GROUPS)

LISTS = {
    "COMMON": (m.COMMON, m.COMMON_EDITS),
    "WORKER": (m.WORKER, m.WORKER_EDITS),
    "SCHED": (m.SCHED, g.EDITS + m.SCHED_EDITS + m.SCHED2B_EDITS),
    "MANAGER": (m.MANAGER, m.MANAGER_EDITS),
}

print("=== 1. no sentinel appears in another edit's replacement text ===")
# This is the bug that silently skipped edits twice in this project.
# Only the dangerous direction is a bug: if edit B's replacement contains edit
# A's sentinel and A is checked AFTER B, then A sees its sentinel already in the
# source and is silently skipped. The reverse -- B repeating a line A already
# added -- is normal, since B's anchor quotes A's output.
for name, (_path, edits) in LISTS.items():
    order = {label: i for i, (label, _o, _n, _s) in enumerate(edits)}
    bad = []
    for label, _old, _new, sentinel in edits:
        for other_label, _o, other_new, _s in edits:
            if (other_label != label and sentinel in other_new
                    and order[other_label] < order[label]):
                bad.append("%s (edit %d) would be skipped by %s (edit %d)"
                           % (label, order[label], other_label,
                              order[other_label]))
    check("%s: no edit is masked by an earlier edit's output" % name, not bad,
          "; ".join(bad))

print("=== 2. every sentinel is actually present in its own replacement ===")
# Otherwise the edit re-applies forever and duplicates code.
for name, (_path, edits) in LISTS.items():
    bad = [label for label, _o, new, sentinel in edits if sentinel not in new]
    check("%s: each new text contains its sentinel" % name, not bad, str(bad))

print("=== 3. the edit chain applies cleanly to pristine upstream ===")
patched = {}
for name, (path, edits) in LISTS.items():
    src_path = PRISTINE / path.relative_to(m.VLLM)
    if not src_path.is_file():
        check("%s: pristine source present" % name, False, str(src_path))
        continue
    src = src_path.read_text()
    applied = []
    ok = True
    for label, old, new, sentinel in edits:
        if sentinel in src:
            check("%s/%s: sentinel already in pristine source" % (name, label),
                  False, "anchor logic would skip this edit")
            ok = False
            continue
        n = src.count(old)
        if n != 1:
            check("%s/%s: anchor matches exactly once" % (name, label), False,
                  "matched %d times (after: %s)" % (n, ",".join(applied) or "-"))
            ok = False
            continue
        src = src.replace(old, new)
        applied.append(label)
    check("%s: all %d edits applied in order" % (name, len(edits)),
          ok and len(applied) == len(edits), "applied " + ",".join(applied))
    patched[name] = src

print("=== 4. the result is valid Python ===")
for name, src in patched.items():
    try:
        compile(src, name, "exec")
        check("%s: compiles" % name, True)
    except SyntaxError as e:
        check("%s: compiles" % name, False, "%s line %s" % (e.msg, e.lineno))

print("=== 5. re-applying the patch is a no-op (idempotent) ===")
for name, (_path, edits) in LISTS.items():
    if name not in patched:
        continue
    skipped = [label for label, _o, _n, sentinel in edits
               if sentinel in patched[name]]
    # (the group-filter edits are included in SCHED and must also be no-ops)
    check("%s: second run skips all %d edits" % (name, len(edits)),
          len(skipped) == len(edits),
          "would re-apply " + ",".join(
              l for l, _o, _n, s in edits if s not in patched[name]))

print("=== 6. inserted methods land on the intended classes ===")
# Indentation errors do not raise; they silently reparent a method. Assert the
# new members belong to the classes that are supposed to own them.
EXPECT = {
    "SCHED": ("OffloadingConnectorScheduler",
              ["_issue_chained_load", "_stream_advance", "_stream_sequential"]),
    "MANAGER": ("TieringOffloadingManager",
                ["stream_residency", "_stream_defer_enabled", "prepare_load",
                 "lookup"]),
}
for name, (cls_name, members) in EXPECT.items():
    if name not in patched:
        continue
    tree = ast.parse(patched[name])
    classes = {n.name: n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef)}
    if cls_name not in classes:
        check("%s: class %s still exists" % (name, cls_name), False)
        continue
    defined = {n.name for n in classes[cls_name].body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [x for x in members if x not in defined]
    check("%s: %s owns %s" % (name, cls_name, ", ".join(members)),
          not missing, "missing: " + ", ".join(missing))
    # the marker class from an earlier draft must not have leaked in
    check("%s: no stray top-level helper class" % name,
          "_StreamDeferredMixinMarker" not in classes)

print("=== 7. the streaming driver is wired into the step loop ===")
if "SCHED" in patched:
    s = patched["SCHED"]
    # _stream_advance must be called from build_connector_meta BEFORE
    # on_schedule_end, or a promotion waits a whole extra step to be submitted.
    body = s[s.index("def build_connector_meta("):]
    body = body[:body.index("def has_pending_push_work(")]
    i_drive = body.find("self._stream_advance(")
    i_end = body.find("self.manager.on_schedule_end(")
    check("driver runs inside build_connector_meta", i_drive != -1)
    check("driver runs before on_schedule_end",
          i_drive != -1 and i_end != -1 and i_drive < i_end,
          "drive@%d end@%d" % (i_drive, i_end))
    check("plan dict initialised in __init__",
          "self._stream_plans: dict = {}" in s)
    check("sequential path does not fall through to the eager loop",
          s.count("self._stream_advance(request.request_id)\n            return")
          == 1)

print("=== 8. residency is never read from the patched lookup ===")
if "MANAGER" in patched:
    tree = ast.parse(patched["MANAGER"])
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "stream_residency"),
              None)
    check("stream_residency exists", fn is not None)
    if fn is not None:
        calls = [ast.unparse(n.func) for n in ast.walk(fn)
                 if isinstance(n, ast.Call)]
        check("residency read from primary_tier.lookup",
              "self.primary_tier.lookup" in calls, str(calls))
        # self.lookup may be called to DRIVE promotion, but its result must not
        # be compared against HIT -- that is the lie this patch introduces.
        cmp_srcs = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Compare)]
        check("self.lookup's result is not treated as residency",
              not any("self.lookup" in c for c in cmp_srcs), str(cmp_srcs))
        # Once lookup() returns before _initiate_promotion, routing promotion
        # through it drives nothing and the driver livelocks instead of the
        # scan. It must promote against the secondary tiers itself.
        check("residency drives promotion directly, not via self.lookup",
              "self._initiate_promotion" in calls
              and "self.lookup" not in calls, str(calls))

    # And the patched lookup must return BEFORE promoting -- promoting during
    # the prefix scan is what ties restore size to tier size, whichever way the
    # promotion goes (MISS truncates the scan, HIT_PENDING defers it forever).
    src_mgr = patched["MANAGER"]
    i_gate = src_mgr.find("if self._stream_defer_enabled():")
    i_promote = src_mgr.find("promoted = self._initiate_promotion(i, key, req_context)")
    check("lookup checks the streaming gate before promoting",
          i_gate != -1 and i_promote != -1 and i_gate < i_promote,
          "gate@%d promote@%d" % (i_gate, i_promote))

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL STREAMING-PATCH TESTS PASSED")
