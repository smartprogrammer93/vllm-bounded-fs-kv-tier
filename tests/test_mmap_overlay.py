#!/usr/bin/env python3
"""Offline test of patch_offload_mmap_region.py against pristine sources.

Every anchor is verified before this goes near a boot -- two boots were lost
today to a launcher change that was not tested first.
"""
import importlib.util, os, shutil, sys, tempfile
from pathlib import Path

# Resolve the patch next to this repo, so the suite runs from a fresh clone.
# PATCH= overrides it when the file lives elsewhere (e.g. a deployment overlay).
PATCH = Path(os.environ.get(
    "PATCH", Path(__file__).resolve().parent.parent / "patch_offload_mmap_region.py"))
# Pristine vLLM sources to patch. PRISTINE= points at an installed copy, e.g.
# PRISTINE=/usr/local/lib/python3.12/dist-packages/vllm
PRIS = Path(os.environ.get("PRISTINE", "/tmp/mmaptest/vllm"))
WORK = Path(tempfile.gettempdir()) / "mmaptest_work"
fails = []

if not (PRIS / "v1/kv_offload/cpu/shared_offload_region.py").is_file():
    print("SKIP: no pristine vLLM sources at %s\n"
          "      set PRISTINE=/path/to/site-packages/vllm to run this suite" % PRIS)
    raise SystemExit(0)


def check(n, c, d=""):
    print("  %-4s %s%s" % ("PASS" if c else "FAIL", n, (" -- " + d) if d else ""))
    if not c:
        fails.append(n)


spec = importlib.util.spec_from_file_location("mm", PATCH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print("=== 1. every anchor matches pristine exactly once ===")
for name, path, edits in (("shared_offload_region.py", PRIS / m.SOR.relative_to(m.VLLM), m.SOR_EDITS),
                          ("gpu_worker.py", PRIS / m.GWK.relative_to(m.VLLM), m.GWK_EDITS)):
    src = path.read_text()
    for label, old, new, sentinel in edits:
        check("%s / %s" % (name, label), src.count(old) == 1,
              "matched %d" % src.count(old))
        check("%s / %s sentinel absent in pristine" % (name, label),
              sentinel not in src)

print("=== 2. the patched result compiles ===")
if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(PRIS, WORK)
os.environ["GLM53_OFFLOAD_MMAP_DIR"] = "/tmp/kvmmap"
m.VLLM = WORK
m.SOR = WORK / "v1/kv_offload/cpu/shared_offload_region.py"
m.GWK = WORK / "v1/kv_offload/cpu/gpu_worker.py"
out1 = m.apply_edits(m.SOR, m.SOR_EDITS)
out2 = m.apply_edits(m.GWK, m.GWK_EDITS)
check("shared_offload_region.py applied+compiles", "applied" in out1, out1)
check("gpu_worker.py applied+compiles", "applied" in out2, out2)

print("=== 3. idempotent: a second run is a no-op ===")
r1 = m.apply_edits(m.SOR, m.SOR_EDITS)
r2 = m.apply_edits(m.GWK, m.GWK_EDITS)
check("second run skips every edit", "applied" not in r1 and "applied" not in r2,
      "%s | %s" % (r1, r2))

print("=== 4. the disabled path leaves vLLM pristine ===")
del os.environ["GLM53_OFFLOAD_MMAP_DIR"]
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = m.main()
check("main() is a no-op with the knob unset",
      rc == 0 and "pristine" in buf.getvalue(), buf.getvalue().strip()[:60])

print("=== 5. no sentinel collides with another edit's new text ===")
for name, edits in (("SOR", m.SOR_EDITS), ("GWK", m.GWK_EDITS)):
    order = {l: i for i, (l, _o, _n, _s) in enumerate(edits)}
    bad = [(l, ol) for l, _o, _n, sent in edits
           for ol, _o2, onew, _s2 in edits
           if ol != l and sent in onew and order[ol] < order[l]]
    check("%s: no edit masked by an earlier one" % name, not bad, str(bad))

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL MMAP-OVERLAY TESTS PASSED")
