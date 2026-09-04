"""Re-run every ladder cell whose completions were cut off, worst first.

A truncated completion is not a missing answer -- it parses, and what it yields is the last
number in a severed reasoning trace. On qm7/gem35flash/high those fragments sit at medAE 1568
against 89 for the intact calls in the same cell. So a truncated cell has to be re-measured with
a completion cap the model does not reach, not merely flagged.

Worst first, one invocation per cell, with a running total against a ceiling: cost extrapolated
from a capped run has under-predicted an uncapped one three times in this study, so the order
matters more than the estimate. If the ceiling stops the sweep, the cells that were most damaged
are the ones already fixed.

    python src/rerun_truncated.py --min-trunc 0.02 --max-tokens 20000 --budget 130
    python src/rerun_truncated.py --dry
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAD = os.path.join(ROOT, "results", "ladder")
ARCH = os.path.join(ROOT, "results", "ladder_archive")


def truncated_cells(min_trunc, skip_tags):
    out = []
    for p in sorted(glob.glob(os.path.join(LAD, "*.json"))):
        d = json.load(open(p, encoding="utf8"))
        m, c = d["meta"], d["calls"]
        if not c or m["tag"] in skip_tags:
            continue
        tr = sum(1 for x in c if x.get("finish") == "length")
        if tr / len(c) > min_trunc:
            out.append(dict(dataset=m["dataset"], tag=m["tag"], level=m["level"],
                            trunc=tr / len(c), cost=m.get("cost", 0.0), n_mol=m["n_mol"]))
    return sorted(out, key=lambda r: -r["trunc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trunc", type=float, default=0.02)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--budget", type=float, default=130.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--skip-tags", default="gpt55")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    cells = truncated_cells(a.min_trunc, set(a.skip_tags.split(",")) - {""})
    print(f"{len(cells)} cells above {a.min_trunc:.0%} truncation, "
          f"${sum(c['cost'] for c in cells):.2f} at the old cap")
    for c in cells:
        print(f"  {c['trunc']:5.0%}  {c['dataset']:14s} {c['tag']:16s} {c['level']:8s} "
              f"(was ${c['cost']:.2f})")
    if a.dry:
        return

    os.makedirs(ARCH, exist_ok=True)
    spent = 0.0
    for i, c in enumerate(cells, 1):
        if spent >= a.budget:
            print(f"\nBUDGET ${a.budget:.2f} reached after {i - 1}/{len(cells)} cells "
                  f"(${spent:.2f}). The remaining ones are the least truncated.")
            break
        name = f"{c['dataset']}__{c['tag']}__{c['level']}"
        src = os.path.join(LAD, name + ".json")
        # Keep the old cell: it is the evidence for what truncation does, and if the re-run
        # somehow comes back worse there has to be something to compare against.
        dst = os.path.join(ARCH, name + "__cap8k.json")
        if os.path.exists(src) and not os.path.exists(dst):
            os.replace(src, dst)
        print(f"\n[{i}/{len(cells)}] {name}  ({c['trunc']:.0%} truncated, ${spent:.2f} spent)",
              flush=True)
        r = subprocess.run(
            [sys.executable, "-u", os.path.join(ROOT, "src", "run_reasoning_ladder.py"),
             "--models", c["tag"], "--datasets", c["dataset"], "--levels", c["level"],
             "--n", str(c["n_mol"]), "--iters", str(a.iters),
             "--max-tokens", str(a.max_tokens), "--threads", str(a.threads),
             "--budget", f"{a.budget - spent:.2f}", "--force"],
            capture_output=True, text=True)
        sys.stdout.write(r.stdout[-1500:])
        if os.path.exists(src):
            m = json.load(open(src, encoding="utf8"))
            spent += m["meta"].get("cost", 0.0)
            tr = sum(1 for x in m["calls"] if x.get("finish") == "length")
            print(f"    -> {tr}/{len(m['calls'])} still truncated, "
                  f"${m['meta'].get('cost', 0):.2f}, running total ${spent:.2f}", flush=True)
        else:
            print("    -> cell not written; the old one stays archived", flush=True)
    print(f"\nDONE. ${spent:.2f} spent.")


if __name__ == "__main__":
    main()
