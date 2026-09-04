"""The blinding experiment scored on the DIGIT RUNGS, so it can be drawn as the map.

WHY THIS EXISTS. `analyze_blinding_sweep.py` scores the L1/L5 cells on hit3 against a floor
simulated at each arm's own accuracy. That is the right statistic for the question that section
asks -- how much of the benchmark survives blinding -- but it is the accuracy-matched
construction `classify.floors` abandoned for the verdicts, and it produces no verdict, no
retention and no q-value. So the blinding cells could not be drawn on the same scale as Fig. 2,
and the section that argues "what remains after blinding" had no map.

WHAT IT DOES. Recomputes m1/m2/m3 per (dataset, model, level) from the raw per-call files in
results/blinding/, then hands them to `classify.classify(gate="rung", floor="label")` -- the same
call the controlled map uses, so the floors, the power gate, the binomial tests and the verdict
strings are the map's by construction rather than by resemblance. Nothing is reimplemented here
except the counting, and the counting comes from `memodetect.nested_matches`, which is also what
`analyze_budget` uses.

THE FAMILY IS THE EXPERIMENT'S OWN. One BH family over the 24 cells (12 L1 + 12 L5), for the
reason the arms are never pooled in classify: the blinding cells are a declared experiment with
its own panel, its own molecules and its own shot count, and correcting them inside the map's
437-test family would correct a comparison against tests that are not part of it. The family is
small, which makes each q SMALLER than it would be in the map's -- so a flag here is not
interchangeable with a flag there and the figure says which arm it is.

    python src/analyze_blinding_map.py            # writes results/blinding_map.csv
    python src/analyze_blinding_map.py --arm ""   # the un-suffixed (default-reasoning) files
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify as C
from memodetect import nested_matches, sig_figs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
BLIND = os.path.join(RES, "blinding")

# The declared panel of the blinding experiment, in the order the section introduces it: the two
# most contaminated models in the study, then the two the zero-shot screen puts near the floor.
PANEL = ["opus5", "grok45", "sol", "kimik3"]
DS_ORDER = ["freesolv", "esol", "ld50"]


def cell(path, pad=True):
    """(m1, m2, m3, n_usable, medae) for one L1/L5 file.

    A pair is scorable only if the PUBLISHED value carries three significant figures: a label
    written to two figures has no third figure to agree with, and counting it in the denominator
    would depress every cell on a benchmark that rounds its values.

    `pad` is the same switch as `analyse_cell(pad_preds=...)` and it is the one place this script
    has to make a choice rather than inherit one.

        pad=True   (default, and what the controlled map uses -- `analyze_budget.py --pad`)
                   the requirement falls on the truth only, so an answer of "2.2" is read as
                   2.20: the prompt asked for three significant figures, so a short answer is
                   the model's assertion AT the requested precision.
        pad=False  what `analyze_blinding_sweep.py` does, and therefore what Table l1l5 and
                   every L1/L5 number in the text rest on.

    The L1/L5 prompts DO request three significant figures ("Provide your answer as a single
    numerical value with 3 significant figures", src/run_blinding_l1_l5.py), so by the rule the
    project states for itself -- pad where the prompt asked for a precision -- padding is
    legitimate here and the sweep is the arm that is out of step. The default is therefore the
    map's convention, because a figure drawn on the map's scale has to be counted on the map's
    rule; --no-pad reproduces the sweep's numbers for anyone checking the table against it.

    BOTH WERE RUN, and the difference is worth knowing before either number is quoted. hit3 moves
    a long way -- esol/opus5/L5 reads 6.7% padded and 28.6% unpadded, because the unpadded rule
    throws away 92 of its 120 pairs and the ones it keeps are the ones the model answered to
    three figures, which are disproportionately the recited ones. ONE verdict moves, and it moves
    for lack of power rather than lack of effect: esol/opus5/L5 falls to `untestable` unpadded
    (m1 = 14, one short of MIN_COND) while reading 71% on R12. Flag counts are 10 of 12 at L1 and
    3 of 12 at L5 padded, against 10 and 2 unpadded. Every other cell agrees.

    One further mismatch, unresolved and small: freesolv/opus5/L5 reads hit3 = 17.9% here against
    21.4% in blinding_sweep_t1024.csv on the same 28 pairs, i.e. one hit. The sweep compares
    `sig_round` strings at three figures; `nested_matches` requires the agreement to nest, and
    the two differ on a rounding boundary (9.99 against 10.0 agrees at three figures and not at
    one). The map's counting is the nested one, so that is what is used here.
    """
    d = json.load(open(path, encoding="utf-8"))
    T, P = [], []
    for c in d["calls"]:
        try:
            t, v = float(c["truth"]), float(c["value"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (np.isfinite(t) and np.isfinite(v)) or sig_figs(t) < 3:
            continue
        if not pad and sig_figs(v) < 3:
            continue
        T.append(t); P.append(v)
    if not T:
        return None
    T, P = np.array(T), np.array(P)
    m1, m2, m3 = nested_matches(T, P)
    return dict(m1=m1, m2=m2, m3=m3, n_usable=len(T),
                medae=float(np.median(np.abs(P - T))), cost=float(d["meta"].get("cost", np.nan)))


def collect(arm="t1024", pad=True):
    suffix = f"__{arm}" if arm else ""
    rows = []
    for ds in DS_ORDER:
        for tag in PANEL:
            for level in ("L1", "L5"):
                p = os.path.join(BLIND, f"{ds}__{tag}__{level}{suffix}.json")
                if not os.path.exists(p):
                    print(f"  [miss] {os.path.basename(p)}")
                    continue
                c = cell(p, pad=pad)
                if c is None:
                    print(f"  [empty] {os.path.basename(p)}")
                    continue
                rows.append(dict(dataset=ds, tag=tag, level=level, arm=arm or "registry", **c))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="t1024", help="file suffix; '' for the un-suffixed files")
    ap.add_argument("--no-pad", action="store_true",
                    help="require 3 s.f. in the PREDICTION too, as analyze_blinding_sweep does")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    C.load_label_floor()
    out_path = a.out or os.path.join(RES, "blinding_map%s.csv" % ("_nopad" if a.no_pad else ""))
    d = collect(a.arm, pad=not a.no_pad)
    if d.empty:
        sys.exit("no blinding cells found under results/blinding/")

    # `classify` wants the model name and a couple of columns the budget arm carries. Supplying
    # them here rather than making them optional there keeps the classifier one code path.
    names = {"opus5": "Claude Opus 5", "grok45": "Grok 4.5", "sol": "GPT-5.6 sol",
             "kimik3": "Kimi K3"}
    d["model"] = d.tag.map(names)
    d["hit3"] = 100.0 * d.m3 / d.n_usable.clip(lower=1)
    d["R12"] = 100.0 * d.m2 / d.m1.clip(lower=1)
    d["R23"] = 100.0 * d.m3 / d.m2.clip(lower=1)

    out = C.classify(d, gate="rung", floor="label")
    out["sig_min_q"] = out[["q_R12", "q_R23"]].min(axis=1, skipna=True)
    out.to_csv(out_path, index=False)

    print(f"\n  {len(out)} cells -> {os.path.relpath(out_path, ROOT)}")
    print(f"  BH family: {int(out.testable.sum())} R12 tests + {int(out.deep.sum())} R23 tests "
          f"= {int(out.testable.sum() + out.deep.sum())}\n")
    show = ["dataset", "tag", "level", "n_usable", "m1", "m2", "m3", "hit3", "R12", "floor_R12",
            "R23", "floor_R23", "q_R12", "q_R23", "verdict"]
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(out[show].round(3).to_string(index=False))
    print("\n  flagged: " + ", ".join(
        f"{r.dataset}/{r.tag}/{r.level}" for _, r in out.iterrows()
        if r.verdict == "contaminated") or "  flagged: none")


if __name__ == "__main__":
    main()
