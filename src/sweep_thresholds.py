"""Sweep every uncalibrated constant in the current detector and print what it costs.

There are five, and none of them is derived from anything:

    MIN_COND    m1 below which no rung has power           15
    HEAVY_HIT3  the partial/heavy cut, in per cent          10
    ALPHA       the BH level                                0.05
    DEGEN_HIT3  simulated hit3 floor above which the        25
    DEGEN_R12   simulated R12 floor above which the         40
                same-accuracy guard is dropped as circular

The table this writes replaces one built for the released scheme, which swept `heavy gate 2-8x
floor` and `partial gate 1.2-3x floor` -- ratio thresholds that the current scheme does not have.

The controls are the point of the sweep: the positive control must stay saturated and the
recency control must stay clean across any defensible setting, or the map is an artefact of a
constant somebody chose.

    python src/sweep_thresholds.py                 # print
    python src/sweep_thresholds.py --tex           # also write paper/tables/thresholds.tex
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify as C

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "tables")

ARM = "t1024"
POS, NEG = "boilingpoint", "antiviral"


def load():
    raw = pd.read_csv(os.path.join(RES, "budget_3sig_pad.csv"))
    raw = raw[~raw.tag.isin(C.DEFERRED)]
    # QM9 was stopped after four cells on cost and appears in no figure, table or verdict count.
    # Leaving it in made this sweep run over 268 cells while the map it is a sensitivity analysis
    # for has 264, so the baseline printed here did not match the caption quoting it.
    raw = raw[raw.dataset != "qm9"]
    raw = raw.rename(columns={"regime": "regime_old"})
    raw = C.join_smooth(raw, os.path.join(RES, "smooth_error_null_budget.csv"))
    return raw[raw.arm == ARM].copy() if "arm" in raw.columns else raw


FLAGGED = ["heavy", "partial", "trace"]      # regime levels that constitute a flag


def counts(d):
    """Cell counts per class. The heavy/partial/trace split is no longer reported, so the three
    are summed into one flagged count; `regime` still carries them internally."""
    v = d.regime.value_counts()
    pos, neg = d[d.dataset == POS], d[d.dataset == NEG]
    return dict(
        flagged=int(sum(v.get(k, 0) for k in FLAGGED)), clean=int(v.get("clean", 0)),
        nosignal=int(v.get("no-signal", 0)), untestable=int(v.get("untestable", 0)),
        pos=f"{int(pos.regime.isin(FLAGGED).sum())}/{len(pos)}",
        neg=f"{int(neg.regime.isin(FLAGGED).sum())}/{len(neg)}")


def run(raw, **kw):
    """Re-classify with one constant overridden. The constants are module-level in classify."""
    old = {k: getattr(C, k) for k in kw}
    for k, v in kw.items():
        setattr(C, k, v)
    try:
        # floor="label" is the floor the paper's verdicts use. Without it this swept the retired
        # accuracy-matched floor and produced a baseline (82 flagged / 171 clean / 0 no-signal /
        # 15 untestable) that disagreed with the caption quoting it (89 / 164 / 11 / 0).
        return counts(C.classify(raw.copy(), alpha=kw.get("ALPHA", C.ALPHA), gate="rung",
                                 floor="label"))
    finally:
        for k, v in old.items():
            setattr(C, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", action="store_true")
    args = ap.parse_args()
    raw = load()
    C.load_label_floor()          # floor="label" reads this module-level table
    base = run(raw)
    print(f"baseline  {base}\n")

    sweeps = [
        ("$m_1$ floor", "MIN_COND", [5, 10, 15, 25, 50], "%d"),
        # HEAVY_HIT3 is not swept any more: it only ever moved cells between `heavy` and
        # `partial`, a split the scheme no longer reports, and it can neither create nor remove
        # a flag. Sweeping it produced a row of constants and an invariance that was tautological.
        ("$\\alpha$", "ALPHA", [0.01, 0.025, 0.05, 0.10], "%g"),
        # DEGEN_HIT3 and DEGEN_R12 are the thresholds at which the simulated same-accuracy guard
        # was dropped as circular. The label-only floor has no such guard, so both are inert:
        # under floor="label" they moved no cell at any setting. Sweeping them printed two rows
        # of constants and invited the reader to count them as evidence of robustness.
    ]
    rows = []
    for label, name, values, fmt in sweeps:
        got = [(v, run(raw, **{name: v})) for v in values]
        lo, hi = got[0][1], got[-1][1]
        rng = lambda k: (f"{lo[k]}" if lo[k] == hi[k] else f"{lo[k]} $\\to$ {hi[k]}")
        print(f"{label:34s} {name}")
        for v, c in got:
            print(f"   {fmt % v:>6}  flagged {c['flagged']:3d}  clean {c['clean']:3d}  "
                  f"no-signal {c['nosignal']:3d}  untest {c['untestable']:3d}  "
                  f"pos {c['pos']}  neg {c['neg']}")
        posset = sorted({c["pos"] for _, c in got})
        negset = sorted({c["neg"] for _, c in got})
        rows.append(f"{label} & {fmt % values[0]} $\\to$ {fmt % values[-1]} & "
                    f"{rng('flagged')} & {rng('clean')} & {rng('nosignal')} & "
                    f"{' / '.join(posset)} & {' / '.join(negset)} \\\\")
        print()

    if args.tex:
        body = "\n".join([
            "\\begin{tabular}{llrrrcc}", "\\toprule",
            "knob & swept over & flagged & clean & no-signal & pos.\\ ctrl.\\ flagged & "
            "recency ctrl.\\ flagged \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"])
        os.makedirs(OUT, exist_ok=True)
        open(os.path.join(OUT, "thresholds.tex"), "w", encoding="utf-8").write(body + "\n")
        print("wrote", os.path.join(OUT, "thresholds.tex"))


if __name__ == "__main__":
    main()
