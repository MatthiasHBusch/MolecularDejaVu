"""What a thinking budget buys, measured paired.

The paper's original default-vs-budget comparison was verdict-level only, and had to be: the
zero-shot screen used a different prompt, a different molecule sample and a different n, so its
rates were not commensurable with the arm's. This compares the two arms that ARE commensurable --
same 500 molecules, same prompt, same detector, same panel -- and differ in one thing:

    reg     each endpoint's lowest reasoning setting  (`none`, `minimal`, `max_tokens:128`)
    t1024   a 1,024-token thinking budget

so a difference between them is a deliberation effect and nothing else. Both hit3 rates come from
the same prompt, so unlike the cross-study comparison they can be subtracted.

    python src/analyze_reasoning_delta.py
"""
import argparse, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FLAG = ("heavy", "partial", "trace")
# The benchmarks the minimum-reasoning arm covers. AqSolDB and QM7 were added last -- they are
# the two contaminated benchmarks the arm was missing -- so anything absent from this list is
# silently dropped from the SUMMARY while still sitting in the CSV, which is how a 9-benchmark
# pairing printed as 7.
# All twelve benchmarks since 25 Aug 2026: BACE, Caco-2 and PPBR were the three the
# minimum-reasoning arm had not been run on, and they now have their 66 cells.
ORDER = ["freesolv", "esol", "ld50", "lipophilicity", "aqsoldb", "caco2", "bace", "ppbr",
         "qm7", "qm8", "antiviral", "boilingpoint"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(RES, "budget_3sig_v3.csv"))
    ap.add_argument("--out", default=os.path.join(RES, "reasoning_delta.csv"))
    args = ap.parse_args()

    d = pd.read_csv(args.src)
    if "arm" not in d.columns:
        sys.exit("no `arm` column -- re-run analyze_budget.py so both arms are in one file")
    cols = ["hit3", "hit3_all", "R12", "R23", "m1", "m2", "m3", "n_usable", "regime", "verdict",
            "spearman", "medae", "think", "cost", "floor_R12", "floor_R23"]
    a = d[d.arm == "reg"].set_index(["dataset", "tag"])
    b = d[d.arm == "t1024"].set_index(["dataset", "tag"])
    keys = a.index.intersection(b.index)
    if not len(keys):
        sys.exit("no paired cells")
    J = pd.concat([a.loc[keys, cols].add_suffix("_lo"), b.loc[keys, cols].add_suffix("_hi")],
                  axis=1)
    J["model"] = b.loc[keys, "model"]
    J["d_hit3"] = J.hit3_hi - J.hit3_lo
    J["d_R23"] = J.R23_hi - J.R23_lo
    J["d_rho"] = J.spearman_hi - J.spearman_lo
    J["d_medae"] = J.medae_hi - J.medae_lo
    J["flag_lo"] = J.regime_lo.isin(FLAG)
    J["flag_hi"] = J.regime_hi.isin(FLAG)
    J["crossed"] = J.flag_hi & ~J.flag_lo
    J.reset_index().to_csv(args.out, index=False)

    order = [k for k in ORDER if k in J.index.get_level_values(0)]
    print(f"{len(J)} paired cells, {len(order)} benchmarks. Same molecules, same prompt; the only "
          f"difference is the thinking budget.\n")
    print(f"{'benchmark':16s}{'flagged @ min':>14}{'flagged @1024':>14}{'crossed':>9}"
          f"{'median hit3':>13}{'-> ':>4}{'':>7}{'median think':>14}")
    for dk in order:
        g = J.loc[dk]
        print(f"  {dk:14s}{int(g.flag_lo.sum()):8d}/{len(g):<5d}{int(g.flag_hi.sum()):8d}/{len(g):<5d}"
              f"{int(g.crossed.sum()):9d}{g.hit3_lo.median():13.2f}{'->':>4}"
              f"{g.hit3_hi.median():7.2f}{g.think_hi.median():14.0f}")
    tot = J
    print(f"\n  {'TOTAL':14s}{int(tot.flag_lo.sum()):8d}/{len(tot):<5d}"
          f"{int(tot.flag_hi.sum()):8d}/{len(tot):<5d}{int(tot.crossed.sum()):9d}")

    print(f"\n\nThe cells that cross from clean to flagged when a budget is allowed "
          f"({int(J.crossed.sum())}):\n")
    print(f"  {'benchmark':13s}{'model':22s}{'hit3 min':>9}{'hit3 1024':>11}"
          f"{'R23 min':>9}{'R23 1024':>10}{'verdict':>10}{'think':>8}")
    for (dk, tag), r in J[J.crossed].sort_values("d_hit3", ascending=False).iterrows():
        print(f"  {dk:13s}{str(r.model)[:21]:22s}{r.hit3_lo:9.2f}{r.hit3_hi:11.2f}"
              f"{r.R23_lo:9.1f}{r.R23_hi:10.1f}{r.regime_hi:>10}{r.think_hi:8.0f}")

    back = J[J.flag_lo & ~J.flag_hi]
    print(f"\n  cells going the other way (flagged at the minimum, clean with a budget): "
          f"{len(back)}")
    for (dk, tag), r in back.iterrows():
        print(f"    {dk:13s}{str(r.model)[:21]:22s}{r.hit3_lo:6.2f} -> {r.hit3_hi:.2f}   "
              f"{r.regime_lo} -> {r.regime_hi}")

    print(f"\n\nWhat else the budget buys, on the same cells:\n")
    for dk in order:
        g = J.loc[dk]
        print(f"  {dk:14s} rho {g.spearman_lo.median():+.3f} -> {g.spearman_hi.median():+.3f}   "
              f"medAE {g.medae_lo.median():.3f} -> {g.medae_hi.median():.3f}   "
              f"hit3 {g.hit3_lo.median():.2f} -> {g.hit3_hi.median():.2f}   "
              f"cost x{g.cost_hi.sum() / max(g.cost_lo.sum(), 1e-9):.0f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
