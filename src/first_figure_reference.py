"""What Pearson correlation does a predictor reach that knows ONLY the first significant figure?

WHY THIS EXISTS. Section 2.2 argues that the models' correlation with the labels is not high
enough to have raised the second-figure match rate by prediction alone. The argument needs a
yardstick: the correlation of a predictor that gets exactly the first significant figure of every
label right and knows nothing below it. If a model's r stays below that yardstick, whatever
accuracy it has on the second and third figures cannot come from the same skill that produced its
correlation -- there is not enough of it -- and the surplus has to be retrieval.

THE REFERENCE PREDICTOR. For every label y in a cell,

    yhat = sig_round(y, 1) * (1 + U(-0.05, +0.05))              ("pm5")

i.e. the label rounded to one significant figure, with everything below the first figure replaced
by uniform multiplicative noise of +-5 %. The +-5 % is the width of the second-figure window at the
first figure 1 (a value written 1.x is within +-5 % of 1.0); at higher first figures it is wider
than that window, so the reference is a mildly PESSIMISTIC predictor of the first figure. A second
reading of "random below the first figure" is also computed:

    yhat ~ U(window of y)                                          ("window")

where the window is the whole interval of values that round to the same first figure (for a
first figure d x 10^k that is [d - 0.5, d + 0.5) x 10^k, narrower on the low side for d = 1 where
the carry from 9.5 ... 9.99 x 10^(k-1) also lands). It is the more generous reference (it is what a
perfect first-figure predictor with NO information below it does on average), and the two bracket
the yardstick. Each is averaged over 200 draws with a fixed seed.

WHAT IS COMPARED. Pearson r of the model's median prediction per molecule against the label, on
each cell's own molecule set, so the two correlations run over identical pairs. Predictions are
clipped to the benchmark's label range before r is taken (`r_clip` in results/generalization.csv,
the paper's convention: a prediction outside the range the property spans is a parse failure,
not a prediction). One cell -- MiniMax M3 on AqSolDB -- carries a wild answer that moves raw r
from 0.27 to 0.86; the clipped value is the one every table in the paper uses.

Labels on ESOL, AqSolDB and LD50 are logarithms and can be negative. sig_round keeps the sign, so
"+-5 %" is 5 % of the logarithm, not of the underlying concentration. That is the right scale
here -- the significant-figure windows of the detector are defined on the number as reported --
but it means the reference is not a physical error model.

Everything is over the controlled map (arm t1024, 22 models x 12 benchmarks, 264 cells).

OUTCOME LABELS. Each cell is reported as flagged / clean / no signal, read from the `regime`
column of results/budget_3sig_v3.csv as classify.py wrote it. `--flagged-table
paper/tables/flagged_cells.tex` is an override that takes the flagged set from the appendix table
instead (every cell listed there, plus the whole positive control); it exists for the case where
the classified table on disk is older than the released classification, and both agree on the
shipped data. `untestable` cells are `no signal` either way.

    python src/first_figure_reference.py
    python src/first_figure_reference.py --draws 1000 --seed 1
"""
import argparse, json, os, sys, zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_budget import cells, truth_lut
from memodetect import sig_round, sig_key_vec, sig_figs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, REG = os.path.join(ROOT, "results"), os.path.join(ROOT, "src", "registry")
TAB = os.path.join(ROOT, "paper", "tables")
# QM9 was stopped after four cells on cost (EXPERIMENT_LOG) and is not in the 12-benchmark panel;
# the same convention as verify_paper.py / figures_paper.py.
DROP_DATASETS = ("qm9",)


def cell_frame(lut, preds, level=3):
    """One row per molecule: truth and the model's median prediction (same construction as
    analyze_generalization.cell_frame, repeated here so this script needs no scipy)."""
    rows = []
    for mol, plist in preds.items():
        tv = lut.get(mol)
        if tv is None or not len(plist) or sig_figs(tv) < level:
            continue
        rows.append((tv, float(np.median(plist))))
    if len(rows) < 30:
        return None
    d = pd.DataFrame(rows, columns=["truth", "pred"])
    mt, et = sig_key_vec(d.truth.to_numpy(), level)
    mp, ep = sig_key_vec(d.pred.to_numpy(), level)
    d["hit"] = (mt == mp) & (et == ep)
    return d


def first_figure(y):
    """Label rounded to one significant figure, as a float."""
    return np.array([float(sig_round(v, 1)) for v in y])


def window_bounds(f):
    """Bounds of the set of values that round to the one-figure value f (elementwise).

    For |f| = d * 10^k the set is [|f| - 0.5*10^k, |f| + 0.5*10^k), except when d = 1, where the
    lower part comes from the carry (9.5 ... 9.99 * 10^(k-1) also rounds to 1 * 10^k) and is only
    0.05*10^k wide. Returned on the magnitude scale; the sign is re-applied by the caller.
    """
    a = np.abs(np.asarray(f, float))
    k = np.floor(np.log10(np.where(a == 0, 1.0, a)))
    step = 10.0 ** k
    d = np.round(a / step)
    lo = a - np.where(d == 1, 0.05, 0.5) * step
    hi = a + 0.5 * step
    return lo, hi


def pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def reference_r(y, kind, draws, rng, lo, hi):
    """Mean (and sd) Pearson r of the first-figure reference against y, over `draws` draws.

    The reference's own predictions are clipped to the same [lo, hi] as the model's, so the two
    are treated identically.
    """
    f = first_figure(y)
    wlo, whi = window_bounds(f)
    rs = []
    for _ in range(draws):
        if kind == "pm5":
            yhat = f * (1 + rng.uniform(-0.05, 0.05, len(f)))
        else:
            yhat = np.sign(f) * rng.uniform(wlo, whi)
        rs.append(pearson(y, np.clip(yhat, lo, hi)))
    rs = np.array(rs)
    return float(np.nanmean(rs)), float(np.nanstd(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="t1024")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RES, "first_figure_reference.csv"))
    ap.add_argument("--flagged-table", default=None,
                    help="LaTeX table whose first two columns (benchmark, model) list the flagged "
                         "cells outside the positive control; overrides the CSV regime column")
    args = ap.parse_args()

    flagged = None
    if args.flagged_table:
        flagged = set()
        for ln in open(args.flagged_table, encoding="utf-8"):
            c = [x.strip() for x in ln.split(" & ")]
            if len(c) > 4 and c[0] != "benchmark":
                flagged.add((c[0], c[1].replace("\\&", "&")))

    verdicts = pd.read_csv(os.path.join(RES, "budget_3sig_v3.csv"))
    if "arm" in verdicts.columns:
        verdicts = verdicts[verdicts.arm == args.arm]
    verdicts = verdicts.set_index(["dataset", "tag"]).sort_index()
    models = {m["tag"]: m for m in json.load(open(os.path.join(REG, "models.json")))["models"]}
    dsets = {d["key"]: d for d in json.load(open(os.path.join(REG, "datasets.json")))["datasets"]}

    short = {"esol": "ESOL", "freesolv": "FreeSolv", "lipophilicity": "Lipophilicity",
             "bace": "BACE", "aqsoldb": "AqSolDB", "caco2": "Caco-2", "ld50": "LD50",
             "ppbr": "PPBR", "qm7": "QM7", "qm8": "QM8", "antiviral": "Antiviral",
             "boilingpoint": "Boiling pt."}

    def outcome(dk, name, regime):
        if regime == "untestable":
            return "no signal"
        if flagged is None:
            return "flagged" if regime in ("heavy", "partial", "trace") else "clean"
        return "flagged" if dk == "boilingpoint" or (short[dk], name) in flagged else "clean"

    luts, rows = {}, []
    for dk, tag, arm, preds, info in cells(args.arm):
        if dk in DROP_DATASETS or (dk, tag) not in verdicts.index:
            continue                      # not a panel cell
        luts.setdefault(dk, truth_lut(dk))
        d = cell_frame(luts[dk], preds)
        if d is None:
            continue
        v = verdicts.loc[(dk, tag)]
        t, p = d.truth.to_numpy(), d.pred.to_numpy()
        lo, hi = float(t.min()), float(t.max())
        r_clip = pearson(t, np.clip(p, lo, hi))
        r_raw = pearson(t, p)
        # one seed per cell, derived from the global seed, so a cell's reference is reproducible
        # on its own and does not depend on iteration order
        rng = np.random.default_rng([args.seed, zlib.crc32(f"{dk}/{tag}".encode())])
        r5, s5 = reference_r(t, "pm5", args.draws, rng, lo, hi)
        rw, sw = reference_r(t, "window", args.draws, rng, lo, hi)
        name = models.get(tag, {}).get("name", tag)
        rows.append(dict(
            dataset=dk, benchmark=dsets[dk]["name"], ds_class=dsets[dk]["class"], tag=tag,
            model=name, n_mol=len(d),
            hit3=float(v.hit3), outcome=outcome(dk, name, v.regime), r_raw=r_raw, r_clip=r_clip,
            ref_pm5=r5, ref_pm5_sd=s5, ref_window=rw, ref_window_sd=sw,
            beats_pm5=bool(r_clip > r5), beats_window=bool(r_clip > rw)))
    if not rows:
        sys.exit("no cells")
    R = pd.DataFrame(rows).sort_values(["dataset", "r_clip"], ascending=[True, False])
    R.round(4).to_csv(args.out, index=False)
    print("wrote", args.out, f"({len(R)} cells)")

    # ------------------------------------------------------------ per-benchmark summary
    order = ["esol", "freesolv", "ld50", "aqsoldb", "lipophilicity", "bace", "caco2", "ppbr",
             "qm7", "qm8", "antiviral", "boilingpoint"]
    S = []
    for dk in [k for k in order if k in set(R.dataset)]:
        g = R[R.dataset == dk]
        b = g.loc[g.r_clip.idxmax()]
        c = g[g.outcome == "clean"]
        bc = c.loc[c.r_clip.idxmax()] if len(c) else None
        S.append(dict(dataset=dk, benchmark=g.benchmark.iloc[0], n_cells=len(g),
                      best_model=b.model, best_r=b.r_clip, best_hit3=b.hit3, best_outcome=b.outcome,
                      n_clean=len(c),
                      best_clean_model=bc.model if bc is not None else "",
                      best_clean_r=bc.r_clip if bc is not None else np.nan,
                      ref_pm5=g.ref_pm5.median(), ref_window=g.ref_window.median(),
                      n_beats_pm5=int(g.beats_pm5.sum()), n_beats_window=int(g.beats_window.sum()),
                      n_flagged=int((g.outcome == "flagged").sum())))
    S = pd.DataFrame(S)
    S.round(4).to_csv(args.out.replace(".csv", "_summary.csv"), index=False)

    print(f"\n{'benchmark':40s}{'cells':>6}{'best r':>8}  {'best cell':28s}{'hit3':>6}"
          f"{'best clean':>11}{'ref+-5%':>9}{'reach':>7}{'ref win':>9}{'reach':>7}")
    for _, s in S.iterrows():
        print(f"{s.benchmark[:39]:40s}{s.n_cells:6d}{s.best_r:8.3f}  "
              f"{s.best_model[:21] + ' (' + s.best_outcome + ')':28s}{s.best_hit3:6.1f}"
              f"{s.best_clean_r:11.3f}"
              f"{s.ref_pm5:9.3f}{s.n_beats_pm5:4d}/{s.n_cells:<3d}"
              f"{s.ref_window:9.3f}{s.n_beats_window:4d}/{s.n_cells:<3d}")

    # the headline: the four retrieved measured benchmarks
    four = R[R.dataset.isin(["esol", "freesolv", "ld50", "aqsoldb"])]
    print(f"\nretrieved benchmarks (ESOL, FreeSolv, LD50, AqSolDB): {len(four)} cells, "
          f"{int(four.beats_pm5.sum())} above the +-5 % reference, "
          f"{int(four.beats_window.sum())} above the window reference")
    for _, r in four[four.beats_window | four.beats_pm5].iterrows():
        print(f"   {r.benchmark:18s}{r.model:22s} r = {r.r_clip:.3f}  ref +-5% {r.ref_pm5:.3f}  "
              f"window {r.ref_window:.3f}  hit3 {r.hit3:.1f} ({r.outcome})")

    # ------------------------------------------------------------ LaTeX table
    tshort = dict(short, antiviral="Antiviral (recency ctrl.)",
                  boilingpoint="Boiling pt. (positive ctrl.)")

    def tex(s):
        return str(s).replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{tabular}{lrrlrrrrr}", r"\toprule",
             r"benchmark & $r_{\pm5\%}$ & $r_{\mathrm{win}}$ & best cell & \hitthree{} (\%) & "
             r"$r$ & $r$ best clean & \multicolumn{2}{c}{cells above} \\",
             r" & & & & & & & $r_{\pm5\%}$ & $r_{\mathrm{win}}$ \\", r"\midrule"]
    for _, s in S.iterrows():
        bc = f"{s.best_clean_r:.3f}" if s.n_clean else "--"
        lines.append(f"{tshort[s.dataset]} & {s.ref_pm5:.3f} & {s.ref_window:.3f} & "
                     f"{tex(s.best_model)} ({s.best_outcome}) & {s.best_hit3:.1f} & {s.best_r:.3f} & "
                     f"{bc} & {s.n_beats_pm5} & {s.n_beats_window} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(TAB, exist_ok=True)
    tp = os.path.join(TAB, "first_figure_reference.tex")
    open(tp, "w").write("\n".join(lines) + "\n")
    print("wrote", tp)


if __name__ == "__main__":
    main()
