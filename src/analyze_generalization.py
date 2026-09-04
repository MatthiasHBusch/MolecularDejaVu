"""Does recall buy accuracy? Per benchmark, rank skill against verbatim recall.

WHY THIS EXISTS. The contamination map says which cells reproduce published digits. It does not
say whether that is what makes them look good, and the two are not the same question. A cell can
be heavily contaminated and still rank the benchmark worse than a model with no recall at all --
in which case its leaderboard position is not bought by retrieval, it is just a weak model that
happens to have seen the file. The reverse also occurs: a clean cell can top the benchmark, which
is the only configuration in which a benchmark score means what it is supposed to mean.

This is a benchmarking table, so it carries the measures a benchmark is actually reported with,
all on the SAME 500 molecules and all in two versions -- over every molecule, and over only the
molecules the model did NOT reproduce to three significant figures (`_res`):

    r          Pearson correlation. The headline here: the question is how well the model
               predicts the VALUE, and Pearson is the measure that answers it. Spearman is
               carried alongside because the map used it and because it is robust to the tails.
    rho        Spearman rank correlation
    mae, rmse, medae   error on the benchmark's own scale
    hit3       the share reproduced to three significant figures -- what the map measures

The `_res` columns are the point of the file: they are what the benchmark would have reported had
the recalled molecules never been in it, so `r - r_res` is the part of the score that rides on
retrieval. It is a lower bound on the damage -- a molecule matched to two figures is contaminated
too and still counts as "residual" here.

Two caveats, stated because they bound what the number can carry:

  * `rho_res` conditions on the model's own misses, and the recalled molecules are not a random
    subset -- they are the famous, small, well-documented ones (see RESEARCH_NOTES 2.9), which
    are also the ones any chemist finds easy. Removing them therefore removes easy molecules as
    well as recalled ones, so a drop in `rho_res` overstates the retrieval contribution to some
    unknown degree. The comparison that IS clean is between models on the same residual set.
  * Spearman over ~400 molecules has a standard error of roughly 0.05 at rho = 0.9, so
    differences below ~0.1 are not differences. Bootstrap intervals are computed per cell.

    python src/analyze_generalization.py
    python src/analyze_generalization.py --datasets esol,freesolv,ld50 --boot 2000
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_budget import cells, truth_lut
from memodetect import sig_key_vec, sig_figs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, REG = os.path.join(ROOT, "results"), os.path.join(ROOT, "src", "registry")

DEFAULT_DS = "freesolv,esol,ld50,aqsoldb"


def cell_frame(lut, preds, level=3):
    """One row per molecule: truth, the model's median prediction, and whether it is an n-sig hit.

    Predictions are collapsed to the median per molecule before anything else, because the unit
    of independence is the molecule -- the same reason the null permutes molecules.
    """
    rows = []
    for mol, plist in preds.items():
        tv = lut.get(mol)
        if tv is None or not len(plist) or sig_figs(tv) < level:
            continue
        pv = float(np.median(plist))
        rows.append((tv, pv))
    if len(rows) < 30:
        return None
    d = pd.DataFrame(rows, columns=["truth", "pred"])
    mt, et = sig_key_vec(d.truth.to_numpy(), level)
    mp, ep = sig_key_vec(d.pred.to_numpy(), level)
    d["hit"] = (mt == mp) & (et == ep)
    return d


def boot_r(x, y, n=1000, seed=0):
    """Bootstrap interval for Pearson r, resampling molecules."""
    if len(x) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(y[i])) < 3 or len(np.unique(x[i])) < 3:
            continue
        v.append(pearsonr(x[i], y[i]).statistic)
    if not v:
        return np.nan, np.nan
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def scores(d, lo=None, hi=None):
    """Pearson, Spearman and the three error measures for one molecule table.

    Pearson and RMSE are the right measures for "does it predict the value" and are also the two
    a single wild answer destroys: Grok 4.3 on ESOL returns predictions that put its RMSE at 211
    and its r at 0.05 while its Spearman is 0.66. A benchmark reporting only r would rank it last
    for four bad parses.

    So each is reported twice. The clipped variant confines predictions to the range the
    benchmark's own targets span -- a prediction outside it is not a prediction of this property,
    it is a failure to answer the question -- and `n_wild` counts how many were moved. Neither
    version is the "true" one: the raw column is what a leaderboard would print, the clipped
    column is what the model would score if its parse failures were treated as such, and the gap
    between them is a property of the model worth seeing.
    """
    if len(d) < 30:
        return {k: np.nan for k in ("r", "rho", "mae", "rmse", "medae", "r_clip", "mae_clip",
                                    "rmse_clip", "n_wild")}
    t, p = d.truth.to_numpy(), d.pred.to_numpy()
    e = p - t
    lo = t.min() if lo is None else lo
    hi = t.max() if hi is None else hi
    pc = np.clip(p, lo, hi)
    ec = pc - t
    return dict(r=pearsonr(t, p).statistic, rho=spearmanr(t, p).statistic,
                mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                medae=float(np.median(np.abs(e))),
                r_clip=pearsonr(t, pc).statistic if len(np.unique(pc)) > 2 else np.nan,
                mae_clip=float(np.mean(np.abs(ec))),
                rmse_clip=float(np.sqrt(np.mean(ec ** 2))),
                n_wild=int(np.sum((p < lo) | (p > hi))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=DEFAULT_DS)
    ap.add_argument("--arm", default="t1024")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(RES, "generalization.csv"))
    args = ap.parse_args()

    keys = args.datasets.split(",")
    # The verdict table now holds both arms, so it must be filtered to the one being scored --
    # otherwise (dataset, tag) is not unique and every lookup returns two rows.
    verdicts = pd.read_csv(os.path.join(RES, "budget_3sig_v3.csv"))
    if "arm" in verdicts.columns:
        verdicts = verdicts[verdicts.arm == args.arm]
    verdicts = verdicts.set_index(["dataset", "tag"]).sort_index()
    models = {m["tag"]: m for m in json.load(open(os.path.join(REG, "models.json")))["models"]}

    luts, rows = {}, []
    for dk, tag, arm, preds, info in cells(args.arm):
        if dk not in keys:
            continue
        luts.setdefault(dk, truth_lut(dk))
        d = cell_frame(luts[dk], preds)
        if d is None or (dk, tag) not in verdicts.index:
            continue
        v = verdicts.loc[(dk, tag)]
        res = d[~d.hit]
        # The clip range is the benchmark's, not the cell's, so every model is judged against
        # the same window.
        lo, hi = float(d.truth.min()), float(d.truth.max())
        a, b = scores(d, lo, hi), scores(res, lo, hi)
        blo, bhi = boot_r(res.truth.to_numpy(), res.pred.to_numpy(), args.boot)
        rows.append(dict(
            dataset=dk, tag=tag, model=models.get(tag, {}).get("name", tag),
            n_mol=len(d), n_hit=int(d.hit.sum()), hit3=float(v.hit3), regime=v.regime,
            **a, **{f"{k}_res": v2 for k, v2 in b.items()},
            r_res_lo=blo, r_res_hi=bhi,
            dr=a["r"] - b["r"], dr_clip=a["r_clip"] - b["r_clip"],
            drho=a["rho"] - b["rho"],
            dmae=b["mae"] - a["mae"], drmse=b["rmse"] - a["rmse"]))
    if not rows:
        sys.exit("no cells")
    R = pd.DataFrame(rows).sort_values(["dataset", "r_clip"], ascending=[True, False])
    R.round(4).to_csv(args.out, index=False)

    FL = ("heavy", "partial", "trace")
    for dk, g in R.groupby("dataset"):
        clean, flag = g[g.regime == "clean"], g[g.regime.isin(FL)]
        print(f"\n{'=' * 104}\n{dk.upper()}   {len(g)} models, {int(g.n_mol.median())} molecules\n"
              f"{'=' * 104}")
        for lab, col in (("r (clipped)", "r_clip"), ("r (raw)    ", "r"), ("rho        ", "rho"),
                         ("MAE (clip) ", "mae_clip"), ("RMSE (clip)", "rmse_clip")):
            rp, rs = pearsonr(g.hit3, g[col]), spearmanr(g.hit3, g[col])
            print(f"  hit3 vs {lab} across models:  Pearson {rp.statistic:+.3f} "
                  f"(p = {rp.pvalue:.4f})   Spearman {rs.statistic:+.3f} (p = {rs.pvalue:.4f})")
        if len(clean) and len(flag):
            best = clean.loc[clean.r_clip.idxmax()]
            print(f"\n  r (clipped): clean {clean.r_clip.min():.3f}-{clean.r_clip.max():.3f} "
                  f"(median {clean.r_clip.median():.3f})   "
                  f"contaminated {flag.r_clip.min():.3f}-{flag.r_clip.max():.3f} "
                  f"(median {flag.r_clip.median():.3f})")
            # The claim this file exists to check: contamination is not competence.
            print(f"  contaminated cells predicting WORSE than the best clean model "
                  f"({best.model}, r = {best.r_clip:.3f}): "
                  f"{int((flag.r_clip < best.r_clip).sum())} of {len(flag)}"
                  f";  by MAE: {int((flag.mae_clip > best.mae_clip).sum())} of {len(flag)}")
        print(f"\n  {'model':22s}{'regime':9s}{'hit3':>7}{'r':>7}{'r|no hit':>10}{'dr':>7}"
              f"{'rho':>7}{'MAE':>7}{'RMSE':>7}{'MAE|no hit':>12}{'wild':>6}{'r raw':>8}")
        for _, r in g.iterrows():
            print(f"  {r.model[:21]:22s}{r.regime:9s}{r.hit3:7.2f}{r.r_clip:7.3f}"
                  f"{r.r_clip_res:10.3f}{r.dr_clip:7.3f}{r.rho:7.3f}{r.mae_clip:7.3f}"
                  f"{r.rmse_clip:7.3f}{r.mae_clip_res:12.3f}{r.n_wild:6.0f}{r.r:8.3f}")

    print(f"\n\n{'=' * 104}\nDOES REMOVING THE RECALLED MOLECULES CHANGE THE RANKING?\n{'=' * 104}")
    for dk, g in R.groupby("dataset"):
        a = g.sort_values("r_clip", ascending=False).tag.tolist()
        b = g.sort_values("r_clip_res", ascending=False).tag.tolist()
        moved = [(t, a.index(t) + 1, b.index(t) + 1) for t in a if a.index(t) != b.index(t)]
        big = [m for m in moved if abs(m[1] - m[2]) >= 3]
        print(f"  {dk:12s} {len(moved)} of {len(a)} models change rank, "
              f"{len(big)} by three places or more"
              + (":  " + ", ".join(f"{t} {i}->{j}" for t, i, j in big[:6]) if big else ""))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
