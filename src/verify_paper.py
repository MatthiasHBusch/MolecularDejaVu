"""Recompute every number the paper states that can be recomputed, and print it in the order
the paper states it.

This exists because the paper has been through three classification schemes and two arms, and
hand-carried numbers do not survive that. Run it, diff it against the manuscript by eye, and any
disagreement is either a stale number or a claim that was never measured.

    python src/verify_paper.py            # everything
    python src/verify_paper.py --sec 3    # one section
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DROP_MODELS = ("kimik26", "glm5", "glm52", "nemotron3u", "qwen35", "dsv4pro", "opus48", "opus45")
DROP_DATASETS = ("qm9",)
# The severity split is retired; a cell is flagged or it is not.
FLAG = "contaminated"
# generalization.csv predates the verdict column and still carries the old regime labels, so the
# checks that read it need the legacy set. Everything else keys on `verdict`.
FL = ("heavy", "partial", "trace")


def hdr(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}")


def sub(s):
    print(f"\n--- {s}")


def load(arm="t1024", drop=True):
    d = pd.read_csv(os.path.join(RES, "budget_3sig_v3.csv"))
    if "arm" in d.columns:
        d = d[d.arm == arm]
    if drop:
        d = d[~d.tag.isin(DROP_MODELS) & ~d.dataset.isin(DROP_DATASETS)]
    return d.copy()


def sec2():
    hdr("SECTION 2 -- METHODS: panel size, call count, molecule counts")
    A = load()
    raw = pd.read_csv(os.path.join(RES, "budget_3sig_pad.csv"))
    sub("panel")
    print(f"  models in the map      {A.tag.nunique()}")
    print(f"  benchmarks in the map  {A.dataset.nunique()}  ({sorted(A.dataset.unique())})")
    print(f"  cells                  {len(A)}")
    print(f"  molecules per cell     min {A.n_mol.min():.0f}  median {A.n_mol.median():.0f}  "
          f"max {A.n_mol.max():.0f}")
    if "n_calls" in raw.columns:
        print(f"  calls (raw, all arms)  {int(raw.n_calls.sum()):,}")
    sub("m1 range, the quantity the R12 conditioning argument quotes as 200-330 of 500")
    t = A[A.verdict != "untestable"]
    print(f"  m1 over cells with a verdict: min {t.m1.min():.0f}  q25 {t.m1.quantile(.25):.0f}  "
          f"median {t.m1.median():.0f}  q75 {t.m1.quantile(.75):.0f}  max {t.m1.max():.0f}")
    real = t[~t.dataset.isin(("boilingpoint", "antiviral"))]
    print(f"  same, real benchmarks only: median {real.m1.median():.0f}  "
          f"IQR {real.m1.quantile(.25):.0f}-{real.m1.quantile(.75):.0f}")


def sec3_map():
    hdr("SECTION 3.1 -- THE MAP")
    A = load()
    sub("verdict counts (main text: 38 heavy, 31 partial, 13 trace, 171 clean, 11 no verdict)")
    print("  ", dict(A.verdict.value_counts()))
    print(f"   total {len(A)}")
    sub("per benchmark (table: cells with a verdict / flagged / of those heavy / strongest)")
    for d in ["ld50", "esol", "aqsoldb", "freesolv", "qm7", "bace", "caco2", "ppbr",
              "lipophilicity", "qm8", "boilingpoint", "antiviral"]:
        g = A[A.dataset == d]
        if not len(g):
            continue
        v = g[g.verdict != "untestable"]
        f = g[g.verdict == FLAG]
        top = f.sort_values("hit3", ascending=False).head(1)
        s = (f"{top.hit3.iloc[0]:.1f}% ({top.model.iloc[0]})" if len(top) else "---")
        print(f"  {d:14s} verdict {len(v):3d}  flagged {len(f):3d}  strongest {s}")
    sub("headline cells")
    for dk, tg in (("esol", "opus5"), ("freesolv", "opus5"), ("ld50", "gem36flash"),
                   ("aqsoldb", "gem35flash"), ("qm7", "opus5")):
        r = A[(A.dataset == dk) & (A.tag == tg)]
        if len(r):
            r = r.iloc[0]
            print(f"  {dk:10s}/{r.model:18s} hit3 {r.hit3:6.2f}  floor {r.floor_hit3:5.2f} "
                  f"(x{r.excess_hit3:5.2f})  R12 {r.R12:5.1f}/{r.floor_R12:5.1f}  "
                  f"R23 {r.R23:5.1f}/{r.floor_R23:5.1f}  {r.verdict}")
    sub("flagged LD50 cells: hit3 range and R23 range (text: 2.6-16.9%, retaining 32-70%)")
    g = A[(A.dataset == "ld50") & (A.verdict == FLAG)]
    print(f"  hit3 {g.hit3.min():.2f}-{g.hit3.max():.2f}   R23 {g.R23.min():.1f}-{g.R23.max():.1f}"
          f"   floor_hit3 median {g.floor_hit3.median():.2f}")
    sub("AqSolDB shape (text: 13 of 15 partial, led by Gemini 3.5 Flash at 7.9%)")
    g = A[A.dataset == "aqsoldb"]
    print("  ", dict(g.verdict.value_counts()),
          f" top {g.hit3.max():.2f} ({g.sort_values('hit3').model.iloc[-1]})",
          f" floor there {g.sort_values('hit3').floor_hit3.iloc[-1]:.2f}")
    sub("the three ADMET benchmarks that are clean (text: 0 of 22 each)")
    for d in ("bace", "caco2", "ppbr"):
        g = A[A.dataset == d]
        print(f"  {d:8s} flagged {int((g.verdict == FLAG).sum())} of {len(g)}")


def sec3_nulls():
    hdr("SECTION 3.1.1 -- THE NULLS")
    # Was the zero-shot screen, now archived. Fig. 3 draws this comparison on the
    # minimum-reasoning arm, so the check follows it there.
    S = pd.read_csv(os.path.join(RES, "budget_3sig_v3.csv"))
    S = S[S.arm == "reg"] if "arm" in S.columns else S
    S = S[~S.tag.isin(DROP_MODELS) & ~S.dataset.isin(DROP_DATASETS)]
    sub("ESOL, global shuffle vs the floor in force (text: 15 vs 5)")
    e = S[S.dataset == "esol"]
    glob = int((e.p_hit3_binom < 0.05).sum()) if "p_hit3_binom" in e.columns else -1
    print(f"  cells {len(e)}   significant on the global shuffle (raw p<0.05) {glob}"
          f"   flagged by the scheme in force {int((e.verdict == FLAG).sum())}")
    sub("floor inflation, pair vs molecule level, by regime")
    for a, b, lab in (("mp_chance_hit", "mb_chance_hit", "hit rate"),
                      ("mp_chance_deep", "mb_chance_deep", "R23")):
        r = S[a] / S[b].replace(0, np.nan)
        out = " ".join(f"{v} {r[S.verdict == v].dropna().median():.2f}x"
                       for v in (FLAG, "clean"))
        print(f"  {lab:8s} {out}")
    sub("global floor range on ESOL (text: 0.17-0.31%)")
    print(f"  chance_hit3 {e.chance_hit3.min():.2f}-{e.chance_hit3.max():.2f}   "
          f"mb_chance_hit3 {e.mb_chance_hit3.min():.2f}-{e.mb_chance_hit3.max():.2f}")
    sub("R23 excess over the simulated floor, by regime (fig3c)")
    M = pd.read_csv(os.path.join(RES, "smooth_error_null_budget.csv"))
    M = M[M.arm == "reg"] if "arm" in M.columns else M
    cur = S.set_index(["dataset", "tag"]).verdict
    m = M[M.dataset != "boilingpoint"].dropna(subset=["R23_excess_smooth"]).copy()
    m["verdict"] = [cur.get((d, t), np.nan) for d, t in zip(m.dataset, m.tag)]
    m = m.dropna(subset=["verdict"])
    for reg in ("clean", FLAG):
        x = m[m.verdict == reg].R23_excess_smooth
        if len(x):
            print(f"  {reg:8s} n={len(x):3d} median {x.median():.2f} "
                  f"IQR {x.quantile(.25):.2f}-{x.quantile(.75):.2f} "
                  f"range {x.min():.1f}-{x.max():.1f}")


def sec3_ladder():
    hdr("SECTION 3.2 -- THE DELIBERATION GATE")
    D = pd.read_csv(os.path.join(RES, "reasoning_delta.csv"))
    sub("paired arm (text: 38 of 154 at minimum, 66 at 1024, 28 crossings, none returning)")
    print(f"  cells {len(D)}   flagged@min {int(D.flag_lo.sum())}   flagged@1024 "
          f"{int(D.flag_hi.sum())}   crossed {int(((~D.flag_lo) & D.flag_hi).sum())}   "
          f"returned {int((D.flag_lo & (~D.flag_hi)).sum())}")
    sub("per benchmark")
    for dk, g in D.groupby("dataset"):
        print(f"  {dk:14s} {int(g.flag_lo.sum()):2d} -> {int(g.flag_hi.sum()):2d}   "
              f"median hit3 {g.hit3_lo.median():6.2f} -> {g.hit3_hi.median():6.2f}   "
              f"medAE {g.medae_lo.median():.3f} -> {g.medae_hi.median():.3f} "
              f"({100 * (g.medae_hi.median() / g.medae_lo.median() - 1):+.0f}%)   "
              f"rho {g.spearman_lo.median():+.3f} -> {g.spearman_hi.median():+.3f}")
    sub("crossings at zero emitted thinking tokens")
    z = D[(~D.flag_lo) & D.flag_hi & (D.think_hi == 0)]
    print(f"  {len(z)}: " + ", ".join(f"{r.dataset}/{r.model}" for _, r in z.iterrows()))
    sub("LD50 clean-model count at each setting (text: 19 of 22 clean at min, 6 at 1024)")
    g = D[D.dataset == "ld50"]
    print(f"  clean@min {int((~g.flag_lo).sum())} of {len(g)}   "
          f"clean@1024 {int((~g.flag_hi).sum())} of {len(g)}")
    L = os.path.join(RES, "ladder_summary.csv")
    if os.path.exists(L):
        Ld = pd.read_csv(L)
        sub("ladder: hit3 range at the top rung per benchmark")
        for dk, g in Ld.groupby("dataset"):
            top = g.sort_values("reasoning_med").groupby("tag").tail(1)
            bot = g.sort_values("reasoning_med").groupby("tag").head(1)
            print(f"  {dk:14s} n_cells {g.tag.nunique():2d}  hit3 low-rung "
                  f"{bot.hit3.min():5.2f}-{bot.hit3.max():5.2f}  high-rung "
                  f"{top.hit3.min():5.2f}-{top.hit3.max():5.2f}  max tokens "
                  f"{g.reasoning_med.max():.0f}")


def sec3_bench():
    hdr("SECTION 3.3 -- CONTAMINATION IS NOT COMPETENCE")
    G = pd.read_csv(os.path.join(RES, "generalization.csv"))
    G = G[~G.tag.isin(DROP_MODELS) & ~G.dataset.isin(DROP_DATASETS)]
    sub("recall against skill, across models")
    for dk in ("freesolv", "esol", "aqsoldb", "ld50"):
        g = G[G.dataset == dk]
        if not len(g):
            continue
        cl, fl = g[g.regime == "clean"], g[g.regime.isin(FL)]
        print(f"  {dk:10s} n={len(g):2d}  sp(hit3,rho) {spearmanr(g.hit3, g.rho).statistic:+.2f}"
              f"  sp(hit3,r_clip) {spearmanr(g.hit3, g.r_clip).statistic:+.2f}"
              f"  sp(hit3,mae_clip) {spearmanr(g.hit3, g.mae_clip).statistic:+.2f}")
        print(f"             best clean MAE {cl.mae_clip.min():.2f} ({cl.sort_values('mae_clip').model.iloc[0]})"
              f"   contaminated worse than it: {int((fl.mae_clip > cl.mae_clip.min()).sum())} of {len(fl)}")
        print(f"             clean rho {cl.rho.min():.2f}-{cl.rho.max():.2f}   "
              f"contaminated rho {fl.rho.min():.2f}-{fl.rho.max():.2f}")
    sub("FreeSolv leaderboard, all molecules vs residual (text: Opus 5 0.44 -> 0.75)")
    g = G[G.dataset == "freesolv"].sort_values("mae_clip")
    for _, r in g.head(6).iterrows():
        print(f"  {r.model:20s} {r.regime:8s} MAE {r.mae_clip:.3f} -> resid {r.mae_clip_res:.3f}"
              f"   r {r.r_clip:.3f} -> {r.r_clip_res:.3f}")
    cl = g[g.regime == "clean"]
    print(f"  best clean model overall: {cl.sort_values('mae_clip').model.iloc[0]} "
          f"{cl.mae_clip.min():.3f}")
    sub("rank movement under the residual scoring")
    for dk, gg in G.groupby("dataset"):
        a = gg.sort_values("mae_clip").tag.tolist()
        b = gg.sort_values("mae_clip_res").tag.tolist()
        moved = [(t, a.index(t) + 1, b.index(t) + 1) for t in a if abs(a.index(t) - b.index(t)) >= 3]
        pr = gg.sort_values("r_clip", ascending=False).tag.tolist()
        prr = gg.sort_values("r_clip_res", ascending=False).tag.tolist()
        movedp = [(t, pr.index(t) + 1, prr.index(t) + 1) for t in pr
                  if abs(pr.index(t) - prr.index(t)) >= 3]
        print(f"  {dk:10s} by MAE: {moved}")
        print(f"  {' ':10s} by Pearson: {movedp}")


def sec3_mit():
    hdr("SECTION 3.4 -- INTERRUPTING RETRIEVAL")
    # The controlled arm where it exists. blinding_sweep.csv is the legacy run at each
    # endpoint's own default reasoning, on a different molecule draw and an older substitution
    # table; the paper reports the t1024 arm, so the checker has to read that one.
    p = os.path.join(RES, "blinding_sweep_t1024.csv")
    if not os.path.exists(p):
        p = os.path.join(RES, "blinding_sweep.csv")
    if os.path.exists(p):
        B = pd.read_csv(p)
        sub("L1 -> L5, every cell in the file")
        cols = [c for c in ("dataset", "model", "tag", "hit3_L1", "hit3_L5", "medae_L1",
                            "medae_L5", "degradation", "deg_lo", "deg_hi") if c in B.columns]
        print(B[cols].to_string(index=False))
        if {"hit3_L1", "medae_L1", "medae_L5"} <= set(B.columns):
            B = B.copy()
            B["ratio"] = B.medae_L5 / B.medae_L1
            d = B.dropna(subset=["hit3_L1", "ratio"])
            print(f"\n  ratio vs in-run recall: rho {spearmanr(d.hit3_L1, d.ratio).statistic:+.3f} "
                  f"p {spearmanr(d.hit3_L1, d.ratio).pvalue:.3f} n {len(d)}")
    r = os.path.join(RES, "randomization_control.csv")
    if os.path.exists(r):
        R = pd.read_csv(r)
        sub("randomised SMILES: survival of the canonical rate")
        if {"hit3_published", "hit3_randomised"} <= set(R.columns):
            R = R.copy()
            R["surv"] = 100 * R.hit3_randomised / R.hit3_published.replace(0, np.nan)
            print(f"  median survival over {len(R)} cells: {R.surv.median():.0f}%")
            big = R[R.hit3_published > 1]
            print(f"  cells above 1%: n={len(big)}  median survival {big.surv.median():.0f}%")
            print(big[[c for c in ("model", "hit3_published", "hit3_randomised", "surv")
                       if c in big.columns]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="all")
    a = ap.parse_args()
    jobs = {"2": sec2, "3.1": sec3_map, "3.1.1": sec3_nulls, "3.2": sec3_ladder,
            "3.3": sec3_bench, "3.4": sec3_mit}
    for k, f in jobs.items():
        if a.sec in ("all", k) or k.startswith(a.sec + "."):
            try:
                f()
            except Exception as e:
                print(f"\n!! {k} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
