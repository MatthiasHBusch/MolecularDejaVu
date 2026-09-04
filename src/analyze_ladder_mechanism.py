"""
Why does the reasoning gate open wide on some benchmarks and not at all on others?

The controlled ladder shows verbatim agreement rising with the thinking budget on LD50 and not
on Lipophilicity. That is a fact about two benchmarks; this asks what property of a benchmark
predicts it, across all of them.

The framing that organises the numbers is that a large gate needs TWO things at once:

    (a) something stored     -- the model must actually hold the value, or no budget will help;
    (b) a forward pass that cannot reach it -- if a single pass already produces the value,
        the budget has nothing left to buy.

That predicts a non-monotone pattern rather than "more famous benchmark, bigger gate". The
positive control fails (b): reference boiling points come out at zero thinking tokens, so the
curve is flat at the ceiling. A clean benchmark fails (a): nothing is stored, so the budget
buys only a better estimate. The gate is widest in between -- where the value is in the weights
but not reachable in one pass.

Five candidate predictors are measured per (benchmark, model) and reported against the gate:

    recall_map      the benchmark's recall in the main screening matrix -- is anything stored?
    hit3_at_zero    verbatim agreement with no deliberation -- can one pass already reach it?
    rho_at_zero     structure-property skill with no deliberation
    prevalence      % of the benchmark's molecules present in public web crawls
    depth4          of the exact 3-sig hits, the share that are also exact at 4 -- whether the
                    model reproduces the published FILE or merely the right value
    hedging         drop in the share of answers carrying 3 significant figures, zero -> max

    python src/analyze_ladder_mechanism.py
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import sig_round, sig_figs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, SCREEN = os.path.join(ROOT, "results"), os.path.join(ROOT, "data", "screening")
LADDER = os.path.join(RES, "ladder")


def truth_lut(dskey):
    df = pd.read_csv(os.path.join(SCREEN, f"{dskey}.csv")).dropna(subset=["value"])
    return {str(k).strip(): float(v) for k, v in zip(df["mol_id"], df["value"])}


def cell_stats(path, lut):
    d = json.load(open(path))
    calls = [c for c in d["calls"] if c["value"] is not None]
    if not calls:
        return None
    T, P = [], []
    n3 = d4 = 0
    coarse = 0
    for c in calls:
        t = lut.get(c["mol_id"])
        if t is None:
            continue
        v = float(c["value"])
        T.append(t); P.append(v)
        coarse += sig_figs(v) < 3
        if sig_figs(t) >= 3 and sig_figs(v) >= 3 and sig_round(v, 3) == sig_round(t, 3):
            n3 += 1
            if sig_figs(t) >= 4 and sig_round(v, 4) == sig_round(t, 4):
                d4 += 1
    if len(T) < 10:
        return None
    T, P = np.array(T), np.array(P)
    from scipy.stats import spearmanr
    return dict(level=d["meta"]["level"],
                reasoning=float(np.median([c["reasoning_tokens"] for c in d["calls"]])),
                n=len(T), n_hit3=n3,
                # Of the exact 3-sig hits, how many carry the published fourth digit too?
                # Coincidence alone would put this near 10%.
                depth4=100.0 * d4 / n3 if n3 else np.nan,
                hit3=100.0 * n3 / len(T),
                sig3_share=100.0 * (1 - coarse / len(T)),
                rho=float(spearmanr(P, T)[0]),
                medae=float(np.median(np.abs(P - T))),
                # A model with no retrieval and no skill falls back on a generic prior, which
                # is narrower than the truth. The ratio makes that visible.
                spread=float(np.std(P) / np.std(T)) if np.std(T) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "ladder_mechanism.csv"))
    args = ap.parse_args()

    # Context from the controlled map and the corpus sweep. This used to read the zero-shot
    # screen, which is retired (results/_archive_zeroshot/); the controlled map answers the same
    # question -- "does this benchmark have anything stored" -- on the reported panel.
    S = pd.read_csv(os.path.join(RES, "budget_3sig_v3.csv"))
    S = S[S.arm == "t1024"] if "arm" in S.columns else S
    recall_map = S.groupby("dataset").apply(
        lambda g: 100.0 * (g.verdict == "contaminated").sum()
        / max((g.verdict != "untestable").sum(), 1), include_groups=False)
    maxhit = S[S.verdict != "untestable"].groupby("dataset").hit3.max()
    prev = pd.Series(dtype=float)
    f = os.path.join(RES, "corpus_prevalence.csv")
    if os.path.exists(f):
        prev = pd.read_csv(f).groupby("dataset").smiles_rate.max()

    luts, rows = {}, []
    for p in sorted(glob.glob(os.path.join(LADDER, "*.json"))):
        ds, tag, _ = os.path.basename(p)[:-5].split("__")
        if ds not in luts:
            luts[ds] = truth_lut(ds)
        st = cell_stats(p, luts[ds])
        if st:
            rows.append(dict(dataset=ds, tag=tag, **st))
    if not rows:
        sys.exit("no ladder cells")
    C = pd.DataFrame(rows)

    out = []
    for (ds, tag), g in C.groupby(["dataset", "tag"]):
        g = g.sort_values("reasoning")
        lo, hi = g.iloc[0], g.iloc[-1]
        if hi.reasoning <= lo.reasoning:
            continue
        peak = g.loc[g.hit3.idxmax()]
        out.append(dict(
            dataset=ds, tag=tag,
            recall_map=float(recall_map.get(ds, np.nan)),
            maxhit_map=float(maxhit.get(ds, np.nan)),
            prevalence=float(prev.get(ds, np.nan)),
            tokens_lo=lo.reasoning, tokens_hi=hi.reasoning,
            hit3_zero=lo.hit3, hit3_peak=peak.hit3, gate=peak.hit3 - lo.hit3,
            gate_ratio=peak.hit3 / max(lo.hit3, 0.3),
            rho_zero=lo.rho, rho_peak=peak.rho,
            medae_zero=lo.medae, medae_peak=peak.medae,
            spread_zero=lo.spread, spread_peak=peak.spread,
            depth4_peak=peak.depth4, n_hit3_peak=peak.n_hit3,
            sig3_zero=lo.sig3_share, sig3_peak=peak.sig3_share,
            hedging=lo.sig3_share - peak.sig3_share))
    M = pd.DataFrame(out).sort_values("gate", ascending=False)
    M.round(3).to_csv(args.out, index=False)

    print("Gate size per (benchmark, model): verbatim agreement at the best level minus at zero"
          "\nthinking tokens. 'stored?' is the benchmark's recall in the main matrix; "
          "'one pass?' is\nwhat a zero-token forward pass already achieves.\n")
    print(f"{'benchmark':14s}{'model':16s}{'stored?':>9}{'one pass?':>10}{'gate':>8}"
          f"{'rho 0->pk':>12}{'medAE 0->pk':>16}{'depth4%':>9}{'hedge':>7}")
    for _, r in M.iterrows():
        print(f"{r.dataset:14s}{r.tag:16s}{r.maxhit_map:8.1f}%{r.hit3_zero:9.1f}%"
              f"{r.gate:+8.1f}{f'{r.rho_zero:.2f}->{r.rho_peak:.2f}':>12}"
              f"{f'{r.medae_zero:.3f}->{r.medae_peak:.3f}':>16}"
              f"{r.depth4_peak:8.0f}%{r.hedging:+7.0f}")

    # --- the two-condition claim, tested rather than asserted -------------------------
    from scipy.stats import spearmanr
    print("\n" + "=" * 78)
    print("CONDITION (a): is anything stored? Benchmarks split by whether the main matrix\n"
          "finds any recall on them at all.")
    print("=" * 78)
    ok = M.dropna(subset=["maxhit_map"]).copy()
    # "Recalled" means the screening flagged at least one model. The positive control is kept
    # apart: everyone recalls it, so it tests the ceiling rather than the gate.
    ok["group"] = np.where(ok.dataset == "boilingpoint", "positive control",
                           np.where(ok.recall_map > 0, "recalled", "clean"))
    g = ok.groupby("group").agg(cells=("gate", "size"), gate_med=("gate", "median"),
                                gate_max=("gate", "max"), rho0_med=("rho_zero", "median"),
                                hit0_med=("hit3_zero", "median"))
    print(g.round(2).to_string())
    print("\nA benchmark with nothing stored has no gate to open, whatever its ladder does to\n"
          "accuracy. That is the control, and it is the half of the claim that can fail.")

    print("\n" + "=" * 78)
    print("CONDITION (b): can one forward pass already get there? Among the RECALLED\n"
          "benchmarks only -- where a gate is possible at all.")
    print("=" * 78)
    rec = ok[ok.group == "recalled"]
    if len(rec) >= 4:
        for lab, col in [("structure-property skill at zero tokens (rho)", "rho_zero"),
                         ("verbatim agreement at zero tokens", "hit3_zero"),
                         ("benchmark recall in the main matrix", "maxhit_map"),
                         ("corpus prevalence of the molecules", "prevalence")]:
            s = rec.dropna(subset=[col])
            if len(s) >= 4 and s[col].nunique() > 2:
                r, p = spearmanr(s[col], s.gate)
                print(f"  gate vs {lab:46s} rho = {r:+.2f}  (p = {p:.3f}, n = {len(s)})")
        print("\n  A negative correlation with zero-token skill is the prediction: the gate is\n"
              "  widest where the property is LEAST computable, because there retrieval is the\n"
              "  only route to an accurate answer and a single pass cannot take it.")
    round_source_test()
    print(f"\nSaved {args.out}")


def round_source_test(dskey="ld50", tol=0.005):
    """Is the LD50 recall table-reading, or recall-then-convert?

    TDC publishes LD50 as -log10(mol/kg), converted from the literature's mg/kg with the molar
    mass -- and the sources are round numbers: 2.343 back-converts to 1147 mg/kg, 2.330 to
    500.2, 2.210 to 1949. So there are two routes to the published value:

      (i)  read the TDC row;
      (ii) recall the round mg/kg fact, look up the molar mass, and do the arithmetic.

    Route (ii) is exactly the sort of thing a thinking budget could buy and a single forward
    pass could not, which would make the reasoning gate a story about ARITHMETIC rather than
    retrieval. It predicts that molecules whose source value is round should be hit far more
    often than the rest, because only for those does the conversion land on the published
    number. Route (i) predicts no split at all.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors
    except ImportError:
        print("\n(rdkit unavailable -- skipping the round-source test)")
        return
    RDLogger.DisableLog("rdApp.*")
    f = os.path.join(SCREEN, f"{dskey}.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f).dropna(subset=["value", "smiles"])
    lut, roundness = {}, {}
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(str(r["smiles"]))
        if m is None:
            continue
        k, v = str(r["mol_id"]).strip(), float(r["value"])
        lut[k] = v
        mg = 10 ** (-v) * Descriptors.MolWt(m) * 1000
        roundness[k] = min(abs(mg - float("%.*g" % (n, mg))) / mg for n in (1, 2))

    print("\n" + "=" * 78)
    print(f"ROUND-SOURCE TEST on {dskey}: table-reading, or recall-then-convert?")
    print("=" * 78)
    print(f"{'model':16s}{'level':9s}{'hit3 round-source':>20}{'hit3 other':>13}"
          f"{'n round/other':>16}")
    for p in sorted(glob.glob(os.path.join(LADDER, f"{dskey}__*__*.json"))):
        d = json.load(open(p))
        if d["meta"]["level"] not in ("none", "medium", "high"):
            continue
        grp = {"round": [0, 0], "other": [0, 0]}
        for c in d["calls"]:
            v, k = c["value"], c["mol_id"]
            t = lut.get(k)
            if v is None or t is None or k not in roundness:
                continue
            if sig_figs(t) < 3 or sig_figs(v) < 3:
                continue
            g = "round" if roundness[k] < tol else "other"
            grp[g][1] += 1
            grp[g][0] += sig_round(v, 3) == sig_round(t, 3)
        if grp["round"][1] < 10 or grp["other"][1] < 10:
            continue
        a = 100 * grp["round"][0] / grp["round"][1]
        b = 100 * grp["other"][0] / grp["other"][1]
        counts = "%d/%d" % (grp["round"][1], grp["other"][1])
        print("%-16s%-9s%19.1f%%%12.1f%%%16s"
              % (d["meta"]["tag"], d["meta"]["level"], a, b, counts))
    print("\nEqual rates mean the models are reproducing the published table, not recomputing")
    print("it from a remembered fact: the budget buys retrieval, not arithmetic.")


if __name__ == "__main__":
    main()
