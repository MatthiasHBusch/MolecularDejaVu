"""
Does blocking structure recognition interrupt retrieval, and does the effect scale with how
much was memorised in the first place?

Reads every results/blinding/<ds>__<tag>__L{1,5}.json cell and reports, per (benchmark, model):

    L1  IUPAC name + published SMILES, property named, TRUE values
    L5  character-substituted structure string only, "sample property", TRUE values

Same molecules, same target scale, one fixed split per cell; only the structure representation
differs. Two readouts per cell -- median absolute error (paired bootstrap over molecules) and
3-sig verbatim agreement against a floor simulated at each arm's own accuracy.

The dose-response is the point of running four models whose memorisation spans 0.9% to 42%.
Three heavy models all collapsing is consistent with "L5 is simply a harder task". A drop whose
SIZE tracks the zero-shot memorisation of the cell is not.

    python src/analyze_blinding_sweep.py
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import sig_figs, sig_round
from smooth_error_null import simulate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, BLIND, SCREEN = (os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "blinding"),
                      os.path.join(ROOT, "data", "screening"))


def arm(path):
    d = json.load(open(path))
    return {c["mol_id"]: c for c in d["calls"] if c["value"] is not None}, d["meta"]


def suffix(a):
    """File-name suffix for a reasoning arm. '' is the original sweep, which ran every cell at
    its registry setting and wrote no tag; 't1024' is the controlled-budget re-run."""
    return "" if a == "legacy" else f"__{a}"


def arms_on_disk():
    out = set()
    for p in glob.glob(os.path.join(BLIND, "*__L1*.json")):
        parts = os.path.basename(p)[:-5].split("__")
        out.add("legacy" if len(parts) == 3 else parts[3])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--out", default="")
    # Never pooled across arms by default. The two are the same cells at two deliberation doses,
    # and averaging them would report a rate no cell was ever measured at.
    ap.add_argument("--arm", default="legacy",
                    help="'legacy' = the original sweep at each model's registry setting; "
                         "'t1024' = the controlled-budget re-run. Never pooled.")
    args = ap.parse_args()
    if not args.out:
        args.out = os.path.join(RES, f"blinding_sweep{'' if args.arm == 'legacy' else '_' + args.arm}.csv")
    have = arms_on_disk()
    if args.arm not in have:
        sys.exit(f"no cells for arm '{args.arm}'; on disk: {', '.join(have) or 'none'}")
    sfx = suffix(args.arm)

    # THE DOSE-RESPONSE PREDICTOR. This was the zero-shot screen, which is retired: it measured
    # what the minimum-reasoning arm measures, on an older pipeline. The minimum-reasoning arm is
    # the replacement and is itself a zero-shot measurement -- every endpoint at the lowest
    # reasoning setting it permits, on the map's own molecules.
    #
    # The choice matters and is reported in the appendix rather than hidden: over the twelve
    # cells the L5/L1 error ratio correlates with the minimum-reasoning rate at rho = +0.60
    # (p = 0.041) and with the controlled-budget rate at rho = +0.48 (p = 0.118). The retired
    # screen gave +0.62 (p = 0.031). The minimum arm is used because it is the same quantity the
    # screen was meant to be; the map's value is reported beside it.
    S = pd.read_csv(os.path.join(RES, "budget_3sig_v3.csv"))
    S = S[S.arm == "reg"] if "arm" in S.columns else S
    zs = S.set_index(["dataset", "tag"]).hit3.to_dict()
    name = S.set_index("tag").model.to_dict()

    truths = {}
    rng = np.random.default_rng(0)
    rows = []
    cells = sorted({tuple(os.path.basename(p)[:-5].split("__")[:2])
                    for p in glob.glob(os.path.join(BLIND, f"*__L1{sfx}.json"))
                    if len(os.path.basename(p)[:-5].split("__")) == (3 if not sfx else 4)})
    print(f"arm '{args.arm}': {len(cells)} L1 cells on disk")
    for dskey, tag in cells:
        p1 = os.path.join(BLIND, f"{dskey}__{tag}__L1{sfx}.json")
        p5 = os.path.join(BLIND, f"{dskey}__{tag}__L5{sfx}.json")
        if not (os.path.exists(p1) and os.path.exists(p5)):
            continue
        (A, m1), (B, _) = arm(p1), arm(p5)
        shared = sorted(set(A) & set(B))
        if len(shared) < 30:
            continue
        if dskey not in truths:
            d = pd.read_csv(os.path.join(SCREEN, f"{dskey}.csv")).dropna(subset=["value"])
            truths[dskey] = d["value"].to_numpy(float)

        t = np.array([A[m]["truth"] for m in shared])
        pa = np.array([A[m]["value"] for m in shared])
        pb = np.array([B[m]["value"] for m in shared])
        ea, eb = np.abs(pa - t), np.abs(pb - t)
        idx = rng.integers(0, len(shared), size=(args.boot, len(shared)))
        dboot = np.median(eb[idx], axis=1) - np.median(ea[idx], axis=1)
        lo, hi = np.percentile(dboot, [2.5, 97.5])

        rec = dict(dataset=dskey, tag=tag, model=name.get(tag, tag), n=len(shared),
                   arm=args.arm, reasoning=m1.get("reasoning", ""), shots=m1.get("shots"),
                   zs_hit3=zs.get((dskey, tag), np.nan),
                   medae_L1=float(np.median(ea)), medae_L5=float(np.median(eb)),
                   d_medae=float(np.median(eb) - np.median(ea)), d_lo=lo, d_hi=hi,
                   ratio=float(np.median(eb) / max(np.median(ea), 1e-9)))
        for lab, P in (("L1", pa), ("L5", pb)):
            ok = np.array([sig_figs(x) >= 3 and sig_figs(y) >= 3 for x, y in zip(t, P)])
            n = int(ok.sum())
            h = int(sum(sig_round(y, 3) == sig_round(x, 3)
                        for x, y, k in zip(t, P, ok) if k))
            med = float(np.median(np.abs(P - t)))
            sim = simulate(truths[dskey], med, args.reps, rng)
            rec[f"usable_{lab}"] = n
            rec[f"hit3_{lab}"] = 100.0 * h / max(n, 1)
            rec[f"floor_{lab}"] = sim["hit3"] if sim else np.nan
        rows.append(rec)
        print(f"  {dskey:10s} {tag:12s} n={len(shared):3d}  "
              f"medAE {rec['medae_L1']:.3f} -> {rec['medae_L5']:.3f}  "
              f"hit3 {rec['hit3_L1']:5.1f} -> {rec['hit3_L5']:5.1f}", flush=True)

    if not rows:
        sys.exit("no complete L1/L5 pairs")
    R = pd.DataFrame(rows).sort_values(["dataset", "zs_hit3"], ascending=[True, False])
    R.round(4).to_csv(args.out, index=False)

    print("\n" + "=" * 104)
    print("L1 -> L5: blocking structure recognition, same molecules and same target scale")
    print("=" * 104)
    print(f"{'benchmark':11s}{'model':22s}{'zs':>6}{'medAE L1':>10}{'medAE L5':>10}"
          f"{'x':>6}{'  degradation [95% CI]':>26}{'hit3 L1->L5':>14}")
    for _, r in R.iterrows():
        star = "" if r.d_lo <= 0 <= r.d_hi else " *"
        print(f"{r.dataset:11s}{r.model[:21]:22s}{r.zs_hit3:6.1f}{r.medae_L1:10.3f}"
              f"{r.medae_L5:10.3f}{r.ratio:6.1f}"
              f"{f'{r.d_medae:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]{star}':>26}"
              f"{f'{r.hit3_L1:.1f} -> {r.hit3_L5:.1f}':>14}")

    print("\n" + "=" * 104)
    print("DOSE-RESPONSE: does the size of the drop track how much was memorised?")
    print("=" * 104)
    from scipy.stats import spearmanr
    ok = R.dropna(subset=["zs_hit3"])
    if len(ok) >= 5:
        for lab, col in (("degradation in medAE", "d_medae"),
                         ("ratio medAE L5/L1", "ratio"),
                         ("verbatim lost (hit3 L1 - L5)", None)):
            y = (ok.hit3_L1 - ok.hit3_L5) if col is None else ok[col]
            rho, p = spearmanr(ok.zs_hit3, y)
            print(f"  zero-shot hit3  vs  {lab:32s} rho = {rho:+.2f}  (p = {p:.4f}, n = {len(ok)})")
        print("\n  A positive correlation means the interruption removes more where more was")
        print("  memorised -- which 'L5 is just harder' does not predict.")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
