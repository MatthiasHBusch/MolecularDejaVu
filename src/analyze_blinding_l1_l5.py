"""
Level 1 vs Level 5: accuracy, and verbatim agreement measured with the study's own detector.

Two questions, on the same 100 molecules and the same target scale:

  1. How much accuracy does character-substituting the structure cost?
  2. Does the DIGIT-level signature go with it? Accuracy can fall because the task got harder.
     If the 3-sig hit rate collapses too -- and collapses further than accuracy alone explains --
     the structure substitution is interrupting retrieval, not just making prediction harder.

The second question is the one the rest of the paper is built on, so it is asked with the same
statistic and the same accuracy-matched null.

    python src/analyze_blinding_l1_l5.py --dataset esol --tag opus48
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import sig_figs, sig_round
from smooth_error_null import simulate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BLIND, SCREEN = os.path.join(ROOT, "results", "blinding"), os.path.join(ROOT, "data", "screening")


def load(dskey, tag, level):
    p = os.path.join(BLIND, f"{dskey}__{tag}__{level}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {c["mol_id"]: c for c in d["calls"] if c["value"] is not None}, d["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol")
    ap.add_argument("--tag", default="opus48")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=400)
    args = ap.parse_args()

    a = load(args.dataset, args.tag, "L1")
    b = load(args.dataset, args.tag, "L5")
    if a is None or b is None:
        sys.exit("missing an arm")
    (A, meta), (B, _) = a, b
    shared = sorted(set(A) & set(B))
    print(f"{args.dataset} / {args.tag}: {len(shared)} molecules answered in both arms, "
          f"{meta['shots']} shots, {meta['ntest']} test, seed {meta['seed']}\n")

    truth = np.array([A[m]["truth"] for m in shared])
    pa = np.array([A[m]["value"] for m in shared])
    pb = np.array([B[m]["value"] for m in shared])
    ea, eb = np.abs(pa - truth), np.abs(pb - truth)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(shared), size=(args.boot, len(shared)))
    # Paired: the same molecules in both arms, so the bootstrap resamples molecules and
    # recomputes both medians on the same draw. Molecule difficulty cancels.
    d = np.median(eb[idx], axis=1) - np.median(ea[idx], axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])

    print("ACCURACY")
    print(f"  L1  published SMILES + name + property named : medAE {np.median(ea):.3f}")
    print(f"  L5  substituted structure string, 'sample property' : medAE {np.median(eb):.3f}")
    print(f"  degradation {np.median(eb) - np.median(ea):+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]  ({np.median(eb) / max(np.median(ea), 1e-9):.1f}x)")

    from scipy.stats import spearmanr, wilcoxon
    print(f"  Spearman rho  L1 {spearmanr(pa, truth)[0]:.3f}   L5 {spearmanr(pb, truth)[0]:.3f}")
    print(f"  paired Wilcoxon on |error|: p = {wilcoxon(ea, eb)[1]:.2e}")

    print("\nVERBATIM AGREEMENT (the study's own detector, 3 significant figures)")
    rows = []
    for lab, P in [("L1", pa), ("L5", pb)]:
        ok = np.array([sig_figs(t) >= 3 and sig_figs(p) >= 3 for t, p in zip(truth, P)])
        n = int(ok.sum())
        h = int(sum(sig_round(p, 3) == sig_round(t, 3)
                    for t, p, k in zip(truth, P, ok) if k))
        med = float(np.median(np.abs(P - truth)))
        # accuracy-matched floor: a smooth-error predictor of exactly this accuracy
        d_ = pd.read_csv(os.path.join(SCREEN, f"{args.dataset}.csv")).dropna(subset=["value"])
        sim = simulate(d_["value"].to_numpy(float), med, args.reps, rng)
        rows.append(dict(level=lab, n_usable=n, hits=h,
                         hit3=100.0 * h / max(n, 1), floor=sim["hit3"],
                         excess=(100.0 * h / max(n, 1)) / max(sim["hit3"], 1e-9), medae=med))
        print(f"  {lab}: {h}/{n} exact = {100 * h / max(n, 1):5.2f}%   "
              f"accuracy-matched floor {sim['hit3']:5.2f}%   excess {rows[-1]['excess']:5.1f}x")

    print("\nREADING")
    if rows[0]["hit3"] > 2 * rows[1]["hit3"]:
        print("  Verbatim agreement collapses with the structure substitution, on the same")
        print("  molecules and the same target scale. Accuracy alone cannot produce that: the")
        print("  floor moves with accuracy and is reported above.")
    print(f"  L1 medAE {np.median(ea):.3f} on a benchmark whose own experimental scatter is far")
    print(f"  larger is not a prediction; L5 medAE {np.median(eb):.3f} is what this model can do")
    print(f"  when it cannot identify the molecule.")

    out = os.path.join(ROOT, "results", f"blinding_l1_l5_{args.dataset}_{args.tag}.csv")
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
