"""
Does the retriever's advantage survive when retrieval is disabled?

Reads results/convergence/<dataset>__<tag>__<smiles>__<label>.json, the factorial written by
Run_Convergence.jl: three structure conditions (published / randomised / substituted SMILES)
crossed with four label conditions (true values / affine / non-monotonic / sine relabel).

    python src/analyze_convergence.py --dataset esol

Scoring across a label axis
---------------------------
Each label condition has its own target scale, so a raw median absolute error is not
comparable across them, and inverting the transform back to the original units is not an
option: the sine relabel is many-to-one and has no inverse. Every cell is therefore scored in
its OWN units and reported as

    nMAE = median|pred - target| / MAD(target)

which is scale-free by construction and robust, so a condition whose targets happen to span a
wider range is not credited with a worse model. Spearman rho is reported alongside; it is
scale-free too, and unlike nMAE it is invariant to any monotone relabelling, which makes the
comparison between the affine condition (monotone) and the two non-monotone ones direct.

For the three invertible label conditions the prediction is additionally mapped back to the
original property scale and the median absolute error reported in the benchmark's own units,
because that is the number a reader wants for the `true` and `affine` rows. The sine column
is blank there, by construction rather than by omission.

What the numbers mean
---------------------
An affine relabel is MONOTONIC: a model that recalls a molecule's true value can recover the
map from a few in-context anchors and push its recollection through it. The two non-monotone
relabels cannot be inverted that way but remain learnable in context. So the signature of an
advantage bought by recall is a gap that survives `true` and `affine` and collapses under
`nonmono` and `sine` -- whereas an advantage from chemistry should be roughly flat across the
label axis and decay only along the structure axis.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import load_preds

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONV = os.path.join(ROOT, "results", "convergence")
SCREEN = os.path.join(ROOT, "data", "screening")
REG = os.path.join(ROOT, "src", "registry")

SMILES_ORDER = ["published", "randomised", "substituted"]
LABEL_ORDER = ["true", "affine", "nonmono", "sine"]
LABEL_COL = {"true": "value", "affine": "value_affine",
             "nonmono": "value_nonmono", "sine": "value_sine"}
# The sine relabel is many-to-one, so there is no inverse to score on the original scale.
INVERTIBLE = {"true", "affine", "nonmono"}


def invert(label, pred, truth_df):
    """Map predictions on a relabelled scale back to the benchmark's own units."""
    p = np.asarray(pred, float)
    v = truth_df["value"].to_numpy()
    if label == "true":
        return p
    if label == "affine":
        lo, hi = float(v.min()), float(v.max())
        return hi - p * (hi - lo) / 100.0
    if label == "nonmono":
        # Inverse of prepare_datasets.nonmono_values: equal-count bins reordered by a fixed
        # derangement, order preserved inside each bin. Recover the bin from the output
        # decile, then the position within it.
        k, perm = 5, (2, 4, 1, 0, 3)
        edges = np.quantile(v, np.linspace(0, 1, k + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        inv_perm = {out: b for b, out in enumerate(perm)}
        out = np.empty_like(p)
        for i, val in enumerate(p):
            slot = int(np.clip(val * k / 100.0, 0, k - 1e-9))
            b = inv_perm.get(int(slot), 0)
            u = np.clip(val * k / 100.0 - perm[b], 0.0, 1.0)
            yb = v[(v > edges[b]) & (v <= edges[b + 1])]
            out[i] = yb.min() + u * (yb.max() - yb.min()) if len(yb) else np.nan
        return out
    return np.full_like(p, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol")
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()

    truth = pd.read_csv(os.path.join(SCREEN, f"{args.dataset}.csv")).dropna(subset=["value"])
    models = {m["tag"]: m["name"]
              for m in json.load(open(os.path.join(REG, "models.json")))["models"]}
    luts = {lab: {str(k).strip(): float(x)
                  for k, x in zip(truth["mol_id"], truth[col]) if pd.notna(x)}
            for lab, col in LABEL_COL.items() if col in truth.columns}
    mads = {lab: float(np.median(np.abs(np.array(list(l.values())) -
                                        np.median(list(l.values())))))
            for lab, l in luts.items()}

    from scipy.stats import spearmanr
    per_mol, rows = {}, []
    for p in sorted(glob.glob(os.path.join(CONV, f"{args.dataset}__*.json"))):
        parts = os.path.basename(p)[:-len(".json")].split("__")
        if len(parts) != 4:
            print(f"  skip {os.path.basename(p)}: not a <ds>__<tag>__<smiles>__<label> file")
            continue
        _, tag, smi_lvl, lab_lvl = parts
        lut = luts.get(lab_lvl)
        if lut is None:
            print(f"  skip {os.path.basename(p)}: no {LABEL_COL.get(lab_lvl)} column")
            continue

        recs = [(mol, lut[mol], float(np.mean(vals)))
                for mol, vals in load_preds(p).items() if mol in lut]
        if len(recs) < 10:
            continue
        mols = [r[0] for r in recs]
        T = np.array([r[1] for r in recs])
        P = np.array([r[2] for r in recs])
        err = np.abs(P - T) / max(mads.get(lab_lvl, 1.0), 1e-9)
        per_mol[(tag, smi_lvl, lab_lvl)] = dict(zip(mols, err))

        native = np.nan
        if lab_lvl in INVERTIBLE:
            Pi = invert(lab_lvl, P, truth)
            Ti = np.array([luts["true"][m] for m in mols])
            native = float(np.nanmedian(np.abs(Pi - Ti)))
        rows.append(dict(model=models.get(tag, tag), tag=tag, smiles=smi_lvl, label=lab_lvl,
                         n=len(T), nmae=float(np.median(err)),
                         spearman=float(spearmanr(P, T)[0]), medae_native=native))

    if not rows:
        sys.exit(f"no convergence results for {args.dataset} in {CONV}")
    R = pd.DataFrame(rows)
    R["so"] = R.smiles.apply(lambda s: SMILES_ORDER.index(s) if s in SMILES_ORDER else 99)
    R["lo"] = R.label.apply(lambda s: LABEL_ORDER.index(s) if s in LABEL_ORDER else 99)
    R = R.sort_values(["tag", "so", "lo"]).drop(columns=["so", "lo"])
    print(R.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    R.to_csv(os.path.join(ROOT, "results", f"convergence_{args.dataset}.csv"), index=False)

    tags = sorted(R.tag.unique())
    if len(tags) != 2:
        print(f"\n({len(tags)} models present; the gap analysis needs exactly two)")
        return
    a, b = tags
    rng = np.random.default_rng(0)
    print(f"\nGap in normalised error, {models.get(b, b)} minus {models.get(a, a)}.")
    print("Positive = the second model is worse. The pairing is over test molecules, which "
          "both models\nsee identically, so molecule difficulty cancels.\n")
    print(f"{'structure':14s}{'label':10s}{'nMAE ' + a:>16s}{'nMAE ' + b:>16s}"
          f"{'gap':>9}{'95% CI':>20}")
    for smi_lvl in SMILES_ORDER:
        for lab_lvl in LABEL_ORDER:
            ea = per_mol.get((a, smi_lvl, lab_lvl))
            eb = per_mol.get((b, smi_lvl, lab_lvl))
            if not ea or not eb:
                continue
            common = sorted(set(ea) & set(eb))
            if len(common) < 10:
                continue
            va = np.array([ea[m] for m in common])
            vb = np.array([eb[m] for m in common])
            idx = rng.integers(0, len(common), size=(args.boot, len(common)))
            boot = np.median(vb[idx], axis=1) - np.median(va[idx], axis=1)
            ci = np.percentile(boot, [2.5, 97.5])
            star = "" if ci[0] <= 0 <= ci[1] else "  *"
            print(f"{smi_lvl:14s}{lab_lvl:10s}{np.median(va):>16.3f}{np.median(vb):>16.3f}"
                  f"{np.median(vb) - np.median(va):>9.3f}"
                  f"{f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>20}{star}")
    print("\n* = paired-bootstrap CI excludes zero. A gap that is significant under `true` and "
          "`affine`\nand not under `nonmono` / `sine` is the signature of an advantage bought "
          "by recall:\nthe monotone relabel is invertible from in-context anchors, the "
          "non-monotone ones are not.")


if __name__ == "__main__":
    main()
