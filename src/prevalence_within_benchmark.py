"""
Are the molecules a model reproduces verbatim more prevalent on the open web than the ones it
misses -- WITHIN the same benchmark?

`corpus_prevalence.py` answers the cross-benchmark version: benchmarks whose molecules are more
common are recalled more (rho = 0.98 over eight points). That is suggestive, but it is eight
points, both controls anchor the ends, and the y-variable is a map we ourselves describe as
benchmark-specifically biased.

The within-benchmark version is the strong test and it is free. Fix the benchmark, fix the
model, split its molecules into the ones it reproduced to three significant figures and the ones
it did not, and ask whether the two groups differ in corpus prevalence. Benchmark identity,
value distribution, molecule size and the model are all held constant by construction; only
molecule-level exposure varies.

Prediction under the contamination hypothesis: hit molecules are more prevalent.
Prediction if the agreement is reconstruction from primary literature: prevalence of the SMILES
string itself need not differ, because what is recalled is the measurement, not the string.

    python src/prevalence_within_benchmark.py --dataset esol --tag opus5 --n 80
    python src/prevalence_within_benchmark.py --dataset ld50 --tag gem31pro --n 80

The cell is read from the controlled map (arm t1024), the run the paper reports, so the split
into reproduced and missed molecules is the same one the map's verdict rests on.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import sig_figs, sig_round
from analyze_budget import cells
import corpus_prevalence as CP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, SCREEN = os.path.join(ROOT, "results"), os.path.join(ROOT, "data", "screening")
OUT = os.path.join(RES, "prevalence_within.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol")
    ap.add_argument("--tag", default="opus5")
    ap.add_argument("--arm", default="t1024")
    ap.add_argument("--index", default="v4_dolma-v1_7_llama")
    ap.add_argument("--n", type=int, default=80, help="molecules per group")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(SCREEN, f"{args.dataset}.csv")).dropna(subset=["value", "smiles"])
    lut = {str(k).strip(): float(v) for k, v in zip(df["mol_id"], df["value"])}
    smi = {str(k).strip(): str(s).strip() for k, s in zip(df["mol_id"], df["smiles"])}

    preds = None
    for dk, tag, _arm, cell_preds, _meta in cells(arm=args.arm):
        if dk == args.dataset and tag == args.tag:
            preds = cell_preds
            break
    if preds is None:
        sys.exit(f"no {args.dataset}/{args.tag} cell in arm {args.arm} under results/budget/")

    hit, miss = [], []
    for mol, vals in preds.items():
        t = lut.get(mol)
        s = smi.get(mol, "")
        # Short SMILES match essentially every document in an n-gram index, so they carry no
        # information either way and are excluded from both groups rather than one.
        if t is None or len(s) < 8 or sig_figs(t) < 3:
            continue
        v3 = [v for v in vals if sig_figs(v) >= 3]
        if not v3:
            continue
        (hit if any(sig_round(v, 3) == sig_round(t, 3) for v in v3) else miss).append(mol)

    rng = np.random.default_rng(args.seed)
    hit = list(rng.permutation(hit))[:args.n]
    miss = list(rng.permutation(miss))[:args.n]
    print(f"{args.dataset}/{args.tag}: {len(hit)} reproduced, {len(miss)} missed "
          f"(sampled from {len(preds)} molecules); index {args.index}\n")
    if len(hit) < 15:
        sys.exit("too few reproduced molecules for a within-benchmark comparison")

    rows = []
    for grp, mols in [("reproduced", hit), ("missed", miss)]:
        for i, mol in enumerate(mols, 1):
            c = CP.count(args.index, smi[mol])
            rows.append(dict(dataset=args.dataset, tag=args.tag, index=args.index,
                             group=grp, mol_id=mol, smiles=smi[mol], count=c))
            time.sleep(0.25)
            if i % 20 == 0:
                print(f"  {grp}: {i}/{len(mols)}", flush=True)

    R = pd.DataFrame(rows)
    R["present"] = R["count"].fillna(0) > 0
    hdr = os.path.exists(OUT)
    R.to_csv(OUT, mode="a", header=not hdr, index=False)

    a = R[R.group == "missed"]["present"].to_numpy(float)
    b = R[R.group == "reproduced"]["present"].to_numpy(float)
    d = b.mean() - a.mean()
    ia = rng.integers(0, len(a), size=(args.boot, len(a)))
    ib = rng.integers(0, len(b), size=(args.boot, len(b)))
    boot = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\npresent in the corpus at all:")
    print(f"  reproduced {100 * b.mean():5.1f}%  ({int(b.sum())}/{len(b)})")
    print(f"  missed     {100 * a.mean():5.1f}%  ({int(a.sum())}/{len(a)})")
    print(f"  difference {100 * d:+5.1f} points   95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]"
          f"{'  *' if not (lo <= 0 <= hi) else ''}")
    med = R.groupby("group")["count"].median()
    print(f"\nmedian occurrence count: reproduced {med.get('reproduced', np.nan):.0f}, "
          f"missed {med.get('missed', np.nan):.0f}")
    print(f"\nSaved (appended) {OUT}")


if __name__ == "__main__":
    main()
