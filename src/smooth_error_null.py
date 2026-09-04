"""
The accuracy-matched null that neither of the other two manages to be.

The bin null matches the *magnitude of the truth*; the residual null matches the model's error
but inherits its point mass at zero. This one constructs a predictor that is, by fiat, both
non-memorising and exactly as accurate as the model under test: a smooth (Laplace) error
distribution centred on the truth, scaled so its median absolute error equals the model's, and
rounded to the same number of significant figures the model emits.

It is a simulation, not a permutation, and that is the point -- it is the only way to ask "what
would an equally accurate predictor that is definitely not reciting achieve?" without drawing
the error distribution from the thing being tested.

Two questions it answers, both of which the released tables get wrong:

1. Is the bin floor conservative? On ESOL and LD50 yes; on FreeSolv's most accurate cells,
   badly not -- which is exactly where the flagship claim lives.
2. Is R23 accuracy-insensitive, as the detector's docstring asserts? No. The claim assumes the
   error density is flat across the 2-sig window, which fails once the median error is small
   relative to that window. R23's floor roughly doubles between medAE 0.6 and 0.025.

    python src/smooth_error_null.py
    python src/smooth_error_null.py --reps 400 --dataset freesolv
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import sig_figs, sig_key_vec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, SCREEN = os.path.join(ROOT, "results"), os.path.join(ROOT, "data", "screening")


def simulate(truths, medae, reps, rng, sig_emit=3):
    """Hit rates and retention for a smooth-error predictor of the stated accuracy.

    Laplace errors: the median absolute deviation of Laplace(0, b) is b*ln(2), so scaling by
    medae/ln(2) reproduces the target median error exactly. Predictions are rounded to
    `sig_emit` significant figures, which is what the models actually emit.
    """
    T = np.asarray(truths, float)
    T = T[np.array([sig_figs(v) >= 3 for v in T])]
    n = len(T)
    if n < 30:
        return None
    kt = [sig_key_vec(T, k) for k in (1, 2, 3)]
    h1 = h2 = h3 = 0
    for _ in range(reps):
        err = rng.laplace(0.0, medae / np.log(2.0), size=n)
        P = T + err
        # emit at a fixed precision, as the models do
        m, e = sig_key_vec(P, sig_emit)
        P = m.astype(float) * 10.0 ** (e - (sig_emit - 1))
        eq = np.ones(n, dtype=bool)
        cnt = []
        for k in (1, 2, 3):
            mt, et = kt[k - 1]
            mp, ep = sig_key_vec(P, k)
            eq = eq & (mt == mp) & (et == ep)
            cnt.append(int(eq.sum()))
        h1 += cnt[0]; h2 += cnt[1]; h3 += cnt[2]
    tot = n * reps
    return dict(n=n, hit1=100 * h1 / tot, hit2=100 * h2 / tot, hit3=100 * h3 / tot,
                R12=100 * h2 / max(h1, 1), R23=100 * h3 / max(h2, 1),
                R13=100 * h3 / max(h1, 1))


def arm_truths(ds, arm):
    """The true values the simulated predictor is scored against.

    For the controlled-budget arm that is the frozen 500-molecule target, not the whole
    benchmark: the floor depends on how the truths are distributed, and the arm's cells were
    all measured on that subset. For the screening it is the whole file, which is what those
    cells sampled from.
    """
    d = pd.read_csv(os.path.join(SCREEN, f"{ds}.csv")).dropna(subset=["value"])
    tgt = os.path.join(RES, "budget", "_target.json")
    if arm == "budget" and os.path.exists(tgt):
        ids = set(json.load(open(tgt))["target"].get(ds, []))
        if ids:
            sub = d[d.mol_id.astype(str).str.strip().isin(ids)]
            if len(sub) >= 30:
                return sub["value"].to_numpy(float)
    return d["value"].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--dataset", default=None)
    # The 'screening' arm is gone: it read results/screening_3sig.csv, which is archived under
    # results/_archive_zeroshot/. Both surviving arms (t1024, reg) live in the budget file.
    ap.add_argument("--arm", default="budget", choices=["budget"],
                    help="'budget' = the controlled-deliberation and minimum-reasoning arms, "
                         "whose truths are the frozen target")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = os.path.join(RES, "budget_3sig_pad.csv")
    out = args.out or os.path.join(RES, "smooth_error_null_budget.csv")

    S = pd.read_csv(src)
    if "study" in S.columns:
        S = S[(S.study == "screening") & (S.variant == "canonical")]
    # NO regime filter. The new classifier decides cells on the 1->2 rung, so a cell the old
    # scheme called `untestable` still needs a floor -- dropping them here is what made the
    # simulated floor unavailable for exactly the cells the 2-figure rung exists to rescue.
    S = S[np.isfinite(S.medae)]
    if args.dataset:
        S = S[S.dataset == args.dataset]
    rng = np.random.default_rng(0)

    truths = {}
    rows = []
    for _, r in S.iterrows():
        ds = r["dataset"]
        if ds not in truths:
            truths[ds] = arm_truths(ds, args.arm)
        sim = simulate(truths[ds], r["medae"], args.reps, rng)
        if sim is None:
            continue
        rows.append(dict(dataset=ds, tag=r["tag"], arm=r.get("arm", "t1024"),
                         model=r["model"], regime=r["regime"],
                         medae=r["medae"], hit3=r["hit3"], hit3_lo=r.get("hit_lo", np.nan),
                         bin_floor=r["mb_chance_hit3"], smooth_floor=sim["hit3"],
                         R12=r["R12"], R12_bin_floor=r["mb_chance_R12"],
                         R12_smooth_floor=sim["R12"],
                         R23=r["R23"], R23_lo=r.get("deep_lo", np.nan),
                         R23_bin_floor=r["mb_chance_R23"], R23_smooth_floor=sim["R23"],
                         R13_smooth_floor=sim["R13"], hit1_smooth_floor=sim["hit1"]))
    R = pd.DataFrame(rows)
    R["excess_bin"] = R.hit3 / R.bin_floor.clip(lower=0.01)
    R["excess_smooth"] = R.hit3 / R.smooth_floor.clip(lower=0.01)
    R["R23_excess_smooth"] = R.R23 / R.R23_smooth_floor.clip(lower=0.01)
    R["R12_excess_smooth"] = R.R12 / R.R12_smooth_floor.clip(lower=0.01)
    R.round(4).to_csv(out, index=False)
    args.out = out

    print("Is the bin floor conservative? hit3 against both floors, flagged cells.\n")
    f = R[R.regime.isin(["heavy", "partial"]) & (R.dataset != "boilingpoint")]
    f = f.sort_values("hit3", ascending=False).head(14)
    print(f"{'cell':30s}{'medAE':>8}{'hit3':>8}{'bin':>8}{'smooth':>9}{'x bin':>8}{'x smooth':>10}")
    for _, r in f.iterrows():
        print(f"{r.dataset + '/' + r.tag:30s}{r.medae:8.3f}{r.hit3:8.2f}{r.bin_floor:8.2f}"
              f"{r.smooth_floor:9.2f}{r.excess_bin:8.1f}{r.excess_smooth:10.1f}")

    print("\n\nIs R23 accuracy-insensitive? Same cells, retention against both floors.\n")
    print(f"{'cell':30s}{'medAE':>8}{'R23':>8}{'CI lo':>8}{'bin':>8}{'smooth':>9}{'x smooth':>10}")
    for _, r in f.iterrows():
        print(f"{r.dataset + '/' + r.tag:30s}{r.medae:8.3f}{r.R23:8.1f}{r.R23_lo:8.1f}"
              f"{r.R23_bin_floor:8.1f}{r.R23_smooth_floor:9.1f}{r.R23_excess_smooth:10.1f}")

    print("\n\nThe accuracy dependence of the R23 floor, one benchmark, accuracy swept.\n")
    for ds in (["freesolv"] if args.dataset is None else [args.dataset]):
        d = pd.read_csv(os.path.join(SCREEN, f"{ds}.csv")).dropna(subset=["value"])
        print(f"  {ds}:  {'medAE':>8}{'hit3 floor':>12}{'R23 floor':>11}")
        for m in (1.0, 0.6, 0.3, 0.1, 0.05, 0.025):
            s = simulate(d["value"].to_numpy(float), m, args.reps, rng)
            print(f"  {'':9}{m:8.3f}{s['hit3']:12.2f}{s['R23']:11.1f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
