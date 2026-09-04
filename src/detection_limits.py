"""When is contamination detectable at all, and how many molecules would it take?

Every 'clean' verdict in this study is an acceptance of a null, and an acceptance is only worth
as much as the power behind it. This module answers the four questions that turn a negative
result into a statement:

    1. how far above its floor must R12 sit before the 1->2 rung fires?
    2. how far above its floor must R23 sit before the 2->3 rung fires?
    3. how far above its floor must the unconditional first-figure rate sit?
    4. how far above its floor must hit3 sit?

and then the planning question: given that a cell missed, how many molecules would have been
needed for it to have had a chance.

WHAT THE FLOOR IS. `results/label_floor.csv`, columns `mode_R12` / `mode_R23`: the best a
molecule-blind procedure can do, which is to emit the modal continuation of the prefix it has
already matched -- see src/label_floor.py. It is a property of the label column and nothing
else, which is what lets a threshold be quoted per benchmark rather than per cell. The
`floor_*` columns in the same file are the superseded sampling floor and must NOT be used here,
or the thresholds describe a test nobody runs.

WHAT DECIDES SIGNIFICANCE. Each rung is a conditional binomial against that floor, and the
family is corrected with Benjamini-Hochberg per arm. BH is adaptive, so there is no fixed
p-threshold in advance; the operating point actually used by the released classification is
recovered from the run (the largest p that was accepted). It is recovered PER ARM and the
tables are quoted at one arm's -- pooling them across arms let the 12-cell randomised arm, in
which every cell is flagged and the largest accepted p is 0.0004, set the threshold for all of
them, which inflated every required sample size by about a factor of two.

THE ASYMMETRY WORTH KNOWING BEFORE READING THE TABLES. The four tests do not cost the same:

  * R23 conditions on m2, which is a few per cent of the benchmark, so its threshold is high in
    percentage points but it is the only test that responds to a handful of recited molecules.
  * R12 conditions on m1, which is a third to a half of the benchmark, so a two-point excess is
    already significant -- and two points of R12 is also what ordinary accuracy can buy. It is
    the powerful test and the ambiguous one.
  * the unconditional first-figure rate is tested against a floor of 10-76%, since it is
    dominated by how tightly the benchmark's values cluster in magnitude, NOT against 10%.
  * hit3 has a floor of well under one per cent on most benchmarks, so it is significant at a
    handful of molecules -- but it is conditioned on the model's emitted precision, which is
    downstream of whether it is reciting.

    python src/detection_limits.py
    python src/detection_limits.py --alpha 0.05 --power 0.8
"""
import argparse, os, sys

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

# The conditioning counts the tables are quoted at. 15 is the current testability cut; 500 is
# one full block of the controlled arm.
COND = (15, 30, 50, 100, 200, 500)

# Which arm's Benjamini-Hochberg operating point the tables are quoted at. The arms have their
# own families and therefore their own thresholds; this is the arm the paper's negative results
# come from.
ARM_FOR_TABLES = "t1024"


def min_significant(n, p0_pct, alpha):
    """Smallest count k out of n that is significant against Binom(n, p0), and k/n in per cent.

    Exact, one-sided, conditional on the observed n -- the same test the classifier runs, so the
    thresholds are the classifier's own and not an approximation of it.
    """
    p0 = min(max(p0_pct / 100.0, 1e-9), 1 - 1e-9)
    if n < 1:
        return np.nan, np.nan
    k = int(binom.isf(alpha, n, p0))
    while k > 0 and binom.sf(k - 1, n, p0) < alpha:
        k -= 1
    while k <= n and binom.sf(k - 1, n, p0) >= alpha:
        k += 1
    return (k, 100.0 * k / n) if k <= n else (np.nan, np.nan)


def required_n(p0_pct, p1_pct, alpha, power):
    """Trials needed to detect a true rate p1 against a floor p0, one-sided normal approximation.

    Used for planning only. The thresholds above are exact; this is the inverse question and the
    normal form is accurate enough to size an experiment by.
    """
    p0, p1 = p0_pct / 100.0, p1_pct / 100.0
    if p1 <= p0:
        return np.inf
    za, zb = norm.ppf(1 - alpha), norm.ppf(power)
    return float(np.ceil((za * np.sqrt(p0 * (1 - p0)) + zb * np.sqrt(p1 * (1 - p1))) ** 2
                         / (p1 - p0) ** 2))


def bh_operating_point(path, alpha):
    """The largest p-value the released classification actually accepted, per arm.

    BH spends its budget adaptively: with many strong signals it accepts far weaker ones than
    Bonferroni would, and with none it is close to Bonferroni. Quoting the nominal 0.05 as if it
    were the operating point would understate every threshold in this file.
    """
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    out = {}
    for arm, g in (d.groupby("arm") if "arm" in d.columns else [(None, d)]):
        ps = []
        for pcol, qcol in (("p_R12_binom", "q_R12"), ("p_R23_binom", "q_R23")):
            if pcol in g.columns and qcol in g.columns:
                s = g[(g[qcol] < alpha) & np.isfinite(g[pcol])][pcol]
                ps.extend(s.tolist())
        out[arm] = max(ps) if ps else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--out", default=os.path.join(RES, "detection_limits.csv"))
    args = ap.parse_args()

    fl = os.path.join(RES, "label_floor.csv")
    if not os.path.exists(fl):
        sys.exit(f"need {fl}; run  python src/label_floor.py")
    L = pd.read_csv(fl).set_index("dataset")

    ops = bh_operating_point(os.path.join(RES, "budget_3sig_v3.csv"), args.alpha)
    print("=" * 100)
    print("THE OPERATING POINT")
    print("=" * 100)
    print("Benjamini-Hochberg is adaptive, so the p-value that actually had to be beaten is not")
    print(f"{args.alpha}. Recovered from the released classification, per arm:\n")
    for arm, p in ops.items():
        print(f"    arm {str(arm):8s}  largest accepted p = {p:.5f}"
              f"   ({'nothing was accepted' if not np.isfinite(p) else f'{args.alpha/p:.0f}x stricter than alpha'})")
    # The operating point belongs to ONE arm and must not be pooled across them. Taking the
    # minimum over arms pulled in the 12-cell randomised arm, where every cell is flagged and the
    # largest accepted p is 0.0004; quoting the tables there made every required sample size
    # about twice too large. The headline arm is the controlled 1,024-token one, which is what
    # the study's negative results are read off.
    eff = ops.get(ARM_FOR_TABLES, np.nan)
    if not np.isfinite(eff):
        eff = args.alpha
        print(f"\n  ! no accepted test in arm '{ARM_FOR_TABLES}'; falling back to alpha")
    print(f"\n  tables below are quoted at the operating point of arm '{ARM_FOR_TABLES}',")
    print(f"  p = {eff:.5f}. Other arms differ; plan against the arm you are planning for.\n")

    rows = []
    for lvl, aa in (("alpha", args.alpha), ("BH", eff)):
        for ds, r in L.iterrows():
            # The floors MUST be the ones the classifier uses, or the thresholds describe a test
            # nobody runs. `mode_*` is the molecule-blind supremum; `floor_*` is the superseded
            # sampling floor and is carried in the CSV for the appendix comparison only.
            for stat, floor, cond in (("R12", r.mode_R12, "m1"), ("R23", r.mode_R23, "m2"),
                                      ("hit1", r.floor_hit1, "n"), ("hit3", r.floor_hit3, "n")):
                for n in COND:
                    k, pct = min_significant(n, floor, aa)
                    rows.append(dict(dataset=ds, level=lvl, alpha=aa, stat=stat, cond=cond,
                                     n=n, floor=floor, k_min=k, pct_min=pct,
                                     excess=pct - floor if np.isfinite(pct) else np.nan))
    D = pd.DataFrame(rows)
    D.round(4).to_csv(args.out, index=False)

    for stat, cond, label in (("R12", "m1", "1. THE 1->2 RUNG: how high must R12 be?"),
                              ("R23", "m2", "2. THE 2->3 RUNG: how high must R23 be?"),
                              ("hit1", "n", "3. THE UNCONDITIONAL FIRST FIGURE: how high must m1/n be?"),
                              ("hit3", "n", "4. THE 3-FIGURE HIT RATE: how high must hit3 be?")):
        print("=" * 100)
        print(label)
        print("=" * 100)
        print(f"smallest value that is significant, at the BH operating point p < {eff:.5f}\n")
        hdr = "".join(f"{c:>9d}" for c in COND)
        print(f"{'benchmark':16s}{'floor':>8s}{hdr}   ({cond} = )")
        sub = D[(D.stat == stat) & (D.level == "BH")]
        for ds, g in sub.groupby("dataset"):
            g = g.set_index("n")
            vals = "".join(f"{g.pct_min.get(c, np.nan):9.1f}" for c in COND)
            print(f"{ds:16s}{g.floor.iloc[0]:7.2f}%{vals}")
        print()

    planning(L, eff, args)


def planning(L, eff, args):
    """What a 'clean' verdict actually excludes, and what more molecules would buy."""
    src = os.path.join(RES, "budget_3sig_v3.csv")
    if not os.path.exists(src):
        print(f"skip planning: {src} not found")
        return
    d = pd.read_csv(src)
    d = d[d.arm == "t1024"] if "arm" in d.columns else d
    clean = d[d.regime == "clean"]

    print("=" * 100)
    print("5. WHAT A 'CLEAN' VERDICT EXCLUDES")
    print("=" * 100)
    print("One-sided 95% upper bound on the rung, over the cells this study calls clean. A clean")
    print("cell does not say 'no retrieval'; it says 'no retrieval above this'.\n")
    from scipy.stats import beta
    def ub(k, n):
        return 100.0 if n == 0 or k >= n else 100 * beta.ppf(0.95, k + 1, n - k)
    rows = []
    for ds, g in clean.groupby("dataset"):
        u12 = np.median([ub(r.m2, r.m1) for _, r in g.iterrows() if r.m1 > 0])
        deep = g[g.m2 >= 15]
        u23 = np.median([ub(r.m3, r.m2) for _, r in deep.iterrows()]) if len(deep) else np.nan
        f12 = L.mode_R12.get(ds, np.nan)
        f23 = L.mode_R23.get(ds, np.nan)
        rows.append((ds, len(g), f12, u12, len(deep), f23, u23))
    print(f"{'benchmark':16s}{'clean':>6s}{'floor12':>9s}{'R12 <=':>9s}"
          f"{'w/ power':>10s}{'floor23':>9s}{'R23 <=':>9s}")
    for ds, n, f12, u12, nd, f23, u23 in sorted(rows):
        print(f"{ds:16s}{n:6d}{f12:8.1f}%{u12:8.1f}%{nd:10d}{f23:8.1f}%"
              f"{(f'{u23:7.1f}%' if np.isfinite(u23) else '      --')}")
    print("\n  where the R23 column is blank, no clean cell on that benchmark had m2 >= 15 --")
    print("  the 2->3 rung was never actually asked, so nothing about recall is excluded there.")

    print("\n" + "=" * 100)
    print("6. HOW MANY MOLECULES WOULD IT TAKE?")
    print("=" * 100)
    print(f"molecules needed for {100*args.power:.0f}% power at the BH operating point, using each")
    print("benchmark's own conditioning rate measured on its clean cells (m1/n and m2/n).\n")
    print(f"{'benchmark':16s}{'m1/n':>7s}{'m2/n':>7s}   "
          f"{'R12 = floor+2':>14s}{'+5':>7s}{'R23 = 2x floor':>16s}{'+10 pt':>9s}")
    for ds, g in clean.groupby("dataset"):
        r1 = float(np.median(g.m1 / g.n_usable))
        r2 = float(np.median(g.m2 / g.n_usable))
        f12, f23 = L.mode_R12.get(ds, np.nan), L.mode_R23.get(ds, np.nan)
        def mols(floor, target, rate):
            need = required_n(floor, target, eff, args.power)
            return np.inf if not np.isfinite(need) or rate <= 0 else np.ceil(need / rate)
        a = mols(f12, f12 + 2, r1)
        b = mols(f12, f12 + 5, r1)
        c = mols(f23, 2 * f23, r2)
        e = mols(f23, f23 + 10, r2)
        f = lambda x: "  >1e6" if not np.isfinite(x) or x > 1e6 else f"{int(x):6d}"
        print(f"{ds:16s}{r1:7.2f}{r2:7.3f}   {f(a):>14s}{f(b):>7s}{f(c):>16s}{f(e):>9s}")
    print("\n  The controlled arm bought 500 molecules per cell. Read the table as: anything")
    print("  under 500 was already answerable, anything above it is what a negative result on")
    print("  that benchmark is currently unable to rule out.")


if __name__ == "__main__":
    main()
