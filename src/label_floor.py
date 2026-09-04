"""The coincidence floor computed from the LABELS ALONE.

WHY THIS EXISTS. The floor the verdicts have rested on so far is the accuracy-matched
molecule-level derangement (`mb_chance_*`): the model's own predictions, permuted onto other
molecules. That floor is a function of the model's output, and it fails in the one direction a
null must not fail. A model that reproduces a benchmark has a prediction set that is nearly a
copy of the label set; permuting that copy onto other molecules collides often; the floor rises;
the retrieval is subtracted from itself. It is not a small effect -- the median floor over the
1024-token arm is 15.2%, but it runs 13.8% on cells the same scheme calls clean and 23.1% on the
cells it calls heavy. The floor is highest exactly where the signal is strongest.

The specific case that makes it indefensible: a model that has learned the DISTRIBUTION of a
benchmark's third significant figures -- that ESOL log-solubilities pile up on certain digits --
is a model that has read ESOL. That is retrieval. The matched null treats it as accuracy and
removes it.

WHAT REPLACES IT. Ask only what a molecule-blind guesser could do. Such a guesser knows the
benchmark's value distribution and nothing about which molecule it has been handed, so its next
significant figure is a draw from the label distribution conditional on the figures already
agreed. Two independent draws from a distribution q agree with probability sum_d q_d^2. This
depends on the labels and on nothing else: not on the model, not on its accuracy, not on the
verdict.

THE BOUND IS 1/11, NOT 1/10, and the difference is not pedantry -- it is the reason a floor of
9.7% is legitimate rather than a bug. A k-figure prefix does not have ten k+1-figure
continuations. It has ELEVEN, and the two at the edges are half-width. Prefix "5" at one figure
covers [4.5, 5.5), and the two-figure values reachable inside it are

    4.5  4.6  4.7  4.8  4.9  5  5.1  5.2  5.3  5.4  5.5      (enumerated in the tests)

because [4.5, 4.55) rounds to 4.5 and [5.45, 5.5) rounds to 5.5 -- each of those buckets keeps
only the half of its mass that lies inside the prefix. Cauchy-Schwarz over eleven outcomes gives

    sum_d q_d^2  >=  1/11  =  9.09% ,

and a perfectly smooth density -- no digit structure whatever -- gives the half-edge value

    2*(1/20)^2 + 9*(1/10)^2  =  9.50% ,

measured back at 9.50% on uniform(4.5, 5.5) and 9.6-10.6% on lognormal, normal and full-decade
uniforms. So the floor sits NEAR ten per cent for a structureless benchmark and may sit a little
under it. Only below 9.09% is it arithmetically impossible, and that is what this module asserts.

    (An earlier attempt at this number produced 0.3% on FreeSolv, which really was a bug: it
    dropped pairs of equal value -- exactly the coincidence being counted -- and drew the partner
    without replacement from a finite stratum, which is a property of the resampling scheme and
    not of guessing. Both are gone; the plug-in collision probability is used directly. The 9.7%
    on AqSolDB survived the fix, and the paragraph above is why.)

WEIGHTING. `floor` is weighted by the label distribution: pick a molecule at random from the
benchmark, ask how often a blind guesser continues its prefix correctly. Nothing about any run
enters. `floor_reached` re-weights the same per-molecule quantities over the molecules a given
cell actually reached, which is a sharper test but re-admits a mild dependence on the run (the
SET is chosen by the model; each molecule's floor is still a pure label property). The two are
reported side by side so the choice is visible; the verdicts use `floor`.

    python src/label_floor.py                 # per benchmark, writes results/label_floor.csv
    python src/label_floor.py --verbose        # digit histograms behind each number
"""
import argparse, os, sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import sig_figs, sig_round

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREEN = os.path.join(ROOT, "data", "screening")
RES = os.path.join(ROOT, "results")

BOUND = 100 / 11    # 9.09%: Cauchy-Schwarz over the eleven continuations of a prefix
SMOOTH = 9.50       # what a structureless benchmark gives: two half-edges, nine full buckets
NCONT = 11          # continuations of a k-figure prefix at k+1 figures; see the module docstring


def continuation_table(values, k):
    """{prefix at k figures: Counter of the k+1-figure continuation} over `values`.

    The continuation is the FULL k+1-figure string rather than the digit, because that is what
    the detector compares. It also silently gets the trailing-zero case right: %g renders a
    value whose next figure is 0 as the shorter string, so "1.2" is the k+1 = 3 continuation of
    prefix "1.2" and lands in its own bucket like any other digit.
    """
    tab = defaultdict(Counter)
    for v in values:
        tab[sig_round(v, k)][sig_round(v, k + 1)] += 1
    return tab


def pair_collision(values, k):
    """THE FLOOR. Of the ordered label pairs (i != j) agreeing at k figures, the share that also
    agrees at k+1, in per cent.

    This is the retention a model would show if it answered every molecule with the true value of
    a DIFFERENT molecule -- the coincidence the detector is supposed to be measuring against,
    written down directly. It needs no Monte Carlo: with A_p labels carrying k-figure prefix p and
    B_c labels carrying k+1-figure value c,

        floor = sum_c B_c (B_c - 1)  /  sum_p A_p (A_p - 1) .

    It is a function of the label multiset and of nothing else. No model, no prediction, no
    accuracy, no verdict.

    Preferred over the plug-in sum_d q_d^2 (`plugin_collision`) because that estimator is badly
    biased upward at these sample sizes: it charges a stratum containing one molecule a collision
    probability of 1. FreeSolv has 486 scorable labels spread over ~200 two-figure prefixes, so
    the plug-in reads 34.4% where this reads a tenth of that. The bias is (1 - sum p^2)/n_s per
    stratum, which is ruinous exactly where the strata are thin.
    """
    a = Counter(sig_round(v, k) for v in values)
    b = Counter(sig_round(v, k + 1) for v in values)
    den = sum(c * (c - 1) for c in a.values())
    num = sum(c * (c - 1) for c in b.values())
    return (100 * num / den if den else np.nan), len(a), den


def plugin_collision(values, k):
    """The stratified plug-in sum_s w_s sum_d q_{d|s}^2 -- reported for contrast, not used.

    Upward-biased at thin strata; see `pair_collision`. Kept because it is the estimator the
    Cauchy-Schwarz bound applies to exactly, so it is the one that demonstrates the >= 1/11 claim
    on real data rather than in the abstract.
    """
    tab = continuation_table(values, k)
    n = sum(sum(c.values()) for c in tab.values())
    if n == 0:
        return np.nan, 0, 0
    acc, widest = 0.0, 0
    for cnt in tab.values():
        ns = sum(cnt.values())
        widest = max(widest, len(cnt))
        acc += (ns / n) * sum((c / ns) ** 2 for c in cnt.values())
    return 100 * acc, len(tab), widest


collision = plugin_collision      # the name the smooth-distribution checks were written against


def per_molecule_floor(values, k):
    """The blind guesser's hit probability for EACH label, {label index: p}.

    q_{d_i|s_i}: how common molecule i's own k+1-figure continuation is among the labels sharing
    its k-figure prefix. Averaging these over all molecules reproduces `collision` exactly, which
    is asserted in `main`; averaging over a subset is what `floor_reached` is.
    """
    tab = continuation_table(values, k)
    out = np.empty(len(values))
    for i, v in enumerate(values):
        cnt = tab[sig_round(v, k)]
        out[i] = cnt[sig_round(v, k + 1)] / sum(cnt.values())
    return out


def mode_floor(values, k, reps=200, seed=0):
    """THE CONSERVATIVE FLOOR: what the BEST molecule-blind estimator achieves, in per cent.

    `pair_collision` answers "how often would a value drawn from this benchmark's distribution
    coincide". That is a coincidence rate, and it is not an upper bound on what a procedure
    knowing only the label distribution can do -- such a procedure should not SAMPLE from the
    distribution, it should play its mode. Against the sampling floor, "significant" means the
    model beat chance; against this one it means the model beat every strategy available without
    molecule-specific information, which is the claim the study actually wants to make.

    THE ESTIMATOR CLASS HAS TO BE NAMED or the floor is not defined. It is: procedures that see
    NO molecule-specific information at all -- not the structure, not the formula, not the mass,
    only the benchmark's label distribution. Within that class, emitting the modal continuation
    of the observed k-figure prefix is exactly optimal, so this floor is the class's supremum
    rather than one member of it. Allow the estimator to see the molecule and it becomes a QSAR
    model that beats everything, which is the accuracy-matched floor `floors()` rejects.

    IT MUST BE ESTIMATED OUT OF SAMPLE. Choosing the mode on the same labels it is scored against
    is fitting and evaluating on one set: in a prefix holding five molecules the empirical mode
    already "hits" 20% under a uniform truth. So the mode is chosen on a random half and scored
    on the other, averaged over `reps` splits. The difference is not cosmetic -- FreeSolv's
    2->3 rung reads 42.4% in sample and 13.9% honestly.

    It sits at or above `pair_collision` on all 24 benchmark-by-rung combinations measured, so
    switching to it can only ever REMOVE a flag.
    """
    rng = np.random.default_rng(seed)
    tab = continuation_table(values, k)
    hits = []
    for _ in range(reps):
        hit = tot = 0
        for cont in tab.values():
            c = np.array(list(cont.elements()))
            if len(c) < 2:
                continue                      # a lone molecule teaches nothing about its prefix
            idx = rng.permutation(len(c))
            fit, test = c[idx[:len(c) // 2]], c[idx[len(c) // 2:]]
            if not len(fit) or not len(test):
                continue
            best = Counter(fit).most_common(1)[0][0]
            hit += int((test == best).sum())
            tot += len(test)
        if tot:
            hits.append(100.0 * hit / tot)
    return float(np.mean(hits)) if hits else np.nan


def flat_collision(values, k):
    """Share of ordered label pairs (i != j) agreeing at k figures -- the floor for hit_k."""
    cnt = Counter(sig_round(v, k) for v in values)
    n = len(values)
    return 100 * sum(c * (c - 1) for c in cnt.values()) / (n * (n - 1)) if n > 1 else np.nan


def labels(dskey, level=3):
    """The benchmark's values, restricted to those the detector can score at `level`.

    A label written to fewer than `level` significant figures has no `level`-th figure to agree
    with, and `analyse_cell` excludes it from the denominator. Excluding it here as well keeps
    the floor and the statistic defined over the same population. This is a property of the
    labels, so it costs no independence.
    """
    f = os.path.join(SCREEN, f"{dskey}.csv")
    if not os.path.exists(f):
        return None
    v = pd.read_csv(f).dropna(subset=["value"]).value.astype(float).to_numpy()
    return v[[sig_figs(x) >= level for x in v]]


def floors_for(dskey, level=3, verbose=False):
    v = labels(dskey, level)
    if v is None or len(v) < 20:
        return None
    f12, n12, d12 = pair_collision(v, 1)
    f23, n23, d23 = pair_collision(v, 2)
    p12, _, w12 = plugin_collision(v, 1)
    p23, _, w23 = plugin_collision(v, 2)
    # BOTH strategies are available to a molecule-blind procedure -- draw from the distribution,
    # or play its mode -- so the supremum over the class is at least the larger of the two, and
    # that is what gets used. The max also absorbs the one place the split-sample mode estimate
    # is noisy: the 41-label positive-control set, where half of a prefix is a handful of
    # molecules and the held-out hit rate wobbles below the sampling floor.
    m12 = max(mode_floor(v, 1), f12)
    m23 = max(mode_floor(v, 2), f23)
    row = dict(dataset=dskey, n_label=len(v), level=level,
               mode_R12=m12, mode_R23=m23,
               # Carried so the classifier can express a cell's accuracy as a RELATIVE error,
               # which is the scale on which the significant-figure windows are defined.
               med_abs_value=float(np.median(np.abs(v))),
               floor_R12=f12, floor_R23=f23,
               plugin_R12=p12, plugin_R23=p23,
               floor_hit1=flat_collision(v, 1), floor_hit2=flat_collision(v, 2),
               floor_hit3=flat_collision(v, 3),
               n_prefix_R12=n12, n_prefix_R23=n23,
               # How many label pairs the ratio is built on. This is the floor's own sample size,
               # and it is what says whether a floor of 10.8% is a number or a rumour.
               pairs_R12=d12, pairs_R23=d23,
               widest_R12=w12, widest_R23=w23)
    if verbose:
        print(f"\n{dskey}: {len(v)} labels with >= {level} significant figures")
        for k, f in ((1, f12), (2, f23)):
            tab = continuation_table(v, k)
            big = sorted(tab.items(), key=lambda kv: -sum(kv[1].values()))[:3]
            print(f"  rung {k}->{k+1}: floor {f:5.2f}%   {len(tab)} prefixes, "
                  f"widest {max(len(c) for c in tab.values())} continuations")
            for pre, cnt in big:
                tot = sum(cnt.values())
                top = ", ".join(f"{s}:{c}" for s, c in cnt.most_common(4))
                print(f"      {pre:>10s}  n={tot:5d}  sum q^2={100*sum((c/tot)**2 for c in cnt.values()):5.2f}%"
                      f"   {top}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=os.path.join(RES, "label_floor.csv"))
    args = ap.parse_args()

    keys = sorted(f[:-4] for f in os.listdir(SCREEN) if f.endswith(".csv"))
    rows = [r for r in (floors_for(k, args.level, args.verbose) for k in keys) if r]
    D = pd.DataFrame(rows)

    # The bound is the whole argument, so it is checked rather than trusted -- on the plug-in,
    # which is the estimator Cauchy-Schwarz applies to. The pair estimator is a ratio of counts
    # over a finite label set and carries sampling noise, so it may sit just under.
    bad = D[(D.plugin_R12 < BOUND - 1e-9) | (D.plugin_R23 < BOUND - 1e-9)]
    assert bad.empty, f"plug-in floor below the Cauchy-Schwarz bound of {BOUND:.2f}%:\n{bad}"
    assert (D.widest_R12 <= NCONT).all() and (D.widest_R23 <= NCONT).all(), \
        f"more than {NCONT} continuations of a prefix"

    # Averaging the per-molecule floors must reproduce the plug-in aggregate exactly.
    for k in keys[:4]:
        v = labels(k, args.level)
        if v is None or len(v) < 20:
            continue
        for rung, kk in (("R12", 1), ("R23", 2)):
            a = 100 * per_molecule_floor(v, kk).mean()
            b = float(D.loc[D.dataset == k, f"plugin_{rung}"].iloc[0])
            assert abs(a - b) < 1e-9, f"{k} {rung}: {a} != {b}"

    print(f"\nLABEL-ONLY COINCIDENCE FLOOR  (level {args.level}, {len(D)} benchmarks)")
    print("the retention a model would show if it answered every molecule with the true value")
    print("of a different molecule. A function of the labels alone -- no model, no accuracy, no")
    print(f"verdict. A structureless benchmark gives {SMOOTH:.2f}%; more than that is digit clumping.")
    print("`plug-in` is the biased stratified estimator, shown to say how thin the strata are.\n")
    print(f"{'benchmark':16s}{'labels':>7s}{'samp12':>8s}{'samp23':>8s}"
          f"{'MODE12':>8s}{'MODE23':>8s}{'hit1':>8s}{'hit3':>8s}")
    for _, r in D.sort_values("mode_R23", ascending=False).iterrows():
        print(f"{r.dataset:16s}{r.n_label:7d}{r.floor_R12:8.2f}{r.floor_R23:8.2f}"
              f"{r.mode_R12:8.2f}{r.mode_R23:8.2f}{r.floor_hit1:8.2f}{r.floor_hit3:8.3f}")
    print(f"\n  median sampling R12 {D.floor_R12.median():.2f}%  R23 {D.floor_R23.median():.2f}%")
    print(f"  median mode     R12 {D.mode_R12.median():.2f}%  R23 {D.mode_R23.median():.2f}%")
    bad = D[(D.mode_R12 < D.floor_R12 - 1e-9) | (D.mode_R23 < D.floor_R23 - 1e-9)]
    assert bad.empty, f"the mode floor must dominate the sampling floor:\n{bad}"
    D.round(6).to_csv(args.out, index=False)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
