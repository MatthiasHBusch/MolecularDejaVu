"""Re-decide every cell, using the 2-figure rung where the 3-figure one has no power.

WHY THIS EXISTS. The original classifier called a cell `untestable` whenever m2 < 15, on the
grounds that R23 = m3/m2 conditions on m2 and a rung built on nine events is noise. That is true,
and it threw away a real verdict: 169 of 386 map cells -- 44% -- were labelled "no verdict" when
most of them are cells where the model is so far from the benchmark that there is no hint of
recall at ANY precision. `untestable` reads as "we could not tell". For those cells we can.

The 2-figure companion is the test that survives. R12 = m2/m1 conditions on m1, which is one to
two orders of magnitude larger than m2 precisely in the cells that fail the 3-figure gate: on
antiviral/DeepSeek V4 Pro, m1 = 211 against m2 = 14. Measured against the same accuracy-matched
molecule-level null, R12 = 6.6% against a 9.3% floor (p = 0.95). That is not an absent verdict.
It is a cell whose 2-figure agreement sits AT chance, which is what "this model is not reciting
this benchmark" looks like one rung down.

THE 2-FIGURE RUNG CERTIFIES ABSENCE. IT NEVER FLAGS. This is a deliberate asymmetry, and it is
forced by a defect in the R12 null rather than by taste. In `permutation_stats`:

    R12 = 100 * c2 / np.maximum(c1, 1)

so a permutation that happens to produce no 1-figure matches contributes a structural value rather
than being excluded, and the null is a ratio of two quantities that both vary instead of being
conditioned on the observed m1. The null MEAN survives this; the upper TAIL does not. Run as a
flagging test it duly manufactured false positives -- aqsoldb/GPT-5.5 came out "significant" at
p = 0.0005 on an excess of 1.10x with m1 = 21, which is not a credible p-value for 21 events, and
ppbr/Gemini 3.5 Flash was "flagged" off m1 = 1, m2 = 1, n_usable = 3.

So the rung is used only in the direction its statistics support:

    m2 >= 15                             -> the 3-figure verdict stands, as before   (decided_at 3)
    else, if m1 >= 15 and the test has power and R12 is consistent with its floor
                                         -> clean, starred                           (decided_at 2)
    else                                 -> no verdict, and we say so                (decided_at 0)

"Consistent with its floor" is a binomial bound, not a permutation p-value: with floor p0 and m1
conditioning matches, the observed retention must satisfy R12 <= p0 + 1.96*sqrt(p0(1-p0)/m1). A
cell whose 2-figure agreement sits within sampling noise of chance is certified clean; a cell with
a real excess gets NO verdict, because adjudicating that excess needs the tail this null cannot
provide. The certification therefore cannot be inflated by the defect -- the defect makes cells
look MORE significant, which here only ever costs a clean certificate.

"Has power" is not a threshold on m1 pulled from the air. For a cell with m1 conditioning matches
and floor p0, the smallest count a one-sided test could call significant is
k* = min{k : P(Binom(m1, p0) >= k) < alpha}, so the smallest detectable retention is k*/m1. We
require the test could have detected a DOUBLING of the floor, and separately that m1 >= 15 -- the
power criterion alone degenerates at tiny m1, where the resolvable floor 100/m1 grows large enough
to satisfy any ratio.

WHAT IS NOT USED AS A GATE. Low accuracy is real evidence -- a model whose predictions barely
correlate with the benchmark cannot be reciting it -- but the accuracy-matched null already
conditions on exactly that, so gating on Pearson r as well would count the same evidence twice.
It is reported next to every starred verdict instead, as context rather than as a criterion.

MULTIPLICITY. Benjamini-Hochberg runs over the tests that produce flags: two per 3-figure cell
(hit and retention). The 2-figure certification is an equivalence-style bound rather than a
discovery, so it enters no FDR family -- there is no false POSITIVE to control in a statement of
absence. This is a different family from the original 2x386, so q-values shift slightly; that is
the correct family for the discoveries actually being reported.

    python src/reclassify.py                 # rewrites both result sets
    python src/reclassify.py --dry           # show the movement without writing
"""
import argparse, os, sys

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import benjamini_hochberg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

# Moved to the deferred stack: run on the full map but never on the controlled-budget arm, because
# their endpoints do not honour a thinking budget and the arm could not afford them unsteered.
# Reporting a model in the map that the arm cannot revisit invites a comparison that cannot be
# made, so they come out of both.
DEFERRED = ("kimik26", "glm5", "glm52", "nemotron3u", "qwen35", "dsv4pro")

FLAGGED = ("heavy", "partial", "trace")


def min_detectable(m1, floor_pct, alpha=0.05):
    """Smallest R12 (%) a one-sided test could have called significant at this m1 and floor."""
    if m1 <= 0:
        return np.inf
    p0 = min(max(floor_pct / 100.0, 1.0 / max(m1, 1)), 0.999)
    k = binom.isf(alpha, m1, p0) + 1        # smallest k with P(X >= k) < alpha
    return 100.0 * min(k, m1) / m1


def consistent_with_floor(m1, r12, floor_pct, z=1.96):
    """Is this retention within sampling noise of its accuracy-matched floor?

    A binomial bound rather than the permutation p-value, because the R12 null's upper tail is
    not trustworthy at these sample sizes (see the module docstring). Used only to CERTIFY, so
    an error here costs a clean certificate and can never create a flag.
    """
    p0 = min(max(floor_pct / 100.0, 0.0), 1.0)
    se = np.sqrt(max(p0 * (1 - p0), 0.0) / max(m1, 1))
    return r12 <= 100.0 * (p0 + z * se)


def decide_level(row, min_m2=15, min_m1=15, power_ratio=2.0, alpha=0.05):
    """3, 2 or 0 -- which rung carries this cell's verdict."""
    if row.get("m2", 0) >= min_m2 and np.isfinite(row.get("R23", np.nan)):
        return 3
    m1 = row.get("m1", 0)
    # min_m1 is a hard floor, not decoration: the power criterion alone degenerates as m1 -> 1,
    # where the resolvable floor 100/m1 is large enough that any ratio satisfies it. That let a
    # cell through on m1 = 1, m2 = 1, n_usable = 3.
    if m1 < min_m1 or not np.isfinite(row.get("R12", np.nan)):
        return 0
    floor = max(row.get("mb_chance_R12", 0.0), 100.0 / m1)
    if min_detectable(m1, floor, alpha) > power_ratio * floor:
        return 0
    return 2 if consistent_with_floor(m1, row.get("R12", np.nan), floor) else 0


def regime_from(sig, excess):
    if sig and excess >= 4:
        return "heavy"
    if sig and excess >= 1.5:
        return "partial"
    if sig:
        return "trace"
    return "clean"


def reclassify(df, alpha=0.05, min_m2=15):
    """Add decided_at, excess, q_used and a regime, over the family of tests actually used."""
    d = df.copy()
    d["decided_at"] = [decide_level(r, min_m2) for _, r in d.iterrows()]

    # The BH family: the tests that generate FLAGS, and nothing else. The 2-figure rung produces
    # only certifications of absence, which are not discoveries and control no false-positive
    # rate, so those cells contribute no p-value here.
    pv, owner = [], []
    for i, r in d.iterrows():
        if r.decided_at == 3:
            pv += [r.mb_p_hit3, r.mb_p_R23]
            owner += [("hit", i), ("deep", i)]
    q = benjamini_hochberg(np.array(pv, dtype=float)) if pv else np.array([])
    qh = {i: np.nan for i in d.index}
    qd = {i: np.nan for i in d.index}
    for (kind, i), v in zip(owner, q):
        (qh if kind == "hit" else qd)[i] = v
    d["q_hit_used"], d["q_deep_used"] = pd.Series(qh), pd.Series(qd)

    regimes, excesses = [], []
    for i, r in d.iterrows():
        if r.decided_at == 0:
            regimes.append("untestable"); excesses.append(np.nan); continue
        if r.decided_at == 3:
            # A floor of zero means "below what this sample resolves", not "zero".
            floor = max(r.mb_chance_hit3, 100.0 / max(r.n_usable, 1))
            ex = r.hit3 / floor
            sig = ((np.isfinite(r.q_deep_used) and r.q_deep_used < alpha)
                   or (np.isfinite(r.q_hit_used) and r.q_hit_used < alpha and ex >= 2))
            # Keep the original two-test structure: retention gates heavy/partial, the hit rate
            # can only reach partial, and either alone reaches trace.
            deep_sig = np.isfinite(r.q_deep_used) and r.q_deep_used < alpha
            hit_sig = np.isfinite(r.q_hit_used) and r.q_hit_used < alpha
            if deep_sig and ex >= 4:
                reg = "heavy"
            elif (deep_sig and ex >= 1.5) or (hit_sig and ex >= 2):
                reg = "partial"
            elif deep_sig or hit_sig:
                reg = "trace"
            else:
                reg = "clean"
        else:
            # decided_at == 2 is a certification, and by construction it is always 'clean':
            # decide_level only returns 2 for a cell whose R12 sits within sampling noise of its
            # floor. The excess is carried for the table so the reader sees how close it was.
            floor = max(r.mb_chance_R12, 100.0 / max(r.m1, 1))
            ex, reg = r.R12 / floor, "clean"
        regimes.append(reg); excesses.append(ex)
    d["regime_new"], d["excess"] = regimes, excesses
    return d


def report(d, name):
    old, new = d.regime, d.regime_new
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"{'':22s}{'flagged':>9}{'clean':>8}{'untestable':>12}")
    for lab, v in (("before", old), ("after", new)):
        print(f"  {lab:20s}{int(v.isin(FLAGGED).sum()):9d}{int((v == 'clean').sum()):8d}"
              f"{int((v == 'untestable').sum()):12d}")
    moved = d[old != new]
    print(f"\n{len(moved)} of {len(d)} cells change label.")
    star = d[d.decided_at == 2]
    print(f"{len(star)} decided on the 2-figure rung (starred): "
          f"{int(star.regime_new.isin(FLAGGED).sum())} flagged, "
          f"{int((star.regime_new == 'clean').sum())} clean")
    rest = d[d.decided_at == 0]
    print(f"{len(rest)} still untestable  (median m1 {rest.m1.median() if len(rest) else 0:.0f}, "
          f"median n_usable {rest.n_usable.median() if len(rest) else 0:.0f})")
    if len(star):
        print("\n  a few starred cells, to show what they look like:")
        s = star.sort_values("m1", ascending=False).head(5)
        for _, r in s.iterrows():
            print(f"    {r.dataset:14s}{str(r.model)[:20]:21s} m1={r.m1:4.0f} m2={r.m2:3.0f}  "
                  f"R12={r.R12:5.1f}% vs floor {max(r.mb_chance_R12, 100/max(r.m1,1)):5.1f}%  "
                  f"r={r.pearson:+.2f}  -> {r.regime_new}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    # The zero-shot screen was the first entry here, labelled "THE MAP". It is retired and
    # archived under results/_archive_zeroshot/; the budget file carries both surviving arms.
    for src, label in ((os.path.join(RES, "budget_3sig_pad.csv"), "THE CONTROLLED-BUDGET ARM"),):
        if not os.path.exists(src):
            print(f"skip {src}: not found")
            continue
        raw = pd.read_csv(src)
        keep = raw[~raw.tag.isin(DEFERRED)].copy()
        dropped = raw.tag.isin(DEFERRED).sum()
        # The map file also holds the randomised-SMILES and other variants; only the canonical
        # zero-shot screen is the map, and only it gets reclassified here.
        if "study" in keep.columns:
            sel = (keep.study == "screening") & (keep.variant == "canonical")
        else:
            sel = pd.Series(True, index=keep.index)
        d = reclassify(keep[sel], alpha=args.alpha)
        print(f"\n{label}: dropped {dropped} deferred-model rows, reclassified {len(d)} cells "
              f"({d.tag.nunique()} models x {d.dataset.nunique()} benchmarks)")
        report(d, label)
        if not args.dry:
            out = src.replace(".csv", "_v2.csv")
            d.drop(columns=["regime"]).rename(columns={"regime_new": "regime"}).to_csv(
                out, index=False)
            print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
