"""Decide every cell: is this benchmark contaminated for this model, and how badly.

THE SCHEME IN FIVE LINES. Significance first, then shape, then size.

    m1 < 15                                  -> untestable
    neither rung significant                 -> clean
    R23 significant, R12 not                 -> TRACE     concentrated: a few per cent recited,
                                                          the benchmark as a whole untouched
    R12 significant, hit3 <  10%             -> PARTIAL   broad: the whole benchmark is shifted
    R12 significant, hit3 >= 10%             -> HEAVY     broad, and a tenth of it comes back

    Figure 2 of the paper is the reference rendering of this rule; everything else follows it.

    1. THE TWO RUNGS MEASURE TWO DIFFERENT THINGS, and that is what the categories are.
       R12 = m2/m1 conditions on every molecule the model gets roughly right -- 200-330 of 500 in
       a typical cell -- so it moves only when agreement is lifted across that whole population.
       It answers HOW CONTAMINATED IS THIS BENCHMARK. R23 = m3/m2 conditions on the few per cent
       already agreeing at two figures, so it fires on a handful of recited molecules while the
       rest of the benchmark is untouched. It answers DOES THIS MODEL RECITE SOME OF THESE VALUES.

       The split is not a relabelling. It separates cells a rate threshold cannot --
       ESOL/Gemini 3 Flash reproduces 5.3% verbatim with R12 at its floor (20.5 vs 20.0,
       concentrated), ESOL/GPT-5 reproduces LESS at 4.0% with R12 26.1 vs 15.3 (broad).
       Ordering by rate reverses them.

    1b. THE RECALL SIGNATURE IS A COLUMN, NOT A GATE. Recall grows down the ladder and competence
       does not: R23's excess over its floor exceeds R12's in 44 of the 45 contaminated cells on
       real benchmarks (median 3.56x against 1.41x; clean cells sit at 0.78x and 0.88x). It
       inverts on the positive control (2 of 22) only because both rungs saturate above 85%.
       Carried per cell as `recall_signature`. Requiring R23 as well as R12 for a verdict was
       implemented and removed -- over 154 cells it changed exactly one, FreeSolv/GPT-5.6 luna,
       and a one-cell exception does not earn a branch in the classifier.

    2. SIGNIFICANCE IS A TEST, SIZE IS A NUMBER, AND THEY ARE REPORTED SEPARATELY. The released
       scheme mixed them: `heavy >= 4x floor, partial >= 1.5x floor` could call a cell partial on
       an effect-size ratio while the evidence that anything was there at all was thin.

    3. THE NULL IS A CONDITIONAL BINOMIAL, NOT THE RATIO'S PERMUTATION TAIL. The defect in
       `permutation_stats` is that R12 = 100*c2/max(c1,1) is a ratio of two quantities that both
       vary across permutations, so its upper tail is dominated by permutations that happened to
       draw a small denominator. Run as a flagging test it produced aqsoldb/GPT-5.5 at p = 0.0005
       on an excess of 1.10x with m1 = 21. The null MEAN is sound, so it is kept as the floor and
       the observed count is tested against Binom(m_k, p0) conditional on the OBSERVED m_k.

    4. THE FLOOR IS THE HARDER OF TWO, EXCEPT WHERE THE SECOND IS CIRCULAR. Per rung: the
       accuracy-matched molecule-level permutation, and a simulated non-memorising predictor with
       the same median error and emitted precision (`smooth_error_null.py`). The simulation
       answers "could a merely good estimator have done this", which is exactly the objection R12
       invites, so where it is meaningful the floor is the max of the two.

       It stops being meaningful when the accuracy is itself a product of recall: Claude Opus 5
       on ESOL has medAE = 0.000, so a predictor simulated at ITS accuracy reproduces the
       benchmark too and the simulated R23 floor is 100%. Taking the max there would call the
       most contaminated cell in the panel clean, and all 22 positive-control cells are
       degenerate the same way by design. The guard is therefore dropped -- and recorded as
       dropped -- wherever the simulated predictor is itself near-reproducing the benchmark
       (sim hit3 floor > 25% or sim R12 floor > 40%): 25 of 154 arm cells. It can only ever
       REMOVE a constraint from a cell that still has to clear the permutation floor.

    5. MULTIPLICITY over every test that can flag: the R12 test of each testable cell plus the
       R23 test of each cell where that rung has power, corrected together, one BH family per
       arm. Two arms are two measurements of the same cells and are compared, not pooled.

    6. `trace` MEANS ONE THING ONLY: FLAGGED ON THE 2->3 RUNG ALONE. The released scheme used the
       name for cells significant on a ratio but tiny in absolute terms; that meaning is gone. It
       now names the shape of the evidence -- concentrated recall on a benchmark whose population
       as a whole is untouched -- and it is a level of `regime`, below `partial`, because a cell
       that never lifts the broad rung has not had its benchmark shifted.

    THE ALTERNATIVES, all implemented so the choice is a number rather than a preference:

      --gate rung      (default)  as above                      t1024: 82 contaminated
                                                                (38 heavy, 31 partial, 13 trace)
      --gate both                 R23 decides where it has      66 contaminated, severity by hit3
                                  power, R12 where it does not  alone (no trace level)
      --gate r12first             R12 screens, R23 confirms     56 contaminated -- DISCARDS 11 cells
                                                                that sit 4-9 sigma above the floor
                                                                on R23 while R12 is at chance

    `r12first` fails because recall concentrates: median evidence over the arm's cells with power
    on the three recalled benchmarks is 2.9 sigma on R12 against 6.7 sigma on R23. The default
    keeps both rungs, which is why it loses nothing.

    python src/classify.py                      # both arms, rewrites *_v3.csv
    python src/classify.py --dry                # show the movement, write nothing
    python src/classify.py --gate both --compare
"""
import argparse, os, sys

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import benjamini_hochberg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

# Never in the controlled arm: their endpoints do not honour a thinking budget, so they cannot be
# dosed and the arm could not afford them unsteered. Reporting them in one arm and not the other
# invites a comparison that cannot be made.
DEFERRED = ("kimik26", "glm5", "glm52", "nemotron3u", "qwen35", "dsv4pro")

# QM9 was stopped after four cells on cost and is reported nowhere. A multiplicity family should
# contain the tests that are actually reported: leaving four unreported cells in it corrects the
# whole arm against evidence no reader ever sees, and shifts the BH threshold by their share of
# the family. Dropped here rather than filtered downstream so the q-values are right at source.
DROP_DATASETS = ("qm9",)

MIN_COND = 15        # conditioning matches needed for a rung to carry a test
HEAVY_HIT3 = 10.0    # per cent of the benchmark reproduced to three significant figures
ALPHA = 0.05

# Above these the simulated same-accuracy predictor is itself near-reproducing the benchmark, so
# it is no longer a floor for coincidence -- it is a second copy of the recall being measured.
DEGEN_HIT3, DEGEN_R12 = 25.0, 40.0


def binom_p(k, n, p0):
    """P(X >= k) for X ~ Binom(n, p0), one-sided, conditional on the observed n."""
    if not np.isfinite(k) or not np.isfinite(n) or n < 1:
        return np.nan
    p0 = float(min(max(p0, 1e-6), 1 - 1e-6))
    return float(binom.sf(int(k) - 1, int(n), p0))


LABEL_FLOOR = None      # {dataset: {"R12": %, "R23": %, "hit3": %}}, filled by load_label_floor


def load_label_floor():
    """The label-only floor, keyed by benchmark. See src/label_floor.py for what it is."""
    global LABEL_FLOOR
    p = os.path.join(RES, "label_floor.csv")
    if not os.path.exists(p):
        sys.exit(f"--floor label needs {p}; run  python src/label_floor.py")
    L = pd.read_csv(p).set_index("dataset")
    LABEL_FLOOR = {k: dict(R12=r.mode_R12, R23=r.mode_R23, hit3=r.floor_hit3,
                           hit1=r.floor_hit1, med_abs=r.med_abs_value,
                           # kept for the appendix comparison: the sampling floor the scheme
                           # used before the class supremum replaced it
                           samp_R12=r.floor_R12, samp_R23=r.floor_R23)
                   for k, r in L.iterrows()}


def floors(row, rung, mode="matched"):
    """(floor in %, whether the simulated guard was applied) for rung '12' or '23'.

    mode='matched'  the accuracy-matched molecule-level derangement, hardened with the simulated
                    same-accuracy predictor. What the released scheme used.

    mode='label'    THE BEST A MOLECULE-BLIND PROCEDURE CAN DO, computed from the benchmark's
                    own labels and nothing else (src/label_floor.py, `mode_floor`).

        The matched floor is a function of the model's output, and it fails in the direction a
        null must not: a model that reproduces the benchmark has a prediction set that is nearly
        a copy of the label set, so permuting it onto other molecules collides often and the
        floor rises with the very signal it is meant to exclude. Measured over the 1024-token
        arm, the matched floor runs 13.8% on cells this scheme calls clean and 23.1% on cells it
        calls heavy. It is highest exactly where the effect is strongest.

        The decisive case is distributional knowledge. A model that has learned which third
        significant figures a benchmark's values prefer has read that benchmark; that IS
        retrieval, and the matched null books it as accuracy and subtracts it.

        Under 'label' the simulated same-accuracy guard goes too, for the same reason -- it is
        built from the model's own median error and emitted precision. What replaces both is one
        number per benchmark that no run can move.

        AND IT IS THE CLASS SUPREMUM, NOT A COINCIDENCE RATE. An earlier version used the rate at
        which two of the benchmark's own labels collide -- "the model answered with a different
        molecule's value". That is a coincidence rate, and it is not an upper bound: a procedure
        knowing only the label distribution should play its modal continuation rather than draw
        from it, and doing so beats the collision rate on 24 of 24 benchmark-by-rung
        combinations. The floor used is therefore the larger of the two, which makes
        "significant" mean `beat every strategy available without molecule-specific information'
        rather than `beat chance'. It costs 17 of 107 flags in the 1024-token arm and 10 of 57 in
        the minimum arm, and 16 of those 17 sit on the benchmarks whose values cluster hardest
        (QM8 11->2, QM7 5->3, PPBR 4->1); ESOL, FreeSolv, AqSolDB and the positive control do not
        move at all.

        THE PRICE, stated because it is real: with no accuracy term anywhere, a model accurate
        enough to resolve the second significant figure by competence alone will lift R12 without
        reciting anything. The retention argument bounds when that can happen -- it needs a
        relative error well inside the 2-figure window, of order 1% -- and `precision_ratio` in
        the output reports each cell's distance from that regime. The 2->3 rung needs another
        decade of accuracy again, which is why it remains the specific test.
    """
    if mode == "label":
        f = LABEL_FLOOR.get(row["dataset"])
        if f is None:
            return 100.0 / 11, False        # unknown benchmark: the arithmetic bound, no guard
        return float(f[f"R{rung}"]), False
    perm = row.get(f"mb_chance_R{rung}", np.nan)
    perm = 10.0 if not np.isfinite(perm) else float(perm)
    sim = row.get(f"R{rung}_smooth_floor", np.nan)
    degen = (row.get("smooth_floor", 0) > DEGEN_HIT3
             or row.get("R12_smooth_floor", 0) > DEGEN_R12
             or not np.isfinite(sim))
    if degen:
        return perm, False
    return max(perm, float(sim)), True


def classify(df, alpha=ALPHA, gate="both", floor="matched"):
    """Significance on the digit rungs, then a severity split on hit3.

    gate="both"     R23 decides where m2 >= 15, R12 where it does not  (the measured choice)
    gate="r12first" R12 screens, R23 confirms                          (kept for comparison)
    floor=          which null the rungs are tested against; see `floors`
    """
    d = df.copy().reset_index(drop=True)

    f12 = [floors(r, "12", floor) for _, r in d.iterrows()]
    f23 = [floors(r, "23", floor) for _, r in d.iterrows()]
    d["floor_R12"] = [x[0] for x in f12]
    d["floor_R23"] = [x[0] for x in f23]
    d["guard_R12"] = [x[1] for x in f12]
    d["guard_R23"] = [x[1] for x in f23]

    d["testable"] = d.m1 >= MIN_COND
    d["deep"] = d.testable & (d.m2 >= MIN_COND)

    # Both rungs are always TESTED where they have power; which one DECIDES is the gate's job.
    d["p_R12_binom"] = [binom_p(r.m2, r.m1, r.floor_R12 / 100.0) if r.testable else np.nan
                        for _, r in d.iterrows()]
    d["p_R23_binom"] = [binom_p(r.m3, r.m2, r.floor_R23 / 100.0) if r.deep else np.nan
                        for _, r in d.iterrows()]
    # The two unconditional rates. NEITHER FLAGS -- see the note in the 'rung' gate for why the
    # label floor is a sufficient null for a retention and not for a rate. They are computed and
    # reported because one of them (m1) carries the negative check, and because a reader should
    # be able to see what was left on the table rather than take the exclusion on trust.
    if floor == "label":
        fh3 = np.array([LABEL_FLOOR.get(x, {}).get("hit3", np.nan) for x in d.dataset])
        fh1 = np.array([LABEL_FLOOR.get(x, {}).get("hit1", np.nan) for x in d.dataset])
    else:
        fh3 = d.get("mb_chance_hit3", pd.Series(np.nan, index=d.index)).to_numpy()
        fh1 = np.full(len(d), np.nan)
    d["p_hit3_binom"] = [binom_p(r.m3, r.n_usable, f / 100.0) if np.isfinite(f) else np.nan
                         for (_, r), f in zip(d.iterrows(), fh3)]
    d["p_m1_binom"] = [binom_p(r.m1, r.n_usable, f / 100.0) if np.isfinite(f) else np.nan
                       for (_, r), f in zip(d.iterrows(), fh1)]
    d["floor_m1"] = fh1
    d["rate_m1"] = 100.0 * d.m1 / d.n_usable.clip(lower=1)

    def bh(col, mask):
        q = np.full(len(d), np.nan)
        idx = np.where(mask.to_numpy() & np.isfinite(d[col]).to_numpy())[0]
        if len(idx):
            q[idx] = benjamini_hochberg(d[col].to_numpy()[idx])
        return q

    d["q_hit3"] = np.nan          # only the 'rung' gate lets hit3 flag; see below

    if gate == "rung":
        # THREE TESTS CAN FLAG, and they enter one BH family because they are one discovery
        # question asked three ways.
        #
        # R12 and R23 are the two rungs: one per cell where only R12 has power, two where both
        # do. They answer different questions, not the same question twice, which is what makes
        # the pair worth correcting jointly.
        #
        # NEITHER hit3 NOR m1 IS IN THIS FAMILY, and both were tried. The reason is the same one
        # in both cases, and it is the reason the rungs are ratios in the first place.
        #
        # R12 and R23 condition on a prior match. Inside a window already localised to 10^-k a
        # smooth error density is close to uniform, so the next digit lands at the coincidence
        # floor no matter how accurate the model is. That is what makes the label-only floor a
        # sufficient null for them: accuracy cannot lift a retention.
        #
        # hit3 and m1 have no such conditioning, so accuracy CAN lift them, and the label floor
        # does not exclude it. Measured:
        #
        #   m1    flags 215 cells no rung flags across the three controlled arms and 352 across
        #         all four including the zero-shot map -- the count moves with the arm set, so
        #         quote it with one. Those cells sit at a median +15.2 points on the
        #         first figure and +0.1 / -0.9 / +0.1 points on R12 / R23 / hit3 -- they place
        #         the order of magnitude and are then indistinguishable from coincidence. That
        #         is competence.
        #   hit3  flags 38 cells no rung flags, 17 of them (benchmark, model) pairs never
        #         flagged anywhere, and unlike m1 they point the right way: all 21 additions in
        #         the controlled arm have R23 above its floor, median excess +12 points. They
        #         are the underpowered tail of the same effect -- R23 conditions on a median m2
        #         of 17, hit3 on n of ~450.
        #
        # hit3 was nonetheless dropped. The only null that would exclude competence for it is a
        # simulated predictor matched to the model's own median error -- and that is the
        # result-dependent construction this file abandoned for the rungs, for the reason given
        # in floors(): it inherits the recall it is meant to null out, and has to be declared
        # degenerate in 25 of 154 cells precisely where the effect is strongest. Building a
        # flag on a null that cannot be defended buys 38 cells at the cost of the argument. The
        # cells are not lost -- they are reported as underpowered, which is what they are, and
        # `detection_limits.py` says what it would take to test them properly.
        #
        # m1 does earn its keep, but only as a NEGATIVE: see `first_figure_at_chance` below.
        d["q_R12"] = bh("p_R12_binom", d.testable)
        d["q_R23"] = bh("p_R23_binom", d.deep)
        pooled = benjamini_hochberg(np.concatenate([
            d.p_R12_binom.fillna(1.0).to_numpy(), d.p_R23_binom.fillna(1.0).to_numpy()]))
        d["q_R12"] = np.where(d.testable, pooled[:len(d)], np.nan)
        d["q_R23"] = np.where(d.deep, pooled[len(d):], np.nan)
    elif gate == "r12first":
        # Sequential gatekeeping: BH over every R12 test, then BH over the R23 tests of the
        # cells that survived. Kept so the comparison is measurable, not asserted.
        d["q_R12"] = bh("p_R12_binom", d.testable)
        d["q_R23"] = bh("p_R23_binom", d.deep & (d.q_R12 < alpha))
    else:
        # The discovery family is the set of tests that can produce a flag: R23 where it has
        # power, R12 only where R23 does not. R12 elsewhere is a reported screen, not a
        # discovery, so it is corrected in its own family and controls nothing here.
        flagging = d.p_R23_binom.where(d.deep, d.p_R12_binom.where(d.testable & ~d.deep))
        d["p_flag"] = flagging
        d["q_flag"] = bh("p_flag", d.testable)
        d["q_R23"] = d.q_flag.where(d.deep)
        d["q_R12"] = np.where(d.deep, bh("p_R12_binom", d.deep), d.q_flag)

    # THE NEGATIVE CHECK, and it is only ever a negative. A cell whose rungs have no power is a
    # blank, not a pass. But if the model does not place the ORDER OF MAGNITUDE either -- if its
    # unconditional first-figure rate sits at or below the rate at which two of the benchmark's
    # own labels share a first figure -- then there is nothing there that could have been
    # retrieved, and the blank can be reported as a negative.
    #
    # Two conditions, not one. "Not significant" is not "at chance": a cell with six usable
    # predictions can miss significance while sitting at three times its floor, and certifying
    # that would sell absence of power as absence of effect. The rate must ALSO be at or below
    # the floor. Measured: in the controlled arm all 15 untestable cells qualify (median rate
    # 0.20% against a floor of 50.9%); in the map 45 of 60 do, and the other 15 stay blank.
    d["first_figure_at_chance"] = (~(d.p_m1_binom < alpha).fillna(False)
                                   & (d.rate_m1 <= d.floor_m1))

    verdict, status = [], []
    for i, r in d.iterrows():
        if not r.testable:
            if r.first_figure_at_chance:
                verdict.append("no-signal")
                # The numbers live in rate_m1 / floor_m1; the string stays constant so the
                # evidence column still aggregates.
                status.append("rungs without power; first figure at chance")
            else:
                verdict.append("untestable"); status.append("m1 < %d" % MIN_COND)
            continue
        sig12, sig23 = r.q_R12 < alpha, r.q_R23 < alpha
        if gate == "rung":
            # The two rungs measure two different things, so the severity IS which one fires.
            # R12 is a whole-benchmark statistic: it moves only when agreement is lifted across
            # the population of molecules the model gets roughly right. R23 conditions on the
            # few per cent that already agree to two figures, so it can fire on a handful of
            # recited molecules while the benchmark as a whole is untouched.
            #
            # EITHER RUNG FLAGS ON ITS OWN. An earlier version required R23 as well wherever it
            # had power, on the argument that a model which has genuinely learned the property
            # can land the second significant figure more often than a Laplace-error predictor
            # of the same median error -- errors piled tightly around zero without ever hitting
            # it -- and that lifts R12 with no retrieval anywhere. The argument is sound and the
            # diagnostic for it is carried per cell as `recall_signature` (R23's excess over its
            # floor against R12's; recall grows down the ladder, competence does not). It is NOT
            # used as a rule: it fired on one cell of 154, FreeSolv/GPT-5.6 luna, and a
            # one-cell exception is not worth a branch in the classifier. The cell is reported
            # with its evidence string, so a reader can see exactly what it is.
            if sig12:
                verdict.append("contaminated")
                status.append("R12 + R23" if sig23 else "R12 only (broad, R23 at chance)")
            elif sig23:
                verdict.append("contaminated"); status.append("R23 only (concentrated)")
            else:
                verdict.append("clean")
                status.append("both rungs at chance" if r.deep else "R12 at chance, R23 no power")
            continue
        if gate == "r12first":
            if not sig12:
                verdict.append("clean"); status.append("R12 at chance")
            elif not r.deep:
                verdict.append("contaminated"); status.append("R12 only, R23 had no power")
            elif sig23:
                verdict.append("contaminated"); status.append("R12 + R23")
            else:
                verdict.append("clean"); status.append("R12 only, R23 declined")
            continue
        if not r.deep:
            # No 2->3 rung to speak of. R12 is all there is, and it can now flag as well as
            # certify -- the released scheme could only certify.
            verdict.append("contaminated" if sig12 else "clean")
            status.append("R12 only, R23 had no power" if sig12 else "R12 at chance, R23 no power")
        elif sig23:
            verdict.append("contaminated")
            # Worth separating: a cell whose 1->2 rung is at chance while its 2->3 rung fires is
            # the dilution case, and it is the case an R12 screen would have thrown away.
            status.append("R23 + R12" if sig12 else "R23 only (R12 diluted)")
        else:
            verdict.append("clean")
            status.append("R12 only, R23 declined" if sig12 else "both rungs at chance")
    d["verdict"], d["evidence"] = verdict, status

    # THREE LEVELS, from which rung fires and then how much of the benchmark is involved:
    #
    #   trace    only the 2->3 rung fires -- a few per cent are reproduced verbatim and the
    #            benchmark as a whole is untouched
    #   partial  the 1->2 rung fires: the whole benchmark is shifted towards its published
    #            values, but under 10% of it comes back to three figures
    #   heavy    the same, with hit3 >= 10%
    #
    # hit3 enters only as the partial/heavy cut, so significance and size stay separate: the
    # rungs decide WHETHER and in what shape, the rate decides HOW MUCH.
    #
    # heavy REQUIRES the broad rung as well as the rate, and that is not a refinement for its
    # own sake -- it is forced by QM7. A hit3 of 10% means different things on different
    # benchmarks, because the coincidence rate depends on how the benchmark's own values are
    # distributed. QM7's atomization energies cluster hard, so its simulated same-accuracy
    # floor runs to 16.4% at three significant figures against 0.3-1.3% on LD50. Claude Opus 5
    # on QM7 reproduces 16.2% -- 0.99x its floor, i.e. exactly chance -- while its R23 clears
    # a 19.9% floor by enough to flag. Under a rate-only cut that cell reads `heavy`, level
    # with LD50 cells sitting at 16x their floor. Requiring R12 makes it `trace`, which is
    # what it is: concentrated agreement on a benchmark where agreement is cheap.
    if gate == "rung":
        # `no-signal` is not a fifth severity, it is the other kind of blank: the rungs had no
        # power AND the model does not place the order of magnitude. `heavy` still requires the
        # broad rung as well as the rate, so a cell flagged by hit3 alone lands in `trace`
        # whatever its rate -- concentrated agreement is what it is, and the QM7 argument below
        # is exactly why a rate alone must not promote a cell.
        d["regime"] = [("no-signal" if v == "no-signal" else
                        "untestable" if v == "untestable" else "clean" if v == "clean" else
                        "trace" if not s.startswith("R12") else
                        "heavy" if h >= HEAVY_HIT3 else "partial")
                       for v, s, h in zip(d.verdict, d.evidence, d.hit3.fillna(0.0))]
    else:
        d["regime"] = [
            ("untestable" if v == "untestable" else
             "clean" if v == "clean" else
             ("heavy" if h >= HEAVY_HIT3 else "partial"))
            for v, h in zip(d.verdict, d.hit3.fillna(0.0))]
    # How much of the benchmark each rung's excess actually covers, as a share of all scorable
    # pairs: the quantity the heavy/partial distinction claims to be about.
    d["breadth_R12"] = 100 * (d.m2 - d.floor_R12 / 100 * d.m1).clip(lower=0) / d.n_usable
    d["breadth_R23"] = 100 * (d.m3 - d.floor_R23 / 100 * d.m2).clip(lower=0) / d.n_usable
    # The recall signature, floor-corrected. Comparing R23 to R12 directly is wrong -- their
    # floors differ by 5-10 points -- but comparing each to its own floor is not: recall grows
    # down the ladder, competence does not. True for 44 of 45 contaminated cells on real
    # benchmarks. It inverts on the positive control (2 of 22) because both rungs saturate above
    # 85%, which is a ceiling effect rather than a counterexample.
    # Reported next to the verdict, not used to reach it: how far above its floor the cell's
    # retention sat, on each rung.
    d["excess_R12"] = d.R12 / d.floor_R12.clip(lower=0.01)
    d["excess_R23"] = d.R23 / d.floor_R23.clip(lower=0.01)
    # The same for the rate the severity cut is made on. It does not move a cell -- the cut is
    # hit3 >= 10% and nothing else -- but it is what tells a reader whether a given 10% is a lot
    # for that benchmark. Same floor rule as the rungs: the harder of permutation and simulation,
    # simulation dropped where it is degenerate.
    if floor == "label":
        floor_h3 = np.array([LABEL_FLOOR.get(x, {}).get("hit3", np.nan) for x in d.dataset])
    else:
        floor_h3 = np.where(d.smooth_floor.fillna(0) > DEGEN_HIT3,
                            d.mb_chance_hit3, np.fmax(d.mb_chance_hit3, d.smooth_floor.fillna(0)))
    d["floor_hit3"] = floor_h3
    # How close the cell is to the accuracy at which competence alone could lift the 1->2 rung.
    # The 2-figure window is about 5% of the value at leading digit 1 and 0.5% at leading digit 9,
    # so a median relative error at or under a per cent is the regime where matching the second
    # figure stops being a coincidence and starts being resolution. Everything in this study sits
    # far above that -- reported per cell rather than asserted, because it is the one objection
    # the label-only floor does not answer by construction.
    if "medae" in d.columns and LABEL_FLOOR:
        scale = np.array([LABEL_FLOOR.get(x, {}).get("med_abs", np.nan) for x in d.dataset])
        d["rel_medae"] = 100 * d.medae.to_numpy() / scale
    d["excess_hit3"] = d.hit3 / np.clip(floor_h3, 0.01, None)
    d["recall_signature"] = d.excess_R23 > d.excess_R12
    return d


def join_smooth(base, smooth_path):
    """Attach the simulated floors, keyed by arm where the file carries one.

    The simulation is driven by the cell's own median error, and that differs between arms for
    the same (dataset, model) -- joining on the pair alone would give the zero-budget cell the
    budget cell's floor.
    """
    S = pd.read_csv(smooth_path)
    cols = [c for c in ("smooth_floor", "R12_smooth_floor", "R23_smooth_floor",
                        "R13_smooth_floor", "hit1_smooth_floor") if c in S.columns]
    on = ["dataset", "tag"] + (["arm"] if "arm" in S.columns and "arm" in base.columns else [])
    if "arm" in S.columns and "arm" not in on:
        S = S.drop(columns=["arm"]).drop_duplicates(subset=["dataset", "tag"])
    return base.merge(S[on + cols], on=on, how="left")


def report(d, name, old_col="regime_old"):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    n = len(d)
    print(f"{n} cells: {int((d.verdict == 'contaminated').sum())} contaminated "
          f"({int((d.regime == 'heavy').sum())} heavy, {int((d.regime == 'partial').sum())} "
          f"partial), {int((d.verdict == 'clean').sum())} clean, "
          f"{int((d.verdict == 'untestable').sum())} untestable")
    print(f"  evidence: " + ", ".join(f"{k} {v}" for k, v in d.evidence.value_counts().items()))
    print(f"  simulated guard applied on the 1->2 rung in {int(d.guard_R12.sum())} of {n} cells")
    if old_col in d.columns:
        print(f"\n  movement against the released scheme:")
        print(pd.crosstab(d[old_col], d.regime).to_string())
    print(f"\n  per benchmark:")
    t = d.pivot_table(index="dataset", columns="regime", values="tag", aggfunc="count",
                      fill_value=0)
    print(t.to_string())


def floor_movement(a, b, name_a, name_b, title):
    """Every cell whose verdict differs between two floors, and what the floors were.

    The comparison IS the finding, so it is printed cell by cell rather than summarised: which
    cells the accuracy-matched null was hiding, and which it was inventing.
    """
    key = ["dataset", "tag"] + (["arm"] if "arm" in a.columns else [])
    A, B = a.set_index(key), b.set_index(key)
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]
    moved = A.regime != B.regime
    print(f"\n{'-' * 96}\nFLOOR: '{name_a}' vs '{name_b}'   --   {title}")
    print(f"{int(moved.sum())} of {len(A)} cells change regime\n")
    print(pd.crosstab(B.regime, A.regime, rownames=[name_b], colnames=[name_a]).to_string())
    if not moved.any():
        return
    print(f"\n{'benchmark':13s}{'model':16s}{'arm':7s}"
          f"{'R12':>7s}{name_b[:5]+'12':>8s}{name_a[:5]+'12':>8s}"
          f"{'R23':>8s}{name_b[:5]+'23':>8s}{name_a[:5]+'23':>8s}   {name_b} -> {name_a}")
    for k in A.index[moved]:
        ra, rb = A.loc[k], B.loc[k]
        arm = k[2] if len(k) > 2 else ""
        print(f"{k[0]:13s}{k[1]:16s}{arm:7s}"
              f"{ra.R12:7.1f}{rb.floor_R12:8.1f}{ra.floor_R12:8.1f}"
              f"{ra.R23:8.1f}{rb.floor_R23:8.1f}{ra.floor_R23:8.1f}   "
              f"{rb.regime} -> {ra.regime}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--gate", default="rung", choices=["rung", "both", "r12first"],
                    help="'rung' = both rungs flag and the severity is WHICH one fires "
                         "(R12 -> heavy/broad, R23 only -> partial/concentrated); "
                         "'both' = R23 decides where it has power, R12 where it does not, "
                         "severity by hit3; 'r12first' = R12 screens and R23 confirms")
    ap.add_argument("--compare", action="store_true",
                    help="also run the other gate and print what it would have cost")
    ap.add_argument("--floor", default="label", choices=["matched", "label"],
                    help="'matched' = accuracy-matched permutation + simulated guard (the "
                         "released scheme, a function of the model's own output); "
                         "'label' = the benchmark's own label coincidence rate, which no run "
                         "can move. See floors().")
    ap.add_argument("--compare-floor", action="store_true",
                    help="classify under BOTH floors and print every cell that moves")
    args = ap.parse_args()
    if args.floor == "label" or args.compare_floor:
        load_label_floor()

    # The zero-shot screen was a second job here. It is retired -- it measured what the
    # minimum-reasoning arm measures, on an older pipeline -- and its files are under
    # results/_archive_zeroshot/. Both surviving arms live in the budget file and are
    # classified together, each as its own BH family.
    jobs = [("THE CONTROLLED-BUDGET ARM", os.path.join(RES, "budget_3sig_pad.csv"),
             os.path.join(RES, "smooth_error_null_budget.csv"),
             os.path.join(RES, "budget_3sig_v3.csv"))]

    for label, src, smooth, out in jobs:
        if not os.path.exists(src):
            print(f"skip {label}: {src} not found")
            continue
        raw = pd.read_csv(src)
        keep = raw[~raw.tag.isin(DEFERRED) & ~raw.dataset.isin(DROP_DATASETS)].copy()
        if "study" in keep.columns:
            keep = keep[(keep.study == "screening") & (keep.variant == "canonical")]
        keep = keep.rename(columns={"regime": "regime_old"})
        if os.path.exists(smooth):
            keep = join_smooth(keep, smooth)
        else:
            print(f"  ! {label}: no simulated floors at {smooth}; permutation floor only")
        # One arm is one experiment and gets its own BH family. The zero-budget arm asks the
        # same cells the same questions at a different deliberation, so pooling the two would
        # correct a comparison over a family that contains both of its sides.
        def run(fl):
            if "arm" in keep.columns and keep.arm.nunique() > 1:
                return pd.concat([classify(g, alpha=args.alpha, gate=args.gate, floor=fl)
                                  for _, g in keep.groupby("arm")], ignore_index=True)
            return classify(keep, alpha=args.alpha, gate=args.gate, floor=fl)

        d = run(args.floor)
        other = "label" if args.floor == "matched" else "matched"
        for a, g in ([(None, d)] if "arm" not in d.columns else d.groupby("arm")):
            report(g, label if a is None else f"{label}  [arm {a}]")
        if args.compare_floor:
            floor_movement(d, run(other), args.floor, other, label)
        if args.compare:
            other = "r12first" if args.gate == "both" else "both"
            o = classify(keep, alpha=args.alpha, gate=other)
            a = d.set_index(["dataset", "tag"]).verdict
            b = o.set_index(["dataset", "tag"]).verdict
            diff = a.index[(a == "contaminated") & (b != "contaminated")]
            gain = a.index[(a != "contaminated") & (b == "contaminated")]
            print(f"\n  gate '{args.gate}' vs '{other}': "
                  f"{len(diff)} cells flagged here and not there, {len(gain)} the other way")
            for k in list(diff)[:20]:
                r = d.set_index(["dataset", "tag"]).loc[k]
                print(f"    {k[0]:14s}{k[1]:16s} hit3 {r.hit3:5.2f}  R12 {r.R12:5.1f}/"
                      f"{r.floor_R12:4.1f}  R23 {r.R23:5.1f}/{r.floor_R23:4.1f}  {r.evidence}")
        if not args.dry:
            d.to_csv(out, index=False)
            print(f"\n  wrote {out}  (floor: {args.floor})")
            if args.compare_floor:
                # The superseded null is kept as its own file rather than deleted: the paper has
                # to be able to show what changed, and that comparison is a result in itself.
                alt = out.replace(".csv", f"_{other}.csv")
                run(other).to_csv(alt, index=False)
                print(f"  wrote {alt}  (floor: {other}, for the comparison only)")


if __name__ == "__main__":
    main()
