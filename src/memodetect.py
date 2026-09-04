"""
Digit-level memorization detector -- shared core.

Statistics
----------
Let sig_n(x) be x rounded to n significant figures. A prediction p is an n-sig hit for
truth t iff sig_n(p) == sig_n(t).

    m1, m2, m3   nested hit counts (a pair is only counted at n+1 if it matched at n)
    h1, h2, h3   unconditional hit counts at each precision
    R12 = m2/m1, R23 = m3/m2   retention: of the pairs matching at n digits, the fraction
                               that also match at n+1.

Why retention rather than hit rate: matching at n significant figures already localises the
error to a window of relative width ~10^-n. If the model's error density is smooth, its
conditional distribution inside that window is close to uniform, so the next digit agrees
with probability ~ the coincidence floor REGARDLESS of how accurate the model is. Retention
therefore departs from the floor only when the error distribution has a point mass at
exactly zero -- the signature of reading out a stored value. That argument is tight at the
2->3 rung and only approximate at 1->2, where the window is ten times wider; R23 is thus
the specific test and R12 the higher-powered but accuracy-sensitive screen.

Two nulls
---------
GLOBAL shuffle: predictions are permuted against truths, destroying the pairing. This
preserves both marginal distributions (and hence value clustering, which a fixed 10% floor
gets badly wrong for a set like QM7) but it also destroys the model's accuracy. Against
this null, an elevated *hit rate* only establishes that the predictions are related to the
truth at all -- which is true of any competent model and is not evidence of recall.

MATCHED-BIN shuffle: predictions are permuted only within bins of similar truth value, so
the shuffled predictions remain about as accurate as the real ones while the molecule-level
identity is gone. This is the null that a hit rate must be tested against for the test to
mean "recall" rather than "competence". Both are computed; the matched-bin floor is the one
the conclusions rest on for hit rates, and R23 is reported against both.

Multiplicity
------------
A full screening matrix runs hundreds of tests. p-values are corrected across the whole
matrix with Benjamini-Hochberg; both raw and corrected values are reported, and a cell only
counts as contaminated on the corrected value.
"""
from __future__ import annotations
import json, os
import numpy as np

N_PERM_DEFAULT = 1000


# ---------------------------------------------------------------- digit utilities
def sig_round(x: float, n: int) -> str:
    return "0" if x == 0 else f"{x:.{n}g}"


def sig_figs(v: float) -> int:
    """Number of significant figures in the value as written."""
    if v == 0:
        return 1
    s = f"{v:.10g}".lstrip("-").replace(".", "").lstrip("0")
    return max(1, len(s.rstrip("0"))) if s else 1


def nested_matches(T: np.ndarray, P: np.ndarray):
    """Nested counts m1 >= m2 >= m3."""
    m1 = m2 = m3 = 0
    for tv, pv in zip(T, P):
        if sig_round(pv, 1) == sig_round(tv, 1):
            m1 += 1
            if sig_round(pv, 2) == sig_round(tv, 2):
                m2 += 1
                if sig_round(pv, 3) == sig_round(tv, 3):
                    m3 += 1
    return m1, m2, m3


# ---------------------------------------------------------------- digit codes
# The permutation test re-counts the same pairs a few thousand times. Rounding is a string
# format call, so doing it inside the loop costs ~70 s per deep cell and puts a full sweep
# of the matrix out of reach. Nothing about the rounding depends on the permutation, though:
# a shuffle only changes which prediction is compared against which truth. Rounding each
# value once, mapping the resulting strings to integers, and comparing integer arrays gives
# byte-identical counts (see `_selfcheck` below) about 300x faster.
def sig_codes(Tmol: np.ndarray, P: np.ndarray):
    """Integer codes for sig_round at n = 1, 2, 3, shared between truths and predictions.

    `Tmol` holds one truth per MOLECULE and `P` one value per PREDICTION; a pair is formed by
    an index `g` into Tmol. Returns ((t1, p1), (t2, p2), (t3, p3)) with t_n[i] == p_n[j] iff
    Tmol[i] and P[j] round to the same n significant figures.
    """
    out = []
    for n in (1, 2, 3):
        ts = np.array([sig_round(v, n) for v in Tmol])
        ps = np.array([sig_round(v, n) for v in P])
        _, inv = np.unique(np.concatenate([ts, ps]), return_inverse=True)
        out.append((inv[:len(ts)], inv[len(ts):]))
    return tuple(out)


def nested_from_codes(codes, g, sel=None):
    """Nested counts m1 >= m2 >= m3, pairing prediction i with the truth of molecule g[i].

    `sel` selects a subset of the predictions (used by the bootstrap, which resamples whole
    molecules); `g` must then be the molecule index of each selected prediction.
    """
    (t1, p1), (t2, p2), (t3, p3) = codes
    if sel is not None:
        p1, p2, p3 = p1[sel], p2[sel], p3[sel]
    e1 = t1[g] == p1
    e2 = e1 & (t2[g] == p2)
    e3 = e2 & (t3[g] == p3)
    return int(e1.sum()), int(e2.sum()), int(e3.sum())


def flat_from_codes(codes, g):
    """Unconditional counts h1, h2, h3."""
    return tuple(int((t[g] == p).sum()) for t, p in codes)


def _selfcheck(Tmol, P, codes, g):
    """Assert the fast path reproduces the reference counters exactly."""
    T = Tmol[g]
    assert nested_from_codes(codes, g) == nested_matches(T, P)
    assert flat_from_codes(codes, g) == flat_matches(T, P)


def flat_matches(T: np.ndarray, P: np.ndarray):
    """Unconditional counts h1, h2, h3.

    These are NOT redundant with the nested ones. Rounding is not transitive: t=0.949 and
    p=0.951 agree at two significant figures (both 0.95) but not at one (0.9 vs 1). The
    nested counter misses such pairs, so m2 slightly understates the true 2-sig agreement
    and R12 = m2/m1 is mildly biased. Reporting both makes the size of that effect visible
    instead of leaving it as an unstated assumption.
    """
    h1 = h2 = h3 = 0
    for tv, pv in zip(T, P):
        h1 += sig_round(pv, 1) == sig_round(tv, 1)
        h2 += sig_round(pv, 2) == sig_round(tv, 2)
        h3 += sig_round(pv, 3) == sig_round(tv, 3)
    return int(h1), int(h2), int(h3)


def ratio(a, b):
    return 100.0 * a / b if b else float("nan")


# ---------------------------------------------------------------- nulls
def matched_bin_permutation(T: np.ndarray, rng, n_bins: int = 20) -> np.ndarray:
    """Pair-level within-bin shuffle. Retained only as the diagnostic described below.

    Bins are equal-count quantiles of the truth, so the shuffled pairing keeps the model's
    accuracy roughly intact while the pairing is broken. The defect this cannot avoid is
    documented at `matched_bin_group_permutation`.
    """
    idx = np.arange(len(T))
    if len(T) < n_bins * 2:
        return rng.permutation(idx)
    edges = np.quantile(T, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    out = idx.copy()
    for b in range(n_bins):
        sel = idx[(T >= edges[b]) & (T < edges[b + 1])]
        if len(sel) > 1:
            out[sel] = rng.permutation(sel)
    return out


def _derange(sel: np.ndarray, rng) -> np.ndarray:
    """A uniformly random permutation of `sel` with no fixed point.

    Rejection sampling. The derangement fraction of all permutations tends to 1/e, so this
    costs about 2.7 draws regardless of size, and is exact rather than approximately-uniform.
    """
    m = len(sel)
    if m < 2:
        return sel
    ar = np.arange(m)
    for _ in range(64):
        p = rng.permutation(m)
        if not np.any(p == ar):
            return sel[p]
    # m == 2 always succeeds on the first draw with probability 1/2, so reaching here means
    # something pathological; a cyclic shift is a derangement for every m >= 2.
    return sel[(ar + 1) % m]


def quantile_bins(V: np.ndarray, n_bins: int = 20):
    """Index groups for equal-count quantile bins of V, computed once per cell.

    The bin edges do not depend on the permutation, so computing them inside the shuffle
    loop -- a quantile plus twenty boolean masks, two thousand times over -- costs more than
    the shuffle itself.
    """
    n = len(V)
    idx = np.arange(n)
    if n < 4:
        return [idx]
    n_bins = max(1, min(n_bins, n // 2))
    edges = np.quantile(V, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    out = []
    for b in range(n_bins):
        sel = idx[(V >= edges[b]) & (V < edges[b + 1])]
        if len(sel) > 1:
            out.append(sel)
    return out


def matched_bin_group_permutation(Tmol: np.ndarray, rng, n_bins: int = 20) -> np.ndarray:
    """Permutation of MOLECULES within quantile bins of the truth, with no fixed points.

    This is the null the verdicts rest on, and it differs from the pair-level version above
    in two ways that both matter.

    Molecule-level. Each molecule is queried several times, so a pair-level shuffle can map a
    prediction onto a *sibling iteration of the same molecule* -- which does not destroy
    identity at all, it preserves it. Whether that inflates the floor depends on how often
    the model reproduces its own answer across iterations, and a model that is reciting a
    stored value reproduces it almost always: on ESOL, Claude Opus 4.8 repeats its own value
    to three significant figures for 48% of molecules against 12% for GPT-4.1, and on the
    45-molecule positive control the rate is 89%. The pair-level floor is therefore inflated
    in direct proportion to the recall it is supposed to null out, which is the one dependency
    a null must not have. Permuting molecules, and comparing every prediction of a molecule
    against the truth of a different molecule, removes it.

    No fixed points. Within a bin of b molecules a plain permutation leaves each molecule
    with itself with probability 1/b; at 20 quantile bins over ~700 molecules that is ~3%,
    against observed hit rates of a few per cent. Sampling derangements removes that term
    exactly instead of accounting for it.

    Returns an index array `perm` such that molecule i is scored against the truth of
    molecule perm[i].
    """
    if len(Tmol) < 4:
        return rng.permutation(len(Tmol))
    return apply_bin_permutation(np.arange(len(Tmol)), quantile_bins(Tmol, n_bins), rng,
                                 derange=True)


def apply_bin_permutation(idx, bins, rng, derange: bool):
    """Shuffle within each precomputed bin, leaving everything else in place."""
    out = idx.copy()
    for sel in bins:
        out[sel] = _derange(sel, rng) if derange else rng.permutation(sel)
    return out


def sig_key_vec(x, n):
    """(mantissa, exponent) of x at n significant figures, vectorised.

    Equivalent to comparing `f"{x:.{n}g}"` strings, without the string. Two values agree at n
    significant figures iff they have the same key. Needed because the residual null below
    generates fresh values on every permutation, so the once-per-cell string codes that make
    the other nulls fast do not apply.
    """
    x = np.asarray(x, dtype=float)
    m = np.zeros(len(x), dtype=np.int64)
    e = np.zeros(len(x), dtype=np.int64)
    nz = x != 0
    if nz.any():
        ex = np.floor(np.log10(np.abs(x[nz]))).astype(np.int64)
        man = np.round(x[nz] / 10.0 ** (ex - (n - 1))).astype(np.int64)
        # Rounding 9.99 up to 10.0 carries into the exponent.
        carry = np.abs(man) >= 10 ** n
        man[carry] //= 10
        ex[carry] += 1
        m[nz], e[nz] = man, ex
    return m, e


def residual_permutation_stats(T, P, n_perm=N_PERM_DEFAULT, seed=0, pool=None, match_scale=True):
    """The accuracy-matched null done properly: resample the model's OWN residuals.

    The matched-bin null holds the *magnitude* of the truth fixed, not the model's error. When
    a model is very accurate that gap is enormous -- Claude Opus 4.8's median error on FreeSolv
    is 0.025 kcal/mol while a 20-quantile bin there is over a unit wide, so the bin null is
    asking how often a molecule collides with a neighbour a hundred median-errors away, which
    is not the same question.

    This null instead builds a synthetic predictor that is *exactly as accurate* as the real
    one and knows nothing about which molecule is which:

        y*_i = y_i + r_j ,   r drawn from the model's own residuals, j != i

    The error distribution is reproduced exactly, by construction. Anything the real
    predictions do beyond this is not explained by the model's accuracy.

    WITH `pool=None` THIS NULL IS DEGENERATE FOR EXACTLY THE CELLS THE STUDY IS ABOUT, and the
    reason is the finding itself. A model that recites has a point mass of residuals at exactly
    zero -- 46.2% of Claude Opus 4.8's FreeSolv residuals are 0.000, which is precisely its 3-sig
    hit rate. Resampling that pool onto other molecules reproduces the point mass, so the floor
    lands on top of the observation (46.7% against 46.2%, p = 0.62). The null inherits the
    signature it is supposed to exclude, which is the same defect as the pair-level matched-bin
    null and is worth reporting rather than hiding: *no null built from the model's own errors
    can detect a point mass in those errors.*

    The non-degenerate form passes `pool` -- residuals from a REFERENCE model that the audit
    finds clean on the same benchmark, rescaled (median absolute deviation) to the accuracy of
    the model under test. That asks the question the bin null only approximates: if a predictor
    were this accurate but had the error profile of something that is not reciting, how often
    would it agree to three significant figures?
    """
    rng = np.random.default_rng(seed)
    T = np.asarray(T, float)
    resid = np.asarray(P, float) - T
    if pool is not None:
        pool = np.asarray(pool, float)
        if match_scale:
            s_t = np.median(np.abs(resid))
            s_p = np.median(np.abs(pool))
            if s_p > 0 and np.isfinite(s_t):
                pool = pool * (s_t / s_p)
        resid = pool
    n = len(T)
    # The derangement below draws j != i, which needs at least two molecules to draw from. A
    # single scorable pair has no such draw and used to raise ValueError out of numpy, killing
    # the whole run on one degenerate cell -- which is how a ladder cell with one usable
    # prediction took down the analysis of the other 191.
    if n < 2 or len(resid) < 2:
        nan = np.full(n_perm, np.nan)
        return dict(m1=nan, m2=nan, m3=nan, R12=nan, R23=nan, hit2=nan, hit3=nan)
    keys_t = [sig_key_vec(T, k) for k in (1, 2, 3)]
    cm = np.empty((n_perm, 3))
    idx = np.arange(n)
    npool = len(resid)
    for i in range(n_perm):
        if npool == n:
            j = rng.integers(0, n - 1, size=n)
            j += (j >= idx)                 # sample j != i without rejection
        else:
            j = rng.integers(0, npool, size=n)
        star = T + resid[j]
        e1 = e2 = None
        counts = []
        for k in (1, 2, 3):
            mt, et = keys_t[k - 1]
            ms, es = sig_key_vec(star, k)
            eq = (mt == ms) & (et == es)
            e1 = eq if e1 is None else (e1 & eq)
            counts.append(int(e1.sum()))
        cm[i] = counts
    c1, c2, c3 = cm[:, 0], cm[:, 1], cm[:, 2]
    return dict(m1=c1, m2=c2, m3=c3,
                R12=100 * c2 / np.maximum(c1, 1),
                R23=100 * c3 / np.maximum(c2, 1),
                hit2=100 * c2 / max(n, 1),
                hit3=100 * c3 / max(n, 1))


def permutation_stats(Tmol, P, G, codes, n_perm=N_PERM_DEFAULT, seed=0, mode="global",
                      n_bins=20):
    """Null distributions of (m1, m2, m3, R12, R23, hit) under one of three shuffles.

    mode='global'        pair-level shuffle of everything. This is what the prior literature
                         does and it is reported for comparability, not relied on: it
                         destroys the model's accuracy along with the pairing, so any
                         accurate model clears it.
    mode='matched'       molecule-level derangement within quantile bins -- accuracy held
                         roughly fixed, identity destroyed. The verdicts rest on this one.
    mode='matched_pair'  the pair-level within-bin shuffle, kept solely to measure how much
                         the iteration-leakage described above was inflating the floor.
    """
    rng = np.random.default_rng(seed)
    n_pair = len(P)
    if mode == "matched":
        idx, bins = np.arange(len(Tmol)), quantile_bins(Tmol, n_bins)
    elif mode == "matched_pair":
        idx, bins = np.arange(n_pair), quantile_bins(Tmol[G], n_bins)
    cm = np.empty((n_perm, 3))
    for i in range(n_perm):
        if mode == "matched":
            g = apply_bin_permutation(idx, bins, rng, derange=True)[G]
        elif mode == "matched_pair":
            g = G[apply_bin_permutation(idx, bins, rng, derange=False)]
        else:
            g = G[rng.permutation(n_pair)]
        cm[i] = nested_from_codes(codes, g)
    c1, c2, c3 = cm[:, 0], cm[:, 1], cm[:, 2]
    return dict(m1=c1, m2=c2, m3=c3,
                R12=100 * c2 / np.maximum(c1, 1),
                R23=100 * c3 / np.maximum(c2, 1),
                hit2=100 * c2 / max(n_pair, 1),
                hit3=100 * c3 / max(n_pair, 1))


def p_upper(null: np.ndarray, obs: float) -> float:
    """One-sided p = P(statistic >= observed | null), with the +1 correction."""
    if not np.isfinite(obs):
        return float("nan")
    return (1 + int(np.sum(null >= obs))) / (len(null) + 1)


def benjamini_hochberg(pvals):
    """BH-adjusted p-values (q-values). NaNs pass through as NaN and are excluded."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        q[i] = prev
    return q


# ---------------------------------------------------------------- result I/O
def leaves(o):
    """Yield (key, list-of-numbers) from the nested result JSON, at any depth."""
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                yield k, v
            else:
                yield from leaves(v)


def reasoning_settings(path):
    """The reasoning settings a result file holds, at the second level of nesting."""
    d = json.load(open(path))
    out = set()
    for variant, styles in d.items():
        if isinstance(styles, dict):
            out |= set(styles.keys())
    return out


def load_preds(path, reasoning=None):
    """Predictions keyed by molecule, optionally restricted to one reasoning setting.

    A cell queried before and after the reasoning calibration holds two branches, and for
    some models the setting changes the answer -- Gemini 3.5 Flash reproduces ESOL values
    under a 128-token thinking budget and not under `minimal`. Pooling the branches would
    average two different experiments, so the caller pins the setting the cell is supposed
    to have been run at and falls back to everything present only if that branch is absent.
    """
    d = json.load(open(path))
    if reasoning is not None:
        sub = {v: {reasoning: s[reasoning]} for v, s in d.items()
               if isinstance(s, dict) and reasoning in s}
        if sub:
            d = sub
    out = {}
    for mol, vals in leaves(d):
        nums = [float(x) for x in vals
                if x is not None and not (isinstance(x, float) and np.isnan(x))]
        if nums:
            out.setdefault(str(mol).strip(), []).extend(nums)
    return out


# ---------------------------------------------------------------- the cell analysis
def _boot_ci(codes, G, n_boot=1000, seed=0, level=3):
    """Percentile bootstrap for the hit rate and the deep rung, resampling MOLECULES.

    The unit of independence is the molecule, not the prediction: three repeats of one molecule
    are one observation queried three times, and resampling predictions would shrink the
    interval by roughly the square root of the iteration count. Molecules are drawn with
    replacement and all of their predictions travel with them.
    """
    rng = np.random.default_rng(seed)
    n_mol = int(G.max()) + 1 if len(G) else 0
    if n_mol < 10:
        return dict(hit_lo=np.nan, hit_hi=np.nan, deep_lo=np.nan, deep_hi=np.nan)
    order = np.argsort(G, kind="stable")
    Gs = G[order]
    starts = np.searchsorted(Gs, np.arange(n_mol), side="left")
    ends = np.searchsorted(Gs, np.arange(n_mol), side="right")
    hits, deeps = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_mol, size=n_mol)
        idx = np.concatenate([order[starts[m]:ends[m]] for m in pick])
        m1, m2, m3 = nested_from_codes(codes, G[idx], sel=idx)
        hits[b] = ratio(m3 if level == 3 else m2, len(idx))
        deeps[b] = ratio(m3, m2) if level == 3 else ratio(m2, m1)
    f = lambda a: (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))
    hl, hh = f(hits)
    dl, dh = f(deeps)
    return dict(hit_lo=hl, hit_hi=hh, deep_lo=dl, deep_hi=dh)


def analyse_cell(truth_lut: dict, preds: dict, n_perm=N_PERM_DEFAULT, seed=0, level=3,
                 n_bins=20, pad_preds=False):
    """All statistics for one (model, dataset, variant) cell.

    level=3 scores 3-significant-figure agreement over pairs where truth and prediction
    both carry >=3 significant figures; level=2 does the 2-sig companion, which keeps power
    for models that report a property coarsely (Claude Sonnet 5 gives logD to two figures,
    leaving 11 usable 3-sig pairs out of 3000 -- that cell is UNTESTABLE, not clean).

    `pad_preds` drops the requirement on the PREDICTION only, so an answer of "2.2" is read as
    2.20. This is legitimate ONLY when the prompt asked for `level` significant figures, as the
    controlled-budget arm does: under that instruction a short answer is the model's assertion
    at the requested precision, not a refusal to commit, and reading it literally is the
    conservative choice -- it can only turn an excluded pair into a scored one, and a scored
    pair is a miss unless the truth's own trailing digit happens to agree.

    It also removes the collider that motivates most of the caveats elsewhere in this file: the
    denominator stops depending on the model's precision choice, which is causally downstream of
    whether the model is reciting. The floor moves with it, because the permutation null is
    computed from exactly the same pairs.

    Do NOT use it on the main map, whose prompt never requested a precision.
    """
    # G indexes each surviving prediction to its molecule, which is what lets the null
    # permute molecules rather than predictions. Without it, repeated iterations of one
    # molecule can be shuffled onto each other and the floor absorbs the recall it is
    # measuring (see matched_bin_group_permutation).
    ally, allp, Tmol, P, G = [], [], [], [], []
    for mol, plist in preds.items():
        tv = truth_lut.get(mol)
        if tv is None:
            continue
        kept = [pv for pv in plist
                if sig_figs(tv) >= level and (pad_preds or sig_figs(pv) >= level)]
        for pv in plist:
            ally.append(tv); allp.append(pv)
        if kept:
            gi = len(Tmol)
            Tmol.append(tv)
            P.extend(kept)
            G.extend([gi] * len(kept))
    ally, allp = np.array(ally), np.array(allp)
    Tmol, P, G = np.array(Tmol), np.array(P), np.array(G, dtype=int)
    if len(P) == 0 or len(ally) == 0:
        return None
    T = Tmol[G]

    codes = sig_codes(Tmol, P)
    m1, m2, m3 = nested_from_codes(codes, G)
    h1, h2, h3 = flat_from_codes(codes, G)
    R12, R23 = ratio(m2, m1), ratio(m3, m2)
    hit3 = ratio(m3, len(T))
    # Hit rate over EVERY prediction, not only those the model chose to report to >=3
    # figures. The >=3-sig filter conditions on model behaviour: a model that emits a full
    # decimal only when it is reciting will have its hit rate flattered by that filter.
    hit3_all = ratio(m3, len(ally))

    glob = permutation_stats(Tmol, P, G, codes, n_perm, seed, mode="global")
    matched = permutation_stats(Tmol, P, G, codes, n_perm, seed + 1, mode="matched",
                                n_bins=n_bins)
    mpair = permutation_stats(Tmol, P, G, codes, n_perm, seed + 2, mode="matched_pair",
                              n_bins=n_bins)
    resid = residual_permutation_stats(T, P, n_perm, seed + 3)

    from scipy.stats import pearsonr, spearmanr
    rmse = float(np.sqrt(np.mean((allp - ally) ** 2)))
    medae = float(np.median(np.abs(allp - ally)))
    r = pearsonr(allp, ally)[0] if len(ally) > 2 else float("nan")
    rho = spearmanr(allp, ally)[0] if len(ally) > 2 else float("nan")

    # The headline statistic depends on the precision the cell is scored at. At level 3 it is
    # the 3-sig hit rate with R23 as the deep rung; at level 2 -- used when a model reports
    # the property to only two figures, so the 3-sig test is not merely negative but
    # impossible -- it is the 2-sig hit rate with R12 as the deep rung. Reporting a 3-sig
    # number for a level-2 run would silently answer a question the data cannot answer.
    hit_key = "hit3" if level == 3 else "hit2"
    deep_key = "R23" if level == 3 else "R12"
    hit_lvl = hit3 if level == 3 else ratio(m2, len(T))
    deep = R23 if level == 3 else R12

    return dict(
        n_pred=len(ally), n_usable=len(T), n_mol=len(Tmol), level=level, n_bins=n_bins,
        # How much the pair-level within-bin shuffle was inflating the floor. Reported per
        # cell so the correction is auditable rather than asserted.
        mp_chance_hit=float(mpair[hit_key].mean()), mp_p_hit=p_upper(mpair[hit_key], hit_lvl),
        mp_chance_deep=float(mpair[deep_key].mean()), mp_p_deep=p_upper(mpair[deep_key], deep),
        # The residual-resampling null: a synthetic predictor with exactly this model's error
        # distribution and no knowledge of molecule identity.
        rs_chance_hit=float(resid[hit_key].mean()), rs_p_hit=p_upper(resid[hit_key], hit_lvl),
        rs_chance_deep=float(resid[deep_key].mean()), rs_p_deep=p_upper(resid[deep_key], deep),
        rs_chance_hit3=float(resid["hit3"].mean()), rs_p_hit3=p_upper(resid["hit3"], hit3),
        rs_chance_R23=float(resid["R23"].mean()), rs_p_R23=p_upper(resid["R23"], R23),
        pearson=r, spearman=rho, rmse=rmse, medae=medae,
        m1=m1, m2=m2, m3=m3, h1=h1, h2=h2, h3=h3,
        hit=hit_lvl, hit_stat=hit_key, deep=deep, deep_stat=deep_key,
        hit3=hit3, hit3_all=hit3_all, R12=R12, R23=R23,
        chance_hit=float(glob[hit_key].mean()), p_hit=p_upper(glob[hit_key], hit_lvl),
        chance_deep=float(glob[deep_key].mean()), p_deep=p_upper(glob[deep_key], deep),
        mb_chance_hit=float(matched[hit_key].mean()), mb_p_hit=p_upper(matched[hit_key], hit_lvl),
        mb_chance_deep=float(matched[deep_key].mean()), mb_p_deep=p_upper(matched[deep_key], deep),
        # Bootstrap intervals on the two headline statistics. Resampling is over MOLECULES,
        # not predictions: the repeats of one molecule are not independent observations, and
        # resampling them separately would understate the interval by roughly sqrt(iters).
        **_boot_ci(codes, G, n_boot=1000, seed=seed + 4, level=level),
        chance_hit3=float(glob["hit3"].mean()), p_hit3=p_upper(glob["hit3"], hit3),
        chance_R12=float(glob["R12"].mean()), p_R12=p_upper(glob["R12"], R12),
        chance_R23=float(glob["R23"].mean()), p_R23=p_upper(glob["R23"], R23),
        mb_chance_hit3=float(matched["hit3"].mean()), mb_p_hit3=p_upper(matched["hit3"], hit3),
        mb_chance_R12=float(matched["R12"].mean()), mb_p_R12=p_upper(matched["R12"], R12),
        mb_chance_R23=float(matched["R23"].mean()), mb_p_R23=p_upper(matched["R23"], R23),
    )


def classify(row, alpha=0.05, min_m2=15):
    """Regime label for one cell.

    Two independent tests must be combined, because they fail in opposite directions:

      * R23 (retention) is the specific test for verbatim recall, but it conditions on the
        2-sig matches and therefore has little power when m2 is small -- exactly the
        situation for a model that recalls only a minority of the benchmark.
      * the 3-sig hit rate against the MATCHED-BIN null is the powerful test. Against the
        global shuffle a hit rate is confounded by accuracy and means nothing; against the
        matched-bin null the model's accuracy is held roughly fixed, so surviving it does
        indicate recall.

    'untestable' is a distinct verdict from 'clean', and the distinction matters: too few
    usable pairs means the detector had no power, which is absence of evidence, not
    evidence of absence.
    """
    # The conditioning count for the deep rung: m2 at 3-sig (R23 = m3/m2), m1 at 2-sig
    # (R12 = m2/m1). Too few and the rung is noise, not a negative result.
    cond = row.get("m2", 0) if row.get("level", 3) == 3 else row.get("m1", 0)
    if not np.isfinite(row.get("deep", np.nan)) or cond < min_m2:
        return "untestable"
    # A permutation floor of zero does not mean the coincidence rate is zero, only that it is
    # below what this sample can resolve: one hit in n_usable. Clipping to a fixed 0.01 instead
    # makes the excess used for classification differ from the excess printed in the figures,
    # so a reader cannot recompute a regime label from the released tables.
    resolvable = 100.0 / max(row.get("n_usable", 1), 1)
    floor = max(row.get("mb_chance_hit", 0.0), resolvable)
    excess = row.get("hit", 0.0) / floor
    # BOTH tests are taken against the accuracy-matched null. Using the global shuffle for
    # the retention rung reintroduces exactly the confound the matched null exists to remove:
    # on QM7 the two most accurate models clear the global R12 floor (20.3%) while sitting
    # exactly on the accuracy-matched one (26.3%), and would otherwise be called contaminated
    # for being good at quantum chemistry. The global floors are still reported, for
    # comparability with the prior literature, but no verdict rests on them.
    # The verdict is an OR of two tests, so the two must be corrected as ONE family or the
    # flagged set is controlled at 2*alpha rather than alpha. `q_*_joint` is that correction;
    # the separately-corrected `q_*_mb` are kept in the tables for comparability and are used
    # only if a caller supplies a row without the joint columns.
    qd = row.get("q_deep_joint", row.get("q_deep_mb", np.nan))
    qh = row.get("q_hit_joint", row.get("q_hit_mb", np.nan))
    deep = np.isfinite(qd) and qd < alpha
    powerful = np.isfinite(qh) and qh < alpha
    # Significance without effect size is not a finding. Claude Sonnet 5 on QM7 clears the
    # deep rung at q = 0.03 with a hit rate 1.08x its floor; on 3,000 predictions almost any
    # systematic nudge reaches significance, so an effect-size gate keeps the label honest.
    if deep and excess >= 4:
        return "heavy"
    if (deep and excess >= 1.5) or (powerful and excess >= 2):
        return "partial"
    if deep or powerful:
        return "trace"
    return "clean"
