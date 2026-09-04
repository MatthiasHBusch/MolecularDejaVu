"""
The detector, run over the controlled-budget arm, and compared against the zero-shot map.

The budget sweep writes one file per BLOCK, not per cell, so that a cell can be extended later
without re-purchasing what it already has:

    results/budget/<dataset>__<tag>__t<cap>__<offset>-<end>.json

This concatenates every block of a cell before analysing it, and refuses to do so if two
blocks share a molecule -- disjointness is guaranteed by construction (one fixed permutation,
non-overlapping slices), but a silent overlap would double-weight a molecule inside the
molecule-level null, which is the one place this pipeline cannot afford it.

Two things make this arm NOT directly comparable to the zero-shot map, and both are reported
rather than hidden:

  1. the prompt asks for three significant figures, which changes the denominator of hit3;
  2. five endpoints ignore `reasoning.max_tokens` and were run at `effort: low` instead, so the
     arm is deliberation-controlled, not token-matched.

*** The precision instruction does NOT remove the hedging collider, contrary to what this
docstring claimed when it was written. Measured on Claude Opus 5, same instruction, 250
molecules per benchmark: three-figure answers on 76% of ESOL, 76% of FreeSolv, 59% of LD50 --
and 5% of Lipophilicity, the one benchmark of the four it does not recall. Asked in plain
language for three significant figures, the model complies where it is reciting and declines
where it is guessing. Emission is coupled to recall strongly enough to survive being
instructed otherwise, so hit3 remains conditioned on a mediator of the treatment and the
collider argument stands. It also means Lipophilicity cannot be made testable at 3 significant
figures by buying more molecules: 5% of 1000 is 50 usable pairs and an m2 of about 3. For that
benchmark the 2-sig companion (R12) is the only route to a verdict. ***

What IS comparable is the verdict: whether a cell is flagged, at the same alpha, under the same
accuracy-matched null. That comparison is the output.

    python src/analyze_budget.py
    python src/analyze_budget.py --perm 500      # faster while the sweep is still running
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import analyse_cell, benjamini_hochberg, classify

# heavy/partial/trace are flags; 'clean' is a pass; 'untestable' is neither.
FLAGGED = ("heavy", "partial", "trace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, REG = os.path.join(ROOT, "results"), os.path.join(ROOT, "src", "registry")
SCREEN = os.path.join(ROOT, "data", "screening")
BUD = os.path.join(RES, "budget")


def truth_lut(dskey):
    f = os.path.join(SCREEN, f"{dskey}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f).dropna(subset=["value"])
    return {str(k).strip(): float(v) for k, v in zip(df.mol_id, df.value) if pd.notna(k)}


def arm_of(meta):
    """Which arm a block belongs to: 't1024' (thinking budget), 'reg' (the endpoint minimum),
    and 'r' suffixed where the prompt carried a randomised SMILES rather than the published one.

    The arms ask the SAME molecules on purpose -- that is what makes the comparison paired --
    so they must never be concatenated into one cell. Before this existed, a second arm on an
    already-swept benchmark tripped the disjointness guard, which is the right failure but the
    wrong diagnosis.

    The variant belongs in here and not only in the filename, because this is what the analysis
    keys on. A randomised block carries `arm: budget` and `thinking_cap: 1024` like any other, so
    without the suffix it would land in the canonical cell and be averaged into the rate it exists
    to be compared against -- the failure would look like a disjointness error two steps later.
    """
    base = "reg" if meta.get("arm") == "registry" else f"t{meta.get('thinking_cap', 1024)}"
    return base if meta.get("variant", "canonical") == "canonical" else f"{base}r"


def cells(arm=None):
    """Every (dataset, tag, arm) with its blocks concatenated, and its metadata."""
    by = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(BUD, "*.json"))):
        if os.path.basename(p).startswith("_"):    # _target.json is a manifest, not a block
            continue
        try:
            d = json.load(open(p))
        except json.JSONDecodeError:
            # A sweep writing this cell right now. Analysing mid-campaign is normal here, and
            # one half-written block is not a reason to abandon the other 150.
            print(f"  [skip] {os.path.basename(p)}: still being written", flush=True)
            continue
        a = arm_of(d["meta"])
        if arm and a != arm:
            continue
        by[(d["meta"]["dataset"], d["meta"]["tag"], a)].append((p, d))
    for (dk, tag, a), blocks in sorted(by.items()):
        preds, seen, meta = defaultdict(list), {}, blocks[0][1]["meta"]
        n_trunc = n_call = 0
        for p, d in blocks:
            for c in d["calls"]:
                mid = str(c["mol_id"]).strip()
                n_call += 1
                n_trunc += bool(c.get("truncated"))
                if mid in seen and seen[mid] != os.path.basename(p):
                    raise SystemExit(
                        f"OVERLAP: {mid} appears in both {seen[mid]} and {os.path.basename(p)}. "
                        f"Blocks must be disjoint; re-check --offset.")
                seen[mid] = os.path.basename(p)
                if c.get("value") is not None:
                    preds[mid].append(float(c["value"]))
        yield dk, tag, a, dict(preds), dict(
            arm=a, blocks=len(blocks), n_call=n_call, n_queried=len(seen),
            trunc=100.0 * n_trunc / max(n_call, 1),
            reasoning=meta.get("reasoning", f"max_tokens:{meta.get('thinking_cap')}"),
            cost=sum(b[1]["meta"].get("cost", 0.0) for b in blocks),
            think=float(np.median([c.get("reasoning_tokens", 0)
                                   for _, d in blocks for c in d["calls"]] or [0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bins", type=int, default=20)
    # The prompt asks for three significant figures, so an answer of "2.2" is the model's
    # assertion at the requested precision and is read as 2.20. Legitimate here and ONLY here;
    # the main map never requested a precision. See analyse_cell(pad_preds=...).
    ap.add_argument("--pad", action="store_true",
                    help="read a short answer as zero-padded to 3 significant figures")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default=None,
                    help="restrict to one arm ('t1024' or 'reg'); default analyses every arm "
                         "on disk and writes an `arm` column")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(RES, "budget_3sig_pad.csv" if args.pad
                                else "budget_3sig.csv")

    models = {m["tag"]: m for m in json.load(open(os.path.join(REG, "models.json")))["models"]}
    dsets = {d["key"]: d for d in json.load(open(os.path.join(REG, "datasets.json")))["datasets"]}
    luts, rows = {}, []
    for dk, tag, arm, preds, info in cells(args.arm):
        luts.setdefault(dk, truth_lut(dk))
        if luts[dk] is None:
            continue
        stats = analyse_cell(luts[dk], preds, n_perm=args.perm, level=3, n_bins=args.bins,
                             pad_preds=args.pad)
        if stats is None:
            print(f"  skip {dk}/{tag}: no usable pairs")
            continue
        rows.append(dict(dataset=dk, dataset_name=dsets.get(dk, {}).get("name", dk),
                         ds_class=dsets.get(dk, {}).get("class", "?"),
                         model=models.get(tag, {}).get("name", tag), tag=tag,
                         vendor=models.get(tag, {}).get("vendor", "?"), **info, **stats))
        print(f"  {arm:5s} {dk:14s} {tag:16s} n={stats['n_usable']:5d} m2={stats['m2']:5d} "
              f"hit3={stats['hit3']:6.2f} floor={stats['mb_chance_hit3']:5.2f} "
              f"R23={stats['deep']:6.1f} think={info['think']:5.0f}", flush=True)

    if not rows:
        sys.exit("no budget results yet")
    R = pd.DataFrame(rows)

    # One BH family PER ARM, both tests corrected jointly -- same rule as the map. Correcting
    # across arms would pool two experiments into one family; they are separate measurements of
    # the same cells and are compared by verdict, not merged.
    R["q_hit_joint"], R["q_deep_joint"] = np.nan, np.nan
    for a, idx in R.groupby("arm").groups.items():
        sub = R.loc[idx]
        qj = benjamini_hochberg(np.concatenate([sub.mb_p_hit.to_numpy(),
                                                sub.mb_p_deep.to_numpy()]))
        R.loc[idx, "q_hit_joint"] = qj[:len(sub)]
        R.loc[idx, "q_deep_joint"] = qj[len(sub):]
    R["regime"] = [classify(r, alpha=args.alpha) for _, r in R.iterrows()]
    R.round(4).to_csv(args.out, index=False)
    # REMOVED 25 Aug 2026: a per-cell comparison of this arm against the zero-shot screen.
    # The screen is retired and its files are under results/_archive_zeroshot/. The
    # comparison the paper actually reports is this arm against the minimum-reasoning arm,
    # which analyze_reasoning_delta.py computes as a paired delta over identical molecules.
    # A free internal control, and it is worth as much as anything else in this arm.
    #
    # The budget is advisory in BOTH directions: Kimi K2.6 emitted 9,879 tokens against a
    # 1,024 request, and Claude Sonnet 5, GPT-4.1, Qwen3-235B, Llama 4 Maverick and Mistral
    # Large emit ZERO. For a cell that emitted no thinking tokens, this arm is not a
    # deliberation manipulation at all -- it is the original map with one extra sentence in
    # the prompt. Those cells therefore isolate the prompt change from the budget change,
    # which is exactly the confound that otherwise makes the two arms non-comparable.
    #
    # Read the two blocks against each other: if hit3 moves in the thinking cells and not in
    # the non-thinking ones, the movement is deliberation and not the precision instruction.
    R["thought"] = R.think > 0
    print(f"\n{moved} of {len(R)} cells change verdict under a deliberation budget.")
    if R.thought.nunique() > 1:
        print("\nIS THE MOVEMENT THE BUDGET OR THE PROMPT?")
        print("  cells that emitted no thinking tokens see only the prompt change.")
        j = R.set_index(["dataset", "tag"]).join(
            m.rename(columns={"hit3": "map_hit3", "regime": "map_regime",
                              "reasoning": "map_reasoning"}), how="inner")
        j["delta"] = j.hit3 - j.map_hit3
        for lab, sel in (("emitted thinking", j.thought), ("emitted none", ~j.thought)):
            d = j[sel]
            if not len(d):
                continue
            flag = d.regime.isin(FLAGGED)
            print(f"    {lab:18s} n={len(d):3d}   median dhit3 {d.delta.median():+7.2f}   "
                  f"clean->flagged {int(((d.map_regime == 'clean') & flag).sum())}   "
                  f"clean->untestable "
                  f"{int(((d.map_regime == 'clean') & (d.regime == 'untestable')).sum())}")
    mm = m.loc[[k for k in zip(R.dataset, R.tag) if k in m.index]]
    print("")
    print(f"{'':18s}{'flagged':>9s}{'clean':>9s}{'untestable':>12s}")
    for lab, v in (("map", mm.regime), ("budget arm", R.regime)):
        print(f"  {lab:16s}{int(v.isin(FLAGGED).sum()):9d}{int((v == 'clean').sum()):9d}"
              f"{int((v == 'untestable').sum()):12d}")
    print("\n  NOTE: untestable is not a pass and not a flag. At 250 molecules a cell whose"
          "\n  2-sig match count falls below 15 loses its verdict entirely, which is the price"
          "\n  of this block size and hits the low-recall models hardest.")
    print(f"\nmedian truncation {R.trunc.median():.2f}%   "
          f"spend ${R.cost.sum():.2f}   saved {args.out}")


if __name__ == "__main__":
    main()
