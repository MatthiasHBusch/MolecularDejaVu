"""
Dose-response: verbatim recall against the deliberation actually spent.

Reads results/ladder/<dataset>__<tag>__<level>.json -- the controlled sweep from
run_reasoning_ladder.py, where the molecule set, the iteration count and the day are held
fixed and only the reasoning effort varies -- and reports, per cell:

    hit3        3-sig verbatim agreement, against the accuracy-matched molecule-level floor
    medAE       median absolute error, which is what improves if deliberation buys COMPUTATION
    selfcons    reproducibility of the model's own answer across the three iterations
    reasoning   the median number of thinking tokens the endpoint actually emitted

The discriminating comparison is between a benchmark the models recall and one they do not.
If deliberation merely makes a model better at chemistry, medAE improves on both and hit3
stays at its floor on both. If deliberation opens a retrieval path, hit3 climbs on the
recalled benchmark only -- and climbs far faster than accuracy alone can explain.

Requested effort is not the dose. Asked for "minimal", "low" and "medium", GPT-5.5 emits 516
thinking tokens every time; asked for the same three, Gemini 3 Flash emits 0, 542 and 2,974.
Everything here is therefore plotted and tabulated against emitted tokens as well as against
the label.

    python src/analyze_ladder.py
    python src/analyze_ladder.py --perm 4000
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import analyse_cell, sig_round

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, SCREEN, REG = (os.path.join(ROOT, "results"), os.path.join(ROOT, "data", "screening"),
                    os.path.join(ROOT, "src", "registry"))
LADDER = os.path.join(RES, "ladder")
LEVEL_ORDER = ["none", "minimal", "low", "medium", "high"]


def truth_lut(dskey):
    df = pd.read_csv(os.path.join(SCREEN, f"{dskey}.csv")).dropna(subset=["value"])
    return {str(k).strip(): float(v) for k, v in zip(df["mol_id"], df["value"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RES, "ladder_summary.csv"))
    args = ap.parse_args()

    models = {m["tag"]: m for m in json.load(open(os.path.join(REG, "models.json")))["models"]}

    # A ladder cell is only a dose-response if every rung was DESIGNED on the same molecules, and
    # one cell was not: esol/gem35flash/high was re-run at 150 molecules on 28 Jul while its own
    # minimal/low/medium stayed at 60. The head sample is recalled far more often than the tail,
    # so the extra 90 molecules dilute the top rung and understate that cell's climb.
    #
    # The restriction is on the DESIGNED set (meta n_mol, i.e. the head-N of the benchmark file),
    # never on the molecules that happened to come back: a call whose retries were exhausted is
    # simply absent from its level's file, so intersecting the observed ids would compound one
    # level's failures onto every other level -- which cost 32 cells up to a third of their
    # molecules when it was tried.
    files = sorted(glob.glob(os.path.join(LADDER, "*.json")))
    nmin = {}
    for p in files:
        m = json.load(open(p))["meta"]
        k = (m["dataset"], m["tag"])
        nmin[k] = min(nmin.get(k, 10 ** 9), int(m["n_mol"]))

    def designed(dskey, n):
        df = pd.read_csv(os.path.join(SCREEN, f"{dskey}.csv")).dropna(subset=["value"])
        return set(df.mol_id.astype(str).str.strip().head(n))

    luts, rows = {}, []
    for p in files:
        d = json.load(open(p))
        meta, calls = d["meta"], d["calls"]
        dskey = meta["dataset"]
        n_want = nmin[(dskey, meta["tag"])]
        if int(meta["n_mol"]) > n_want:
            keep = designed(dskey, n_want)
            got = {str(c["mol_id"]).strip() for c in calls}
            # The restriction assumes the oversized rung is a head sample, so the small set is
            # nested inside it. When it is not -- esol/gem35flash/high was re-run on a RANDOM
            # 150, overlapping the head-60 by 15 molecules -- restricting silently leaves a
            # handful of pairs and a rate computed on them looks like a measurement. Say so.
            if len(keep & got) < 0.9 * len(keep):
                print(f"  [WARN] {dskey}/{meta['tag']}/{meta['level']}: {meta['n_mol']} molecules "
                      f"overlap its siblings' {n_want} by only {len(keep & got)}. Not a paired "
                      f"rung; re-run it on the designed set or drop it.")
            calls = [c for c in calls if str(c["mol_id"]).strip() in keep]
            print(f"  [pair] {dskey}/{meta['tag']}/{meta['level']}: "
                  f"{meta['n_mol']} -> {len(keep & got)} molecules, to match its own other rungs")
        if dskey not in luts:
            luts[dskey] = truth_lut(dskey)
        lut = luts[dskey]

        # A completion cut off at the token cap still parses, and what it yields is NOT an
        # answer: the parser takes the last number in the text, which in a truncated reply is a
        # fragment of the arithmetic. Measured on qm7/gem35flash/high, the truncated calls sit at
        # medAE 1568 against 89 for the intact ones, and their "predictions" are 1.0, 2.0, 7.0,
        # 19.0 on a benchmark whose values run to -1500. They are missing answers wearing a
        # number, and they corrupt medAE and the rank correlation far more than they touch hit3.
        preds, cut = {}, 0
        for c in calls:
            if c.get("finish") == "length":
                cut += 1
                continue
            if c["value"] is None:
                continue
            preds.setdefault(c["mol_id"], []).append(float(c["value"]))
        if cut:
            print(f"  [cut] {dskey}/{meta['tag']}/{meta['level']}: dropped {cut}/{len(calls)} "
                  f"truncated completions")
        if not preds:
            continue
        s = analyse_cell(lut, preds, n_perm=args.perm, level=3)
        if s is None:
            continue

        # Self-consistency: the model reproducing its OWN answer across iterations. Retrieval
        # is reproducible; a guess at temperature 1 is not, so this separates the two
        # independently of whether the answer happens to be right.
        cons = [all(sig_round(v, 3) == sig_round(vals[0], 3) for v in vals[1:])
                for vals in preds.values() if len(vals) > 1]
        rt = [c["reasoning_tokens"] for c in calls]
        rows.append(dict(
            dataset=dskey, tag=meta["tag"], model=models.get(meta["tag"], {}).get("name", meta["tag"]),
            level=meta["level"], n_mol=len(preds), n_calls=len(calls),
            parsed=100.0 * sum(c["value"] is not None for c in calls) / max(len(calls), 1),
            reasoning_med=float(np.median(rt)) if rt else 0.0,
            reasoning_mean=float(np.mean(rt)) if rt else 0.0,
            cost=meta.get("cost", np.nan),
            hit1=100.0 * s["h1"] / max(s["n_usable"], 1),
            hit2=100.0 * s["h2"] / max(s["n_usable"], 1),
            hit3=s["hit3"], floor=s["mb_chance_hit3"], p_hit=s["mb_p_hit3"],
            R23=s["R23"], m2=s["m2"], m3=s["m3"], n_usable=s["n_usable"],
            medae=s["medae"], spearman=s["spearman"],
            selfcons=100.0 * float(np.mean(cons)) if cons else np.nan))

    if not rows:
        sys.exit(f"no ladder results in {LADDER} -- run src/run_reasoning_ladder.py first")
    R = pd.DataFrame(rows)
    R["lvl"] = R.level.apply(lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 99)
    R = R.sort_values(["dataset", "tag", "lvl"]).drop(columns="lvl")
    R.round(4).to_csv(args.out, index=False)

    show = ["level", "reasoning_med", "n_usable", "m2", "hit3", "floor", "p_hit", "R23",
            "medae", "spearman", "selfcons", "parsed"]
    for (ds, tag), g in R.groupby(["dataset", "tag"]):
        # A permutation floor of zero means the coincidence rate is below what the sample can
        # resolve -- one hit in n_usable -- not that it is zero. Dividing by a clipped 0.01
        # prints an 86x excess on a cell with four hits.
        resolvable = 100.0 / g.n_usable.clip(lower=1)
        floor = np.maximum(g.floor, resolvable)
        excess = g.hit3 / floor
        bounded = g.floor < resolvable
        print(f"\n=== {ds} / {g.model.iloc[0]} ===")
        print(g[show].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
        print("    hit3/floor: " + "  ".join(
            f"{l}={'>' if b else ''}{e:.1f}x" for l, e, b in zip(g.level, excess, bounded)))

    # The headline contrast, in one place: what does more deliberation buy on a benchmark the
    # models recall, versus one they do not?
    print("\n" + "=" * 78)
    print("What deliberation buys, none -> highest level actually distinct")
    print("=" * 78)
    print(f"{'dataset':14s}{'model':22s}{'reasoning':>12}{'hit3':>16}{'medAE':>16}")
    for (ds, tag), g in R.groupby(["dataset", "tag"]):
        g = g.sort_values("reasoning_med")
        lo, hi = g.iloc[0], g.iloc[-1]
        if hi.reasoning_med <= lo.reasoning_med:
            continue
        print(f"{ds:14s}{g.model.iloc[0]:22s}"
              f"{f'{lo.reasoning_med:.0f}->{hi.reasoning_med:.0f}':>12}"
              f"{f'{lo.hit3:.2f}->{hi.hit3:.2f}':>16}"
              f"{f'{lo.medae:.3f}->{hi.medae:.3f}':>16}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
