"""
Look at the actual replies for molecules whose recall switches on with the thinking budget.

Everything else in this repo argues from summary statistics. This prints the raw material: for
each molecule that the model misses at zero thinking tokens and reproduces verbatim at a high
budget, the published value and what the model actually said at each level.

    python src/inspect_switchers.py --dataset ld50 --tag gem3flash
    python src/inspect_switchers.py --dataset ld50 --tag gem3flash --show non-switchers
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import sig_round, sig_figs

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LADDER = os.path.join(ROOT, "results", "ladder")
SCREEN = os.path.join(ROOT, "data", "screening")


def load(dskey, tag, level):
    p = os.path.join(LADDER, f"{dskey}__{tag}__{level}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for c in d["calls"]:
        out.setdefault(c["mol_id"], []).append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ld50")
    ap.add_argument("--tag", default="gem3flash")
    ap.add_argument("--lo", default="none")
    ap.add_argument("--hi", default="high")
    ap.add_argument("--show", default="switchers",
                    choices=["switchers", "non-switchers", "both"])
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(SCREEN, f"{args.dataset}.csv")).dropna(subset=["value"])
    lut = {str(k).strip(): float(v) for k, v in zip(df["mol_id"], df["value"])}
    smi = {str(k).strip(): str(s) for k, s in zip(df["mol_id"], df["smiles"])}

    lo, hi = load(args.dataset, args.tag, args.lo), load(args.dataset, args.tag, args.hi)
    if lo is None or hi is None:
        sys.exit(f"missing a level for {args.dataset}/{args.tag}")

    def hits(calls, t):
        return [c for c in calls if c["value"] is not None
                and sig_figs(t) >= 3 and sig_figs(c["value"]) >= 3
                and sig_round(c["value"], 3) == sig_round(t, 3)]

    groups = {"switchers": [], "non-switchers": [], "always": []}
    for mol in sorted(set(lo) & set(hi) & set(lut)):
        t = lut[mol]
        h_lo, h_hi = hits(lo[mol], t), hits(hi[mol], t)
        key = ("always" if h_lo and h_hi else
               "switchers" if h_hi and not h_lo else
               "non-switchers" if not h_hi and not h_lo else "always")
        groups[key].append(mol)

    print(f"{args.dataset} / {args.tag}: {args.lo} -> {args.hi}")
    print(f"  {len(groups['switchers']):4d} molecules hit ONLY with the budget (switchers)")
    print(f"  {len(groups['always']):4d} hit at both levels")
    print(f"  {len(groups['non-switchers']):4d} hit at neither\n")

    want = (["switchers"] if args.show == "switchers" else
            ["non-switchers"] if args.show == "non-switchers" else
            ["switchers", "non-switchers"])
    for g in want:
        print("=" * 100)
        print(f"{g.upper()}  (published value | replies at {args.lo} | replies at {args.hi})")
        print("=" * 100)
        for mol in groups[g][:args.n]:
            t = lut[mol]
            a = ", ".join(f"{c['value']}" for c in lo[mol])
            b = ", ".join(f"{c['value']}" for c in hi[mol])
            print(f"\n{mol[:60]}")
            print(f"   SMILES     {smi.get(mol, '')[:78]}")
            print(f"   published  {t}")
            print(f"   {args.lo:9s}  {a}")
            print(f"   {args.hi:9s}  {b}")
            texts = {c["text"] for c in hi[mol] if c.get("text")}
            if any(len(x) > 12 for x in texts):
                print(f"   raw({args.hi}) {list(texts)[0][:78]}")


if __name__ == "__main__":
    main()
