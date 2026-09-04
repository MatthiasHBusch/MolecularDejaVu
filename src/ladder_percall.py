"""Within one cell, do the calls that thought get the digits?

Every dose-response in this paper is between cells: a cell asked for `none` against the same
cell asked for `high`. That comparison cannot separate "reasoning retrieves the value" from
"the effort label changes something else about the request", because the label moves with the
dose.

Opus 5 makes the cleaner test possible. At `minimal` it emits nothing on most calls and a
hundred-odd tokens on the rest, so a single cell contains both branches under one label, one
prompt and one molecule set. If the tokens are what retrieves the value, the calls that emitted
them hit more often -- and the molecules are the same molecules, since every one is queried
three times.

    python src/ladder_percall.py --tag opus5 --levels minimal,low

Writes results/meta/ladder_percall.csv.
"""
import argparse, glob, json, os, re, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import sig_key_vec  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAD = os.path.join(ROOT, "results", "ladder")
SCREEN = os.path.join(ROOT, "data", "screening")


def matches3(truth, pred):
    """Agreement at 3 significant figures, scorable pairs only."""
    T, P = np.asarray(truth, float), np.asarray(pred, float)
    ok = np.isfinite(T) & np.isfinite(P)
    mt, et = sig_key_vec(T, 3)
    mp, ep = sig_key_vec(P, 3)
    return ok, (mt == mp) & (et == ep) & ok


def scorable(ds):
    """mol_id -> truth, for truths that carry three significant figures."""
    df = pd.read_csv(os.path.join(SCREEN, f"{ds}.csv")).dropna(subset=["value"])
    t = df.value.to_numpy(float)
    keep = np.array([len(re.sub(r"[^0-9]", "", f"{abs(v):.10g}".split("e")[0]).lstrip("0")) >= 3
                     for v in t])
    return dict(zip(df.mol_id.astype(str).str.strip()[keep], t[keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="opus5")
    ap.add_argument("--levels", default="minimal,low,medium")
    ap.add_argument("--min-per-branch", type=int, default=15)
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(LAD, f"*__{a.tag}__*.json"))):
        m = json.load(open(p, encoding="utf8"))
        meta = m["meta"]
        if meta["level"] not in a.levels.split(","):
            continue
        truth = scorable(meta["dataset"])
        rec = [(c["mol_id"], c["value"], c["reasoning_tokens"]) for c in m["calls"]
               if c.get("value") is not None and str(c["mol_id"]).strip() in truth]
        if not rec:
            continue
        mol = np.array([r[0] for r in rec])
        pred = np.array([r[1] for r in rec], float)
        tok = np.array([r[2] for r in rec], int)
        tru = np.array([truth[str(x).strip()] for x in mol], float)
        _, hit = matches3(tru, pred)
        z, nz = tok == 0, tok > 0
        if z.sum() < a.min_per_branch or nz.sum() < a.min_per_branch:
            continue
        # The molecules common to both branches, so the comparison is not one branch getting
        # the easy molecules: within a cell every molecule is queried three times, and the
        # endpoint decides per call whether to think.
        common = set(mol[z]) & set(mol[nz])
        cz = np.array([x in common for x in mol]) & z
        cnz = np.array([x in common for x in mol]) & nz
        rows.append(dict(dataset=meta["dataset"], level=meta["level"],
                         n_zero=int(z.sum()), n_think=int(nz.sum()),
                         hit_zero=100 * hit[z].mean(), hit_think=100 * hit[nz].mean(),
                         med_tok_think=float(np.median(tok[nz])),
                         n_common_mol=len(common),
                         hit_zero_c=100 * hit[cz].mean() if cz.sum() else np.nan,
                         hit_think_c=100 * hit[cnz].mean() if cnz.sum() else np.nan))

    d = pd.DataFrame(rows)
    if d.empty:
        print("no cell has both branches at the requested size")
        return
    d["lift"] = d.hit_think / d.hit_zero.replace(0, np.nan)
    os.makedirs(os.path.join(ROOT, "results", "meta"), exist_ok=True)
    d.to_csv(os.path.join(ROOT, "results", "meta", "ladder_percall.csv"), index=False)
    pd.set_option("display.width", 200)
    print(d.to_string(index=False))
    print("\nsame comparison restricted to molecules that appear in BOTH branches:")
    print(d[["dataset", "level", "n_common_mol", "hit_zero_c", "hit_think_c"]].to_string(index=False))
    print("\nwrote results/meta/ladder_percall.csv")


if __name__ == "__main__":
    main()
