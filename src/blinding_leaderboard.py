"""Does blinding reorder the leaderboard?

Section 3.3 asks what a benchmark score is worth once retrieval is taken out of it, and answers
it twice with two different removals. The first removes the retrieved ROWS and rescores the rest.
This one removes the retrieval itself, by substituting the structure string, and rescores the
same models on the same molecules -- which is the stronger form of the question, because it also
takes away whatever the model would have retrieved for a row it did not reproduce verbatim.

Reads the raw L1/L5 runs of the controlled arm (results/blinding/*__t1024.json) and writes
paper/tables/blinding_leaderboard.tex.

    python src/blinding_leaderboard.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "tables")

DS_LABEL = {"esol": "ESOL", "freesolv": "FreeSolv", "ld50": "LD50"}
DS_ORDER = ["freesolv", "esol", "ld50"]


def cell(path):
    d = json.load(open(path, encoding="utf8"))
    m = d["meta"]
    y = np.array([c.get("truth") for c in d["calls"]], float)
    p = np.array([c.get("value") if c.get("value") is not None else np.nan
                  for c in d["calls"]], float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return dict(dataset=m["dataset"], tag=m["tag"], level=m["level"], n=int(ok.sum()),
                r=float(pearsonr(y, p)[0]) if ok.sum() > 2 else np.nan,
                medae=float(np.median(np.abs(y - p))))


def main():
    rows = [cell(p) for p in sorted(glob.glob(os.path.join(RES, "blinding", "*__t1024.json")))]
    d = pd.DataFrame(rows)
    names = (pd.read_csv(os.path.join(RES, "blinding_sweep_t1024.csv"))
             .drop_duplicates("tag").set_index("tag").model.to_dict())
    d["model"] = d.tag.map(names)

    w = d.pivot_table(index=["dataset", "tag", "model"], columns="level",
                      values=["r", "medae"]).reset_index()
    w.columns = ["dataset", "tag", "model", "medae_L1", "medae_L5", "r_L1", "r_L5"]

    # The ranks are what the section is about: a leaderboard is an ordering, and the question is
    # whether the ordering survives the intervention that removes retrieval.
    for c in ("medae_L1", "medae_L5"):
        w[c.replace("medae", "rank_mae")] = w.groupby("dataset")[c].rank().astype(int)
    for c in ("r_L1", "r_L5"):
        w[c.replace("r_", "rank_r_")] = w.groupby("dataset")[c].rank(ascending=False).astype(int)

    w["ds_ord"] = w.dataset.map({k: i for i, k in enumerate(DS_ORDER)})
    w = w.sort_values(["ds_ord", "medae_L1"])

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    w.to_csv(os.path.join(ROOT, "results", "blinding_leaderboard.csv"), index=False)

    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             "benchmark & model & \\multicolumn{3}{c}{median abs.\\ error} & "
             "\\multicolumn{3}{c}{Pearson $r$} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             " & & L1 & L5 & rank & L1 & L5 & rank \\\\", "\\midrule"]
    for ds, g in w.groupby("ds_ord", sort=True):
        if lines[-1] != "\\midrule":
            lines.append("\\midrule")
        for _, x in g.iterrows():
            arrow = (f"{x.rank_mae_L1}\\,$\\to$\\,{x.rank_mae_L5}")
            arrow_r = (f"{x.rank_r_L1}\\,$\\to$\\,{x.rank_r_L5}")
            lines.append(
                f"{DS_LABEL[x.dataset]} & {x.model} & {x.medae_L1:.3f} & {x.medae_L5:.3f} & "
                f"{arrow} & {x.r_L1:.3f} & {x.r_L5:.3f} & {arrow_r} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "blinding_leaderboard.tex"), "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")

    print(w[["dataset", "model", "medae_L1", "medae_L5", "rank_mae_L1", "rank_mae_L5",
             "r_L1", "r_L5", "rank_r_L1", "rank_r_L5"]].to_string(index=False))
    same_mae = (w.rank_mae_L1 == w.rank_mae_L5).sum()
    same_r = (w.rank_r_L1 == w.rank_r_L5).sum()
    print(f"\nrank unchanged: {same_mae}/12 by medAE, {same_r}/12 by Pearson")
    for ds, g in w.groupby("dataset"):
        print(f"  {ds:9s} winner L1 {g.loc[g.rank_mae_L1 == 1, 'model'].iloc[0]:14s}"
              f" -> L5 {g.loc[g.rank_mae_L5 == 1, 'model'].iloc[0]}")
    print("\nwrote paper/tables/blinding_leaderboard.tex")


if __name__ == "__main__":
    main()
