"""Does an endpoint honour an effort label, and is the dose monotone?

The ladder's x-axis is emitted reasoning tokens, so a model earns a place in it only if its
effort labels resolve to an ORDERED span of tokens. Accepting the parameter is not enough: an
endpoint that returns 200 on every level but emits 1,146 tokens at `minimal` and 116 at `medium`
has a knob that turns but is not connected to anything, and plotting it would put the same model
at two unrelated doses with no way to tell which is which.

Several molecules per level, because one call cannot separate a non-monotone endpoint from a
sampling accident.

    python src/probe_effort.py --models kimik3,opus5,sol --n 6

Writes results/meta/effort_probe.csv.
"""
import argparse, json, os, sys, urllib.error

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_reasoning_ladder import LEVELS, call, key, prompts  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG = os.path.join(ROOT, "src", "registry")
SCREEN = os.path.join(ROOT, "data", "screening")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="kimik3,opus5,sol")
    ap.add_argument("--dataset", default="ld50")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=8000)
    a = ap.parse_args()

    models = {m["tag"]: m for m in json.load(open(os.path.join(REG, "models.json")))["models"]}
    dsets = {d["key"]: d for d in json.load(open(os.path.join(REG, "datasets.json")))["datasets"]}
    ds = dsets[a.dataset]
    df = pd.read_csv(os.path.join(SCREEN, f"{a.dataset}.csv")).dropna(
        subset=["value", "smiles"]).head(a.n)
    k = key()

    rows, spent = [], 0.0
    for t in a.models.split(","):
        for lv in a.levels.split(","):
            for _, row in df.iterrows():
                rec = dict(tag=t, level=lv, mol_id=str(row["mol_id"]).strip())
                try:
                    r = call(k, models[t], lv, *prompts(ds, str(row["smiles"]).strip()),
                             a.max_tokens)
                    spent += r["cost"]
                    rec.update(reasoning=r["reasoning_tokens"], completion=r["completion"],
                               cost=r["cost"], finish=r["finish"],
                               answered=r["value"] is not None)
                except urllib.error.HTTPError as e:
                    body = ""
                    try:
                        body = e.read().decode()[:200]
                    except Exception:
                        pass
                    rec.update(reasoning=None, finish=f"http{e.code}", answered=False, cost=0.0,
                               error=body)
                except Exception as e:
                    # The whole point of this script is to find out WHY a level fails, so the
                    # exception type and message are recorded rather than swallowed.
                    rec.update(reasoning=None, finish=f"{type(e).__name__}: {e}"[:200],
                               answered=False, cost=0.0)
                rows.append(rec)
            d = pd.DataFrame([r for r in rows if r["tag"] == t and r["level"] == lv])
            got = d.reasoning.dropna()
            print(f"  {t:8s} {lv:8s} n={len(d)} answered={int(d.answered.sum())} "
                  f"reasoning median={got.median() if len(got) else float('nan'):8.0f} "
                  f"[{got.min() if len(got) else float('nan'):.0f}-"
                  f"{got.max() if len(got) else float('nan'):.0f}]  "
                  f"${d.cost.sum():.4f}  {d.finish.mode().iloc[0] if len(d) else ''}",
                  flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results", "meta"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "results", "meta", "effort_probe.csv"), index=False)
    print(f"\nspent ${spent:.3f}")

    print("\nmedian emitted reasoning tokens per level:")
    piv = out.pivot_table(index="tag", columns="level", values="reasoning", aggfunc="median")
    piv = piv.reindex(columns=[l for l in LEVELS if l in piv.columns])
    print(piv.to_string())
    print("\nmonotone in the requested order?")
    for t, r in piv.iterrows():
        v = r.dropna()
        ok = bool((v.diff().dropna() >= 0).all()) and len(v) >= 3
        print(f"  {t:8s} {'YES' if ok else 'NO ':3s}  {list(v.round(0).astype(int))}"
              f"  levels answered: {len(v)}/{len(piv.columns)}")


if __name__ == "__main__":
    main()
