"""
Cost projection for the screening sweeps.

Two modes.

  * Before the pilot has run, costs are projected from the OpenRouter list prices in
    registry/models.json plus a token model calibrated on the completed 13-model x
    3-dataset study (results/memo_matrix.log: $117.25 -> $223.48 for 39 cells x 3,000
    calls, i.e. $0.91 per 1,000 calls averaged over models).

  * Once results/usage_log.csv exists, the per-model cost per 1,000 calls is taken from
    what was actually charged and the list-price model is used only for models the pilot
    has not yet touched. Measured beats estimated: reasoning-token overhead varies by more
    than an order of magnitude between endpoints and is not predictable from price alone.

    python src/estimate_cost.py                        # plan for the three stages
    python src/estimate_cost.py --pilot-n 60 --deep-n 1000 --deep-iters 3
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG = os.path.join(ROOT, "src", "registry")
USAGE = os.path.join(ROOT, "results", "usage_log.csv")

# Token model for one zero-shot call. The prompt is a system message plus a one-line
# question containing a SMILES string; the reply is a bare number. Reasoning endpoints
# additionally burn their thinking budget, billed as output.
TOK_IN = 130
TOK_OUT_PLAIN = 12
TOK_OUT_REASONING = 150


def list_price_per_1k(m):
    out = TOK_OUT_REASONING if m["reasoning"] != "none" else TOK_OUT_PLAIN
    per_call = (TOK_IN * m["price_in"] + out * m["price_out"]) / 1e6
    return 1000 * per_call


def measured_per_1k():
    if not os.path.exists(USAGE):
        return {}
    u = pd.read_csv(USAGE)
    u = u[u.n_calls > 0]
    if u.empty:
        return {}
    g = u.groupby("model").apply(
        lambda d: 1000 * d.cost_usd.sum() / d.n_calls.sum(), include_groups=False)
    return g.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-n", type=int, default=60)
    ap.add_argument("--pilot-iters", type=int, default=1)
    ap.add_argument("--deep-n", type=int, default=1000)
    ap.add_argument("--deep-iters", type=int, default=3)
    ap.add_argument("--deep-model-tags", default="opus48,gem35flash,gem31pro",
                    help="models that get the all-datasets sweep. Default = the heaviest "
                         "retrievers found so far; replace with the pilot's ranking.")
    ap.add_argument("--deep-datasets", type=int, default=4,
                    help="how many flagged datasets get the all-models sweep")
    ap.add_argument("--budget", type=float, default=600.0)
    args = ap.parse_args()

    models = json.load(open(os.path.join(REG, "models.json")))["models"]
    dsets = json.load(open(os.path.join(REG, "datasets.json")))["datasets"]
    meas = measured_per_1k()

    rows = []
    for m in models:
        src = "measured" if m["tag"] in meas else "list price"
        p1k = meas.get(m["tag"], list_price_per_1k(m))
        rows.append(dict(tag=m["tag"], model=m["name"], vendor=m["vendor"],
                         reasoning=m["reasoning"], usd_per_1k=p1k, basis=src))
    M = pd.DataFrame(rows).sort_values("usd_per_1k", ascending=False)

    print(f"Cost per 1,000 zero-shot calls ({len(M)} models, "
          f"{(M.basis == 'measured').sum()} measured / {(M.basis == 'list price').sum()} estimated)")
    print(M[["model", "vendor", "reasoning", "usd_per_1k", "basis"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))

    nD, nM = len(dsets), len(M)
    mean1k = M.usd_per_1k.mean()

    # Stage 1: the whole matrix, shallow.
    c1 = (M.usd_per_1k.sum() * nD) * (args.pilot_n * args.pilot_iters / 1000)
    # Stage 2: the heaviest RETRIEVERS (not the priciest models) against every dataset.
    deep_tags = [t.strip() for t in args.deep_model_tags.split(",") if t.strip()]
    sel = M[M.tag.isin(deep_tags)]
    missing = set(deep_tags) - set(sel.tag)
    if missing:
        print(f"\nWARNING: --deep-model-tags not in registry: {sorted(missing)}")
    top = sel.usd_per_1k.sum()
    args.deep_models = len(sel)
    c2 = top * nD * (args.deep_n * args.deep_iters / 1000)
    # Stage 3: every model against the flagged datasets, at depth.
    c3 = M.usd_per_1k.sum() * args.deep_datasets * (args.deep_n * args.deep_iters / 1000)
    # Chemistry-invariance control: randomised SMILES on the flagged cells only.
    c4 = M.usd_per_1k.sum() * 1 * (args.deep_n * args.deep_iters / 1000)

    print(f"\n{'stage':38s}{'cells':>7}{'calls':>10}{'USD':>10}")
    plan = [
        (f"1 pilot: {nM} models x {nD} datasets @ {args.pilot_n}x{args.pilot_iters}",
         nM * nD, nM * nD * args.pilot_n * args.pilot_iters, c1),
        (f"2 deep: {args.deep_models} heaviest x {nD} datasets @ {args.deep_n}x{args.deep_iters}",
         args.deep_models * nD, args.deep_models * nD * args.deep_n * args.deep_iters, c2),
        (f"3 deep: {nM} models x {args.deep_datasets} flagged @ {args.deep_n}x{args.deep_iters}",
         nM * args.deep_datasets, nM * args.deep_datasets * args.deep_n * args.deep_iters, c3),
        (f"4 randomised-SMILES control on 1 flagged dataset",
         nM, nM * args.deep_n * args.deep_iters, c4),
    ]
    for name, cells, calls, usd in plan:
        print(f"{name:38s}{cells:>7}{calls:>10}{usd:>10.0f}")
    total = sum(p[3] for p in plan)
    print(f"{'TOTAL':38s}{'':>7}{'':>10}{total:>10.0f}")
    print(f"\nBudget {args.budget:.0f} USD -> {'OK' if total <= args.budget else 'OVER by %.0f' % (total - args.budget)}")

    if total > args.budget:
        # Iterations buy less than molecules do: repeats of the same molecule are correlated,
        # independent molecules are not. Cutting iterations is the first lever.
        k = args.budget / total
        print(f"\nTo fit: drop deep iterations 3 -> 2 (saves ~{(1 - 2/3) * (c2 + c3 + c4):.0f} USD), "
              f"or scale deep-n by {k:.2f} to {int(args.deep_n * k)} molecules.")
        print("Prefer cutting ITERATIONS over MOLECULES: repeats of one molecule are "
              "correlated draws, extra molecules are independent ones, so per dollar the "
              "molecule count buys more statistical power.")

    if not meas:
        print("\nNOTE: no results/usage_log.csv yet -- every figure above is a list-price "
              "estimate. Run stage 1 and re-run this script for measured numbers.")


if __name__ == "__main__":
    main()
