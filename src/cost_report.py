"""What the study cost, from the per-cell records rather than from memory.

Three campaigns keep a per-cell cost, so they can be scoped to the panel this paper reports:

    the controlled map and the minimum-reasoning run   results/budget_3sig_v3.csv (cost, n_call)
    the zero-shot screen                                  results/usage_log.csv      (cost_usd, n_calls)
    the reasoning ladder                                  results/ladder_summary.csv (cost, n_calls)

The smaller campaigns (the L1/L5 blinding sweep, the structure/label factorial, the earlier
partial replicate, the trace capture) were logged as campaign totals only and cannot be split by
model, so they are reported as one figure and never mixed into a per-arm table.

    python src/cost_report.py
"""
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

DROP_MODELS = ("kimik26", "glm5", "glm52", "nemotron3u", "qwen35", "dsv4pro", "opus48", "opus45")
DROP_DATASETS = ("qm9",)

# Campaign totals with no per-cell record. Measured spend, taken from the run logs at the time.
UNSPLIT = {"L1/L5 blinding sweep": 50.23, "structure/label factorial": 24.21,
           "earlier partial replicate": 143.00, "trace capture": 0.23}


def panel(d, model_col, ds_col):
    return d[~d[model_col].isin(DROP_MODELS) & ~d[ds_col].isin(DROP_DATASETS)]


def main():
    B = pd.read_csv(os.path.join(RES, "budget_3sig_v3.csv"))
    U = pd.read_csv(os.path.join(RES, "usage_log.csv"))
    L = pd.read_csv(os.path.join(RES, "ladder_summary.csv"))

    rows = []
    for arm, label in (("t1024", "controlled map (1024 tokens)"),
                       ("reg", "minimum-reasoning run")):
        g = panel(B[B.arm == arm], "tag", "dataset")
        rows.append((label, len(g), int(g.n_call.sum()), float(g.cost.sum())))
    # The zero-shot screen was a third campaign row here. It is retired (see
    # results/_archive_zeroshot/), so its spend is no longer part of what this paper reports and
    # the row is gone. Its calls sit in the usage log, which is why that log is no longer summed
    # into the panel total.
    lp = panel(L, "tag", "dataset")
    rows.append(("reasoning ladder", lp.groupby(["dataset", "tag"]).ngroups,
                 int(lp.n_calls.sum()), float(lp.cost.sum())))

    print(f"{'campaign':34s}{'cells':>7}{'calls':>10}{'cost':>10}")
    for lab, n, c, k in rows:
        print(f"{lab:34s}{n:7d}{c:10,d}{k:10.2f}")
    tc, tk = sum(r[2] for r in rows), sum(r[3] for r in rows)
    print(f"{'  panel total':34s}{'':7}{tc:10,d}{tk:10.2f}")

    off = ((B[B.arm == "t1024"].cost.sum() - rows[0][3])
           + (U.cost_usd.sum() - rows[2][3]))
    print(f"\noff-panel models and benchmarks (measured, not reported): {off:.2f}")
    print("campaigns logged as totals only:")
    for k, v in UNSPLIT.items():
        print(f"   {k:32s}{v:8.2f}")
    print(f"\nwhole study: {tk + off + sum(UNSPLIT.values()):.2f}")

    def num(x, money=False):
        s = f"\\${x:,.0f}" if money else f"{x:,}"
        return s.replace(",", "{,}")

    out = ["\\begin{tabular}{lrrr}", "\\toprule",
           "campaign & cells & calls & cost \\\\", "\\midrule"]
    for lab, n, c, k in rows:
        out.append(f"{lab} & {n} & {num(c)} & {num(k, True)} \\\\")
    out += ["\\midrule",
            f"\\emph{{what this paper reports}} & & \\textbf{{{num(tc)}}} & "
            f"\\textbf{{{num(tk, True)}}} \\\\",
            "\\midrule",
            f"models and benchmarks measured, not reported & & & {num(off, True)} \\\\",
            f"further campaigns, logged as campaign totals & & & "
            f"{num(sum(UNSPLIT.values()), True)} \\\\",
            f"\\emph{{whole study}} & & & \\emph{{{num(tk + off + sum(UNSPLIT.values()), True)}}} \\\\",
            "\\bottomrule", "\\end{tabular}"]
    p = os.path.join(ROOT, "paper", "tables", "cost.tex")
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
