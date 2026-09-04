"""
LaTeX tables for the appendix, written straight from results/*.csv.

Nothing in the appendix is typed by hand, so a table cannot drift from the pipeline that
produced it. Each file is a bare tabular, \\input by paper/appendix.tex.

    python src/export_appendix_tables.py
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, REG = os.path.join(ROOT, "results"), os.path.join(ROOT, "src", "registry")
OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)

DS = {"esol": "ESOL", "freesolv": "FreeSolv", "lipophilicity": "Lipophilicity", "bace": "BACE",
      "aqsoldb": "AqSolDB", "caco2": "Caco-2", "ld50": "LD50", "ppbr": "PPBR", "qm7": "QM7",
      "qm8": "QM8", "qm9": "QM9", "antiviral": "Antiviral", "boilingpoint": "Boiling pt."}
# One panel everywhere, so a benchmark that is dropped has no column either -- not even an
# empty one. ORDER is filtered through DROP_DATASETS at use.
ORDER = [d for d in ["freesolv", "esol", "ld50", "lipophilicity", "aqsoldb", "caco2",
                     "bace", "ppbr", "qm7", "qm8", "qm9", "antiviral", "boilingpoint"]
         if d not in ("qm9",)]


def esc(s):
    return str(s).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def write(name, body):
    p = os.path.join(OUT, name + ".tex")
    open(p, "w", encoding="utf-8").write(body.rstrip() + "\n")
    print(f"  {name}.tex")


def tabular(colspec, header, rows, midrules=()):
    out = ["\\begin{tabular}{" + colspec + "}", "\\toprule", header, "\\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            out.append("\\midrule")
        out.append(r)
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def longtabular(colspec, header, rows, midrules=()):
    """A table that outgrows a page. A tabular inside a float is CLIPPED, not broken, and the
    lost rows are silent -- longtable breaks and repeats the header instead. The caption is set
    by the appendix with \\captionof, because a longtable is not a float."""
    out = ["\\begin{longtable}{" + colspec + "}",
           "\\toprule", header, "\\midrule", "\\endfirsthead",
           "\\toprule", header, "\\midrule", "\\endhead",
           "\\bottomrule", "\\endfoot"]
    for i, r in enumerate(rows):
        if i in midrules:
            out.append("\\midrule")
        out.append(r)
    out += ["\\bottomrule", "\\end{longtable}"]
    return "\n".join(out)


# Kept out of every table for the same reason as the figures: one panel everywhere. Both are
# described in the appendix section they are excluded into.
DROP_MODELS = ("kimik26", "glm5", "glm52", "nemotron3u", "qwen35", "dsv4pro", "opus48", "opus45")
DROP_DATASETS = ("qm9",)


def main():
    # The headline map is the controlled-budget arm at its 1024-token setting, classified by
    # src/classify.py (_v3). The zero-shot screen is an appendix arm and is exported separately.
    S = pd.read_csv(os.path.join(RES, "budget_3sig_v3.csv"))
    if "arm" in S.columns:
        S = S[S.arm == "t1024"]
    S = S[~S.tag.isin(DROP_MODELS) & ~S.dataset.isin(DROP_DATASETS)].copy()
    models = json.load(open(os.path.join(REG, "models.json")))["models"]
    dsets = {d["key"]: d for d in json.load(open(os.path.join(REG, "datasets.json")))["datasets"]}
    tags = [m["tag"] for m in models if m["tag"] in set(S.tag)]
    S["ord"] = S.tag.map({t: i for i, t in enumerate(tags)})

    # ---------------------------------------------------------------- A: flagged cells
    F = S[(S.verdict == "contaminated") & (S.dataset != "boilingpoint")].copy()
    F = F.sort_values(["dataset", "hit3"], ascending=[True, False])
    rows = []
    for _, r in F.iterrows():
        q = min(r.q_hit_joint, r.q_deep_joint)
        rows.append(
            f"{DS[r.dataset]} & {esc(r.model)} & {r.medae:.3f} & "
            f"{r.hit3:.2f} [{r.hit_lo:.2f}, {r.hit_hi:.2f}] & "
            f"{r.R23:.1f} [{r.deep_lo:.1f}, {r.deep_hi:.1f}] & {r.floor_R23:.1f} & "
            f"{int(r.m2)} & {int(r.n_usable)} & "
            f"{'$<$0.003' if q < 0.003 else f'{q:.3f}'} \\\\")
    # "floor" is the label-only floor of R23 (floor_R23), which is what the appendix caption
    # describes. The smooth-error null's floors are a different quantity and are reported in
    # the nulls section, not here -- two floors in one table read as one.
    write("flagged_cells", longtabular(
        "llrlrlrrl",
        "benchmark & model & medAE & $\\textsc{hit}_3$ [95\\% CI] & "
        "$R_{23}$ [95\\% CI] & floor & $m_2$ & $n$ & $q$ \\\\",
        rows))

    # ---------------------------------------------------------------- B: the full map
    piv = S.pivot_table(index="tag", columns="dataset", values="verdict", aggfunc="first")
    piv = piv.reindex(index=tags, columns=ORDER)
    code = {"contaminated": "\\textbf{C}",
            "clean": "\\textperiodcentered"}
    name = S.drop_duplicates("tag").set_index("tag").model.to_dict()
    hit = S.pivot_table(index="tag", columns="dataset", values="hit3", aggfunc="first")
    rows = []
    for t in tags:
        cells = [code.get(piv.loc[t, d], "--") if isinstance(piv.loc[t, d], str) else "--"
                 for d in ORDER]
        rows.append(esc(name[t]) + " & " + " & ".join(cells) + " \\\\")
    write("full_map", tabular(
        "l" + "c" * len(ORDER),
        "model & " + " & ".join("\\rotatebox{90}{" + DS[d] + "}" for d in ORDER) + " \\\\",
        rows))

    # hit3 version of the same grid
    rows = []
    for t in tags:
        cs = []
        for d in ORDER:
            v = hit.loc[t, d] if d in hit.columns else np.nan
            reg = piv.loc[t, d]
            if not isinstance(reg, str):
                cs.append("--")
            elif np.isnan(v):
                cs.append("--")
            else:
                s = f"{v:.2f}" if v < 10 else f"{v:.1f}"
                cs.append("\\textbf{" + s + "}" if reg == "contaminated" else s)
        rows.append(esc(name[t]) + " & " + " & ".join(cs) + " \\\\")
    write("full_map_hit3", tabular(
        "l" + "r" * len(ORDER),
        "model & " + " & ".join("\\rotatebox{90}{" + DS[d] + "}" for d in ORDER) + " \\\\",
        rows))

    # ---------------------------------------------------------------- C: model panel
    rows = []
    for m in models:
        if m["tag"] not in set(S.tag):
            continue
        rows.append(f"{esc(m['name'])} & {esc(m['vendor'])} & {m['release']} & "
                    f"\\texttt{{{esc(m['reasoning'])}}} \\\\")
    write("model_panel", tabular("llll",
                                 "model & vendor & release & reasoning setting \\\\", rows))

    g = S.groupby("dataset").agg(n_mol=("n_mol", "max"), cells=("tag", "count"),
                                 nus=("n_usable", "median"))
    reg = S.pivot_table(index="dataset", columns="verdict", values="tag",
                        aggfunc="count").fillna(0).astype(int)
    rows = []
    for d in [x for x in ORDER if x in g.index]:
        cls = {"measured": "measured", "computed": "computed",
               "pos_control": "positive control",
               "neg_control": "recency control"}[dsets[d]["class"]]
        rows.append(f"{DS[d]} & {cls} & {esc(dsets[d]['prompt_unit'])} & {dsets[d]['n']} & "
                    f"{int(g.loc[d, 'n_mol'])} & {int(g.loc[d, 'nus'])} & "
                    f"{reg.loc[d].get('contaminated', 0)} & "
                    f"{reg.loc[d].get('clean', 0)} & {reg.loc[d].get('untestable', 0)} \\\\")
    write("benchmark_panel", tabular(
        "llllrrrrr",
        "benchmark & role & unit & size & mol.\\ scorable & median $n$ & flagged & clean & untest. \\\\",
        rows))

    # ---------------------------------------------------------------- D: ladder
    L = pd.read_csv(os.path.join(RES, "ladder_summary.csv"))
    L = L.sort_values(["dataset", "tag", "reasoning_med"])
    rows = []
    prev = None
    mid = []
    for i, (_, r) in enumerate(L.iterrows()):
        if prev is not None and (r.dataset, r.tag) != prev:
            mid.append(i)
        prev = (r.dataset, r.tag)
        rows.append(f"{DS[r.dataset]} & {esc(r.model)} & {esc(r.level)} & "
                    f"{r.reasoning_med:.0f} & {int(r.n_usable)} & {int(r.m2)} & "
                    f"{r.hit3:.2f} & "
                    f"{'' if np.isnan(r.R23) else f'{r.R23:.1f}'} & {r.medae:.3f} & "
                    f"{r.spearman:.2f} & {r.selfcons:.1f} & {r.cost:.3f} \\\\")
    head = ("benchmark & model & effort & tokens & $n$ & $m_2$ & $\\textsc{hit}_3$ & "
            "$R_{23}$ & medAE & $\\rho$ & self-c. & \\$ \\\\")
    body = ["\\begin{longtable}{lllrrrrrrrrr}", "\\toprule", head, "\\midrule", "\\endfirsthead",
            "\\toprule", head, "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    for i, r in enumerate(rows):
        if i in set(mid):
            body.append("\\midrule")
        body.append(r)
    body.append("\\end{longtable}")
    write("ladder", "\n".join(body))

    # ---------------------------------------------------------------- E: L1/L5
    B = pd.read_csv(os.path.join(RES, "blinding_sweep.csv")).sort_values(
        ["dataset", "zs_hit3"], ascending=[True, False])
    # The first cell run (ESOL/Opus 4.8) predates the --reasoning flag, so its meta has no
    # setting recorded; it ran at whatever the registry says, which is what we print.
    reg_reasoning = {m["tag"]: m["reasoning"] for m in models}
    B["reasoning"] = [x if isinstance(x, str) else reg_reasoning.get(t, "?")
                      for x, t in zip(B.reasoning, B.tag)]
    rows = []
    for _, r in B.iterrows():
        star = "" if r.d_lo <= 0 <= r.d_hi else "$^{\\ast}$"
        rows.append(f"{DS[r.dataset]} & {esc(r.model)} & \\texttt{{{esc(r.reasoning)}}} & "
                    f"{int(r.n)} & {r.zs_hit3:.1f} & {r.medae_L1:.3f} & {r.medae_L5:.3f} & "
                    f"{r.ratio:.1f} & {int(r.usable_L1)} & {r.hit3_L1:.1f} & {r.floor_L1:.2f} & "
                    f"{int(r.usable_L5)} & {r.hit3_L5:.1f} & {r.floor_L5:.2f} & "
                    f"${r.d_medae:+.3f}$ [{r.d_lo:+.3f}, {r.d_hi:+.3f}]{star} \\\\")
    write("l1l5", tabular(
        "lllrrrrrrrrrrrl",
        "benchmark & model & reasoning & $n$ & map & \\multicolumn{2}{c}{medAE} & $\\times$ & "
        "\\multicolumn{3}{c}{L1} & \\multicolumn{3}{c}{L5} & degradation [95\\% CI] \\\\"
        "\n & & & & $\\textsc{hit}_3$ & L1 & L5 & & $n$ & $\\textsc{hit}_3$ & floor & "
        "$n$ & $\\textsc{hit}_3$ & floor & \\\\",
        rows))

    # ---------------------------------------------------------------- F: randomised SMILES
    R = pd.read_csv(os.path.join(RES, "randomization_control.csv"))
    rows = [f"{esc(r.model)} & {r.hit3_can:.2f} & {r.hit3_rand:.2f} & "
            f"{r.R23_can:.1f} & {r.R23_rand:.1f} & {r.rmse_can:.3f} & {r.rmse_rand:.3f} \\\\"
            for _, r in R.iterrows()]
    write("randomised", tabular(
        "lrrrrrr",
        "model & $\\textsc{hit}_3$ pub. & $\\textsc{hit}_3$ rand. & "
        "$R_{23}$ pub. & $R_{23}$ rand. & RMSE pub. & RMSE rand. \\\\", rows))

    # ---------------------------------------------------------------- G: corpus prevalence
    P = pd.read_csv(os.path.join(RES, "corpus_prevalence.csv"))
    P["idx"] = P["index"].str.extract(r"_(dolma-v1_7|dclm-baseline|olmo-mix-1124)_")[0]
    piv2 = P.pivot_table(index="dataset", columns="idx",
                         values=["header_count", "smiles_rate"], aggfunc="max")
    mx = S.groupby("dataset").hit3.max()
    nt = S[S.regime != "untestable"].groupby("dataset").size()
    rows = []
    for d in ORDER:
        if d not in piv2.index:
            continue
        h = [piv2.loc[d, ("header_count", k)] for k in ("dclm-baseline", "dolma-v1_7",
                                                        "olmo-mix-1124")]
        s = [piv2.loc[d, ("smiles_rate", k)] for k in ("dclm-baseline", "dolma-v1_7",
                                                       "olmo-mix-1124")]
        f = lambda v: "--" if pd.isna(v) else f"{v:.0f}"
        g_ = lambda v: "--" if pd.isna(v) else f"{v:.1f}"
        # A max hit rate over cells the detector cannot test is not a measurement; blank it.
        k = int(nt.get(d, 0))
        rows.append(f"{DS[d]} & " + " & ".join(f(x) for x in h) + " & " +
                    " & ".join(g_(x) for x in s) +
                    f" & {(f'{mx[d]:.2f}' if k else '--')} & {k} \\\\")
    write("prevalence", tabular(
        "lrrrrrrrr",
        "benchmark & \\multicolumn{3}{c}{header hits} & "
        "\\multicolumn{3}{c}{\\% molecules present} & max & testable \\\\"
        "\n & DCLM & Dolma & OLMo & DCLM & Dolma & OLMo & $\\textsc{hit}_3$ & cells \\\\",
        rows))

    # ---------------------------------------------------------------- H: threshold sweep
    # Removed 25 Aug 2026. This copied results/threshold_sensitivity_3sig.csv, the output of
    # src/threshold_sensitivity.py, which swept the RELEASED scheme's effect-size gates
    # (excess >= 4 for heavy, >= 1.5 for partial). Those constants no longer exist: the current
    # scheme flags on rung significance, and src/sweep_thresholds.py sweeps what it does have.
    # The copied file was never \input by the paper -- tab:thresholds reads tables/thresholds.tex.

    # ---------------------------------------------------------------- I: dissociation
    rows = []
    for d in ("esol", "freesolv", "ld50", "aqsoldb", "lipophilicity", "qm8"):
        x = S[S.dataset == d]
        for lab, sel in (("clean", x.verdict == "clean"),
                         ("flagged", x.verdict == "contaminated")):
            y = x[sel]
            if not len(y):
                continue
            rows.append(f"{DS[d]} & {lab} & {len(y)} & {y.spearman.min():.3f} & "
                        f"{y.spearman.max():.3f} & {y.medae.min():.3f} & {y.medae.max():.3f} \\\\")
    write("dissociation", tabular(
        "llrrrrr",
        "benchmark & group & $n$ & $\\rho_{\\min}$ & $\\rho_{\\max}$ & "
        "medAE$_{\\min}$ & medAE$_{\\max}$ \\\\", rows))

    # ------------------------------------------------- J: score against recall, per model
    # The main text promises all four measures per model; before this they lived only in the
    # CSV, so the promise was unkeepable from the paper alone.
    try:
        G = pd.read_csv(os.path.join(RES, "generalization.csv"))
    except FileNotFoundError:
        G = None
    if G is not None and len(G):
        G = G[~G.tag.isin(DROP_MODELS) & ~G.dataset.isin(DROP_DATASETS)]
        DOT, FLAG = "\\textperiodcentered", "\\textbf{C}"
        rows, mids = [], []
        for d in [x for x in ORDER if x in set(G.dataset)]:
            g = G[G.dataset == d].sort_values("mae_clip")
            if len(rows):
                mids.append(len(rows))
            for _, r in g.iterrows():
                rows.append(
                    # generalization.csv predates the verdict column and carries the old regime
                    # labels; anything that is not one of the three negatives was flagged.
                    f"{DS[d]} & {esc(r.model)} & "
                    f"{DOT if r.regime in ('clean', 'no-signal', 'untestable') else FLAG} & "
                    f"{r.hit3:.2f} & {r.r_clip:.3f} & {r.rho:.3f} & {r.mae_clip:.3f} & "
                    f"{r.rmse_clip:.3f} & {r.mae_clip_res:.3f} & {int(r.n_wild)} \\\\")
        # 88 rows: a tabular in a float would be silently clipped at the page bottom, so
        # this one is a longtable with a repeating header.
        head = ("benchmark & model & flag & $\\textsc{hit}_3$ & $r$ & $\\rho$ & MAE & RMSE & "
                "MAE resid. & outside \\\\")
        body = ["\\begin{longtable}{llcrrrrrrr}", "\\toprule", head, "\\midrule",
                "\\endfirsthead", "\\toprule", head, "\\midrule", "\\endhead",
                "\\bottomrule", "\\endfoot"]
        for i, r in enumerate(rows):
            if i in mids:
                body.append("\\midrule")
            body.append(r)
        body.append("\\end{longtable}")
        write("generalization_per_model", "\n".join(body))

    print("wrote tables to", OUT)


if __name__ == "__main__":
    main()
