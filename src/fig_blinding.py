"""The blinding subsection's two figures.

fig5_mitigation  what L5 does to verbatim retrieval, as paired bars
fig_board_*      three candidate renderings of the leaderboard table, to choose between

    python src/fig_blinding.py            # writes all of them to paper/figures
    python src/fig_blinding.py --variants # only the three board candidates
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures_paper import DS_SHORT, INK, INK2, W, save, sig_ramp, use_serif  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")

DS_ORDER = ["freesolv", "esol", "ld50"]
# One hue per benchmark, one lightness step per model inside it, so a reader can read "which
# benchmark" at a glance and "which model" on a second look. Ordered dark to light by the
# model's retrieval, which is also the order the bars come in.
DS_HUE = {"freesolv": ["#0b3d66", "#1f6ea8", "#5b9dd1", "#a8c9e8"],
          "esol":     ["#7a3b06", "#c2650f", "#e39445", "#f2c391"],
          "ld50":     ["#12503a", "#1f8f66", "#5cbb96", "#a8dcc6"]}
MODEL_ORDER = ["opus5", "grok45", "sol", "kimik3"]


def load():
    b = pd.read_csv(os.path.join(RES, "blinding_sweep_t1024.csv"))
    b["m_ord"] = b.tag.map({t: i for i, t in enumerate(MODEL_ORDER)})
    b["d_ord"] = b.dataset.map({d: i for i, d in enumerate(DS_ORDER)})
    return b.sort_values(["d_ord", "m_ord"])


# =============================================================== fig 5, panel a as bars
def fig5_bars(b):
    """Verbatim retrieval before and after substituting the structure string.

    The paired-line version of this panel plotted twelve lines between two x positions, which
    reads as a single collapsing bundle and hides which benchmark and which model each line is.
    Bars carry the same numbers with the identity attached: hue = benchmark, lightness = model,
    hatch = blinded.
    """
    use_serif()
    fig = plt.figure(figsize=(W, 2.5))
    ax = fig.add_axes([0.075, 0.20, 0.905, 0.72])

    x, ticks, labels, gap = 0.0, [], [], 0.9
    for ds in DS_ORDER:
        g = b[b.dataset == ds]
        start = x
        for i, (_, r) in enumerate(g.iterrows()):
            c = DS_HUE[ds][int(r.m_ord)]
            ax.bar(x, r.hit3_L1, width=0.4, color=c, zorder=3)
            ax.bar(x + 0.42, r.hit3_L5, width=0.4, color=c, zorder=3,
                   hatch="////", edgecolor="black", linewidth=0.45)
            x += 1.0
        ticks.append((start + x - 1.0 + 0.21) / 2)
        labels.append(DS_SHORT[ds])
        x += gap

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("published values reproduced\nto 3 s.f. (%)", labelpad=3)
    ax.set_xlim(-0.7, x - gap - 0.3)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=2)
    ax.grid(axis="x", visible=False)

    mh = [Patch(fc=DS_HUE["esol"][i], ec="none",
                label=b[b.tag == t].model.iloc[0]) for i, t in enumerate(MODEL_ORDER)]
    l1 = ax.legend(handles=mh, loc="upper right", fontsize=6.6, handlelength=1.0,
                   labelspacing=0.2, title="shade = model", title_fontsize=6.6,
                   bbox_to_anchor=(1.0, 1.02))
    ax.add_artist(l1)
    ax.legend(handles=[Patch(fc="#cfcfcf", ec="none", label="published SMILES"),
                       Patch(fc="#cfcfcf", ec="black", lw=0.45, hatch="////",
                             label="substituted string")],
              loc="upper right", fontsize=6.6, handlelength=1.0, labelspacing=0.2,
              bbox_to_anchor=(0.78, 1.02))
    save(fig, "fig5_mitigation")


# =============================================================== three board candidates
def board():
    d = pd.read_csv(os.path.join(RES, "blinding_leaderboard.csv"))
    d["m_ord"] = d.tag.map({t: i for i, t in enumerate(MODEL_ORDER)})
    return d.sort_values(["ds_ord", "m_ord"])


def fig_board_slope(d, fname="figX_board_slope"):
    """Chosen variant -- slopegraph on the error axis, one panel per benchmark.

    Three renderings were built and compared. A bump chart on rank slots is the cleanest reading
    of "the ordering changes", but it discards magnitude, so a swap worth 0.001 kcal/mol draws
    the same line as one worth 0.7 -- unacceptable here, where the LD50 ranks are the unstable
    ones. A scatter against the diagonal carries magnitude and shows that the best-looking models
    fall furthest, but it cannot show an ordering at all, which is the claim.

    This one carries both, and it also shows the thing neither of the others does: the published
    side is spread over more than a decade and the blinded side converges into a band. What
    blinding removes is not one model's advantage but the benchmark's ability to separate the
    models at all.
    """
    use_serif()
    # The axes box is squashed to 60% of its former height (1.98in -> 1.19in). The top and bottom
    # strips hold the benchmark title and the two-line x tick labels, so they are text and must
    # not scale with it: they are held at the 0.385in they had at 2.75in and the fractions are
    # recomputed against the new figure height.
    ax_h, pad = 0.60 * 2.75 * 0.72, 0.385
    fh = ax_h + 2 * pad
    fig = plt.figure(figsize=(W, fh))
    gs = fig.add_gridspec(1, 3, wspace=0.62, left=0.085, right=0.995,
                          top=1 - pad / fh, bottom=pad / fh)
    for k, ds in enumerate(DS_ORDER):
        ax = fig.add_subplot(gs[k])
        g = d[d.dataset == ds].sort_values("medae_L5")
        for _, r in g.iterrows():
            ax.plot([0, 1], [r.medae_L1, r.medae_L5], "-o", ms=3.4, lw=1.6,
                    color=DS_HUE[ds][int(r.m_ord)], zorder=3)
        ax.set_yscale("log")
        # Set the limits rather than autoscaling them, because the label declutter below needs to
        # know the log span before anything is drawn.
        v = np.log10(np.concatenate([g.medae_L1.to_numpy(), g.medae_L5.to_numpy()]))
        lo, hi = v.min(), v.max()
        ax.set_ylim(10 ** (lo - 0.08 * (hi - lo)), 10 ** (hi + 0.08 * (hi - lo)))
        ax.set_xlim(-0.12, 1.95); ax.set_xticks([0, 1])
        # The blinded side converges -- that is the panel's point, and at this height it is also
        # what makes the four labels overlap on LD50, where the whole panel spans a third of a
        # decade. Walk them upward into a minimum pitch of one line of type, then shift the stack
        # back so it stays centred on where the points actually are. No leader lines: at a 6pt
        # offset each label is still nearest its own dot in every panel.
        FS = 5.9
        pitch = (FS * 1.25) / (ax_h * 72.0) * (hi - lo) * 1.16   # 1.16 = the 8% pad on each side
        ys = np.log10(g.medae_L5.to_numpy()).astype(float)
        adj = ys.copy()
        for j in range(1, len(adj)):
            adj[j] = max(adj[j], adj[j - 1] + pitch)
        adj -= adj.mean() - ys.mean()
        for j, (_, r) in enumerate(g.iterrows()):
            ax.annotate(r.model.replace("Claude ", "").replace("GPT-5.6 ", ""),
                        (1, 10 ** adj[j]), xytext=(6, 0), textcoords="offset points",
                        fontsize=FS, color=DS_HUE[ds][int(r.m_ord)], va="center",
                        xycoords="data", annotation_clip=False)
        ax.set_xticklabels(["published\nSMILES", "substituted\nstring"], fontsize=6.6)
        ax.set_title(DS_SHORT[ds], fontsize=8.5, pad=4)
        if k == 0:
            ax.set_ylabel("median absolute error", labelpad=2)
        # The spread between best and worst model, before and after, is the panel's second
        # point. It was tried as an in-plot annotation and landed on top of the lines and the
        # axis labels in every placement worth trying, so it lives in the caption instead.
        ax.grid(axis="x", visible=False); ax.tick_params(length=2)
    save(fig, fname)


def fig_board_rank(d):
    """Variant 2 -- bump chart on rank positions, all three benchmarks in one panel.

    Strips the magnitude away entirely and shows only what a leaderboard is: an ordering. Every
    model occupies a rank slot on the left and a rank slot on the right, per benchmark.
    """
    use_serif()
    fig = plt.figure(figsize=(W, 2.4))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.06, right=0.995, top=0.86, bottom=0.14)
    for k, ds in enumerate(DS_ORDER):
        ax = fig.add_subplot(gs[k])
        g = d[d.dataset == ds]
        for _, r in g.iterrows():
            c = DS_HUE[ds][int(r.m_ord)]
            ax.plot([0, 1], [r.rank_mae_L1, r.rank_mae_L5], "-o", ms=4.5, lw=2.0, color=c,
                    zorder=3)
            ax.annotate(r.model.replace("Claude ", "").replace("GPT-5.6 ", ""),
                        (0, r.rank_mae_L1), xytext=(-5, 0), textcoords="offset points",
                        fontsize=5.8, color=c, va="center", ha="right")
        ax.set_ylim(4.6, 0.4); ax.set_yticks([1, 2, 3, 4])
        ax.set_xlim(-0.95, 1.12); ax.set_xticks([0, 1])
        ax.set_xticklabels(["published", "blinded"], fontsize=6.8)
        ax.set_title(DS_SHORT[ds], fontsize=8, pad=4)
        if k == 0:
            ax.set_ylabel("rank by median error", labelpad=2)
        ax.grid(axis="x", visible=False); ax.tick_params(length=2)
    save(fig, "figX_board_rank")


def fig_board_scatter(d):
    """Variant 3 -- error under blinding against error without it, one point per cell.

    A cell on the diagonal is a model whose score did not depend on the structure string. The
    vertical distance is what blinding costs, and the horizontal position is how good the model
    looked before it. The best-looking models sit furthest left and fall furthest.
    """
    use_serif()
    fig = plt.figure(figsize=(W * 0.62, 2.7))
    ax = fig.add_axes([0.145, 0.155, 0.83, 0.80])
    lo, hi = 0.01, 2.2
    ax.plot([lo, hi], [lo, hi], "--", color="#9aa0a6", lw=1.0, zorder=1)
    ax.annotate("no cost to blinding", (0.9, 0.9), fontsize=6.2, color=INK2,
                rotation=38, ha="center")
    for _, r in d.iterrows():
        c = DS_HUE[r.dataset][int(r.m_ord)]
        ax.scatter(r.medae_L1, r.medae_L5, s=34, color=c, zorder=3, edgecolor="white", lw=0.5)
        ax.annotate(r.model.replace("Claude ", "").replace("GPT-5.6 ", ""),
                    (r.medae_L1, r.medae_L5), xytext=(5, -1), textcoords="offset points",
                    fontsize=5.4, color=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("median error, published SMILES", labelpad=2)
    ax.set_ylabel("median error, substituted string", labelpad=2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4.5, color=DS_HUE[x][1],
                              label=DS_SHORT[x]) for x in DS_ORDER],
              loc="lower right", fontsize=6.6, handletextpad=0.2, labelspacing=0.2)
    ax.tick_params(length=2)
    save(fig, "figX_board_scatter")


def fig_blindmap(fname="fig_blindmap", smax=None):
    """The blinding cells as the map of Fig. 2, every cell split: left L1, right L5.

    The same construction as `_fig4_paired_map`, and deliberately so -- that panel splits a cell
    by reasoning budget and this one splits it by whether the structure string was blinded, so a
    reader who has learned to read one has learned to read the other. What it adds to the
    subsection is a verdict: `blinding_sweep_t1024.csv` carries hit3 against an accuracy-matched
    floor and no significance at all, so until now "three cells still retrieve under L5" rested
    on a rate rather than on the rungs. It does not any more.

    THE PANEL IS 4 x 3, not 22 x 12, so the cells are wide. Wide enough to carry their hit3, in
    fact, which the paired map cannot do -- and since the argument here is about three named
    cells rather than about a pattern, the numbers earn their place.

    smax: pass the map's own value to put this panel on the map's colour scale. Left None it
    normalises to its own strongest cell, which is the honest default for a separate experiment
    with its own BH family -- the q-values are not comparable across families, so a shared ramp
    would invite exactly the comparison the family split exists to prevent.
    """
    from figures_paper import SIG_CMAP, NODATA, NOSIGNAL, sig_strength, sig_norm

    p = os.path.join(RES, "blinding_map.csv")
    if not os.path.exists(p):
        print("  (no blinding_map.csv -- run src/analyze_blinding_map.py)")
        return
    d = pd.read_csv(p)
    d["sig"] = sig_strength(d)
    names = d.drop_duplicates("tag").set_index("tag").model.to_dict()
    tags = [t for t in MODEL_ORDER if t in set(d.tag)]
    dss = [x for x in DS_ORDER if x in set(d.dataset)]

    def half(level, col="sig"):
        return (d[d.level == level].pivot_table(index="tag", columns="dataset", values=col)
                .reindex(index=tags, columns=dss).to_numpy(float))

    L1, L5 = half("L1"), half("L5")
    H1, H5 = half("L1", "hit3"), half("L5", "hit3")
    V1 = (d[d.level == "L1"].pivot_table(index="tag", columns="dataset", values="verdict",
                                         aggfunc="first").reindex(index=tags, columns=dss)
          .to_numpy(object))
    V5 = (d[d.level == "L5"].pivot_table(index="tag", columns="dataset", values="verdict",
                                         aggfunc="first").reindex(index=tags, columns=dss)
          .to_numpy(object))
    if smax is None:
        smax = float(np.nanmax(np.concatenate([L1.ravel(), L5.ravel()])))

    use_serif()
    # HALF THE AREA of the first version (0.58 W x 2.61 in), with every font size left where
    # it was, so the type is ~1.4x larger relative to the panel rather than smaller. That is
    # only possible because the long header line is gone: at this width it did not fit.
    fig = plt.figure(figsize=(0.41 * W, 0.21 * W + 0.51))
    ax = fig.add_subplot(111)
    for i in range(len(tags)):
        for j in range(len(dss)):
            for x0, v, h, vd in ((j - .5, L1[i, j], H1[i, j], V1[i, j]),
                                 (j, L5[i, j], H5[i, j], V5[i, j])):
                # `untestable` is a hatch and `no-signal` a flat grey, as in the map: a cell the
                # rungs could not reach must not be filled white, which is the colour of a cell
                # they reached and cleared.
                if vd == "untestable":
                    ax.add_patch(Rectangle((x0, i - .5), .5, 1, fill=False, hatch="////",
                                           ec="#c2c0bb", lw=0.0, alpha=0.9))
                elif vd == "no-signal":
                    ax.add_patch(Rectangle((x0, i - .5), .5, 1, fc=NOSIGNAL, ec="none"))
                else:
                    fc = NODATA if not np.isfinite(v) else SIG_CMAP(sig_norm(v, smax))
                    ax.add_patch(Rectangle((x0, i - .5), .5, 1, fc=fc, ec="none"))
                if np.isfinite(h):
                    dark = np.isfinite(v) and sig_norm(v, smax) >= 0.62
                    ax.text(x0 + .25, i, f"{h:.0f}" if h >= 10 else f"{h:.1f}", fontsize=6.6,
                            ha="center", va="center", color="white" if dark else INK)
    ax.set_xlim(-.5, len(dss) - .5); ax.set_ylim(len(tags) - .5, -.5)
    # The halves are labelled ON the axis rather than in a header line. At this canvas width a
    # sentence naming them is wider than the whole figure, and it was the only thing forcing the
    # figure to be big: "left half: L1, published SMILES  right half: L5, blinded" is 2.7 in of
    # type. Two ticks per benchmark say the same thing in four characters and cannot drift out
    # of register with the columns they describe. What the two levels ARE stays in the caption.
    ax.set_xticks([j + dx for j in range(len(dss)) for dx in (-.25, .25)])
    ax.set_xticklabels(["L1", "L5"] * len(dss), fontsize=6.4)
    for j, x in enumerate(dss):
        ax.text(j, len(tags) - .5 + 0.42, DS_SHORT[x], fontsize=7.4, ha="center", va="top",
                color=INK, clip_on=False)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([names[t] for t in tags], fontsize=7.0)
    ax.set_xticks(np.arange(-.5, len(dss), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(tags), 1), minor=True)
    ax.grid(which="minor", color="#e4e2dd", lw=0.7); ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # The map key, laid out by the same function fig2 and fig4a use: ramp at the top left,
    # caption above it, `none` / `strongest` below it. The wording is shortened only where it
    # has to be -- this panel's axes is 2.0 in wide against fig2's 6.2, so fig2's full caption
    # would be wider than the figure. The POSITIONS are the shared thing, and they come from
    # sig_ramp rather than from numbers repeated here.
    sig_ramp(ax, width=0.44, caption="significance of the retentions",
             right="strongest here", fs=6.8, fs_small=6.4)
    save(fig, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", action="store_true")
    a = ap.parse_args()
    d = board()
    if not a.variants:
        fig5_bars(load())
        fig_board_slope(d, "fig_blindboard")      # the one the paper uses
        fig_blindmap()
    fig_board_slope(d)
    fig_board_rank(d)
    fig_board_scatter(d)


if __name__ == "__main__":
    main()
