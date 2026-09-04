"""
Figures for the arXiv manuscript.

One file, six figures, every number read from results/*.csv so a figure cannot drift from the
data behind it. Deliberately separate from make_figures.py, which is the exploratory set.

  fig0_abstract    graphical abstract, page 1
  fig1_protocol    what the measurement is
  fig2_map         the contamination map + what depth buys
  fig3_nulls       the null comparison and the false positives it manufactures
  fig4_ladder      recall against the deliberation budget
  fig5_mitigation  four attempts to interrupt retrieval

    python src/figures_paper.py

Colour does one job each:
  regime      ordered severity -> one hue, light to dark. 'untestable' is missing power, not a
              severity, so it is neutral grey with a hatch.
  benchmark   categorical, at most four at a time.
  polarity    change under an intervention -> diverging, grey midpoint.
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.patheffects as pe

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
REG = os.path.join(ROOT, "src", "registry")
os.makedirs(FIG, exist_ok=True)

SURFACE = "#ffffff"
INK, INK2, GRID = "#111111", "#5a5a5a", "#dddad3"
REGIME_COLOR = {"clean": "#cfe0f6", "trace": "#87b3e6", "partial": "#3b7fcc", "heavy": "#123c72"}
REGIME_ORDER = ["clean", "trace", "partial", "heavy"]
NODATA = "#e6e4df"
# Two kinds of cell carry no verdict from the rungs, and they are not the same kind. A cell whose
# unconditional first-figure rate is itself at or below the label floor is a NEGATIVE -- the model
# does not place the order of magnitude, so there is nothing that could have been retrieved -- and
# it gets a flat fill. A cell that merely ran out of power stays hatched, because it is a blank.
NOSIGNAL = "#f0efec"
# The map is coloured by how strong the evidence in a cell is, not by which class it fell into.
# SIG_CMAP runs white (nothing above the floor) to the old 'heavy' navy (strongest in the panel).
SIG_CMAP = LinearSegmentedColormap.from_list(
    "sig", ["#ffffff", "#e3edf9", "#a9c8ec", "#5f95d8", "#2b62ab", REGIME_COLOR["heavy"]])
SIG_ALPHA = 0.05          # the flagging threshold: q at or above this is white
SIG_GAMMA = 0.5           # sqrt-compression, or the positive control owns the whole ramp
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#8d63c6", "#c9a227"]
DIV_LO, DIV_HI = "#2a78d6", "#d1442f"
SEQ = LinearSegmentedColormap.from_list(
    "seqblue", ["#e8f0fb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5, "axes.labelsize": 7.8, "axes.titlesize": 8.2,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "legend.fontsize": 6.8, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

W = 6.3          # text width in inches at 2.5 cm margins on A4

# Schematic panels are typeset, not plotted, so they follow the manuscript rather than the
# plotting defaults: one size for every string, the body font, and the same grey box everywhere.
# The figure is placed at \textwidth and W equals \textwidth, so a point here is a point on the
# page -- SCHEM_FS is directly comparable to the 11 pt body and the 10 pt caption.
SCHEM_FS = 8.0                       # every string in a schematic panel
SCHEM_HEAD = 8.6                     # panel headings only
SCHEM_BOX = dict(boxstyle="round,pad=0.45", fc="#f6f7f9", ec=GRID, lw=0.7)


def use_serif():
    """Match the manuscript's body font (\\usepackage{times}).

    Not text.usetex: that needs a LaTeX run per figure and dies on a missing package in a way
    that is hard to diagnose from a build log. Times plus the STIX math set is visually the
    same at these sizes and costs nothing.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

# Kept out of every MAIN-TEXT figure so the panel is one panel everywhere, and reported in the
# appendix rather than dropped. Three reasons, all of them "this cell exists but not in the arm
# the paper is about":
#   kimik26, glm5, glm52, nemotron3u, qwen35, dsv4pro -- endpoints that ignore a thinking budget,
#       so the controlled arm could not dose them at all
#   opus48, opus45 -- in the zero-shot screen and the blinding sweep, but the controlled arm
#       never ran them, so they have no cell in the map
#   qm9 -- the extension was stopped there on cost after four cells (EXPERIMENT_LOG); a benchmark
#       with 4 of 22 cells beside benchmarks with 22 invites a comparison the data cannot support
DROP_MODELS = ("kimik26", "glm5", "glm52", "nemotron3u", "qwen35", "dsv4pro",
               "opus48", "opus45")
DROP_DATASETS = ("qm9",)


def _drop_names():
    """Display names of the dropped models, because not every result file carries the tag.

    `randomization_control.csv` keys on the human-readable name only, so filtering on tags alone
    left Claude Opus 4.8 and GLM 5.2 in the randomised-SMILES panel while the text quoted the
    panel-only numbers.
    """
    reg = json.load(open(os.path.join(REG, "models.json")))["models"]
    return {m["name"] for m in reg if m["tag"] in DROP_MODELS}


def drop_extras(d):
    """Remove the models and benchmarks that only appear in the appendix."""
    if d is None or not len(d):
        return d
    names = DROP_MODELS + tuple(_drop_names())
    for col in ("tag", "model"):
        if col in d.columns:
            d = d[~d[col].isin(names)]
    if "dataset" in d.columns:
        d = d[~d.dataset.isin(DROP_DATASETS)]
    return d.copy()


# --------------------------------------------------------------------------- data
def _classified(name, fallback=None):
    """The current verdicts (src/classify.py), with the old file as a fallback.

    _v3 tests significance first -- R23 against a conditional binomial where the rung has power,
    R12 where it does not -- and splits the significant cells by hit3. `decided_at` is kept as a
    shim so every panel that stars a cell decided on the 1->2 rung keeps working.
    """
    p3 = os.path.join(RES, f"{name}_v3.csv")
    d = pd.read_csv(p3 if os.path.exists(p3)
                    else os.path.join(RES, fallback or f"{name}_v2.csv"))
    if "deep" in d.columns and "decided_at" not in d.columns:
        d["decided_at"] = np.where(d.deep, 3, np.where(d.testable, 2, 0))
    return d


def label_floor():
    """The label-only coincidence floor per benchmark (src/label_floor.py), or {} if absent.

    This is the null the verdicts now rest on, so any panel that draws the OTHER nulls has to
    draw this one too or it shows the reader everything except the thing being used.
    """
    p = os.path.join(RES, "label_floor.csv")
    if not os.path.exists(p):
        return {}
    return pd.read_csv(p).set_index("dataset").to_dict("index")


def load():
    # The zero-shot screen is retired: it measured what the minimum-reasoning arm measures,
    # on an older pipeline, with a panel that was never the reported one. Its files are
    # under results/_archive_zeroshot/ so that any code still reaching for them fails loudly
    # rather than quietly drawing them.
    L = pd.read_csv(os.path.join(RES, "ladder_summary.csv"))
    # The blinding sweep exists in two arms and the figure must show the one the text reports.
    # The legacy file ran each endpoint at its own default reasoning, on a molecule set drawn
    # with `disjoint_from` set, and against version 1 of the substitution table; the controlled
    # arm ran four models at the map's own 1,024-token budget. They are not comparable cell by
    # cell -- their molecule sets overlap by 10%, 27% and 1% on the three benchmarks -- so the
    # figure takes the controlled arm where it exists and never mixes them.
    _bl = os.path.join(RES, "blinding_sweep_t1024.csv")
    B = pd.read_csv(_bl if os.path.exists(_bl) else os.path.join(RES, "blinding_sweep.csv"))
    R = pd.read_csv(os.path.join(RES, "randomization_control.csv"))
    # The budget-arm simulated floors, not the screening ones: the screen is retired.
    M = pd.read_csv(os.path.join(RES, "smooth_error_null_budget.csv"))
    # The controlled-budget arm: 22 models x 12 benchmarks, the SAME 500 molecules per cell.
    # This is the headline map; S is the wide, shallow zero-shot screen and now lives in the
    # appendix.
    A = _classified("budget_3sig", fallback="budget_3sig_pad_v2.csv")
    # The verdict file holds both arms since the minimum-deliberation arm was run. The map is
    # the 1,024-token arm; without this filter `pivot_table(aggfunc="first")` silently draws
    # whichever row happens to come first and the whole panel is a mixture.
    # The randomised-SMILES control also exists in two versions, and the figure has to use the
    # one the text reports. `randomization_control.csv` is the original 13-model ESOL-only run;
    # the controlled arm now carries a `t1024r` arm -- the same four models and three benchmarks
    # as the canonical cells, same molecules, same budget, only the SMILES rewritten. Where that
    # exists it is the paired comparison and the old file is not.
    if "arm" in A.columns and {"t1024", "t1024r"} <= set(A.arm):
        can = A[A.arm == "t1024"].set_index(["dataset", "tag"])
        ran = A[A.arm == "t1024r"].set_index(["dataset", "tag"])
        k = can.index.intersection(ran.index)
        R = pd.DataFrame({"dataset": [i[0] for i in k], "tag": [i[1] for i in k],
                          "hit3_can": can.loc[k, "hit3"].to_numpy(),
                          "hit3_rand": ran.loc[k, "hit3"].to_numpy()})
    if "arm" in A.columns:
        A = A[A.arm == "t1024"].copy()
    # The blinding sweep is its own declared experiment with its own four models, one of them
    # (Kimi K2.6) deliberately a model with nothing to recall. It is not filtered to the map's
    # panel -- only the single orphan cell of a model that appears nowhere else is dropped.
    B = B[B.tag != "opus48"].copy()
    L, R, M, A = (drop_extras(x) for x in (L, R, M, A))
    models = json.load(open(os.path.join(REG, "models.json")))["models"]
    order = [m["tag"] for m in models]
    A["ord"] = A.tag.map({t: i for i, t in enumerate(order)})
    return L, B, R, M, A, {m["tag"]: m for m in models}


DS_LABEL = {"esol": "ESOL", "freesolv": "FreeSolv", "lipophilicity": "Lipophilicity",
            "bace": "BACE", "aqsoldb": "AqSolDB", "caco2": "Caco-2", "ld50": "LD50",
            "ppbr": "PPBR", "qm7": "QM7", "qm8": "QM8", "qm9": "QM9",
            "antiviral": "Antiviral\n(recency ctrl.)", "boilingpoint": "Boiling pt.\n(pos. control)"}
DS_SHORT = {k: v.split("\n")[0] for k, v in DS_LABEL.items()}
DS_ORDER = ["freesolv", "esol", "ld50", "lipophilicity", "aqsoldb", "caco2", "bace", "ppbr",
            "qm7", "qm8", "qm9", "antiviral", "boilingpoint"]

# Benchmarks the controlled map flags on at least one cell, i.e. the ones with something stored.
# Defined once because two figures split on it (fig4's panels and fig7's colour) and they drifted:
# fig7 still called only FreeSolv/ESOL/LD50 recalled after AqSolDB had joined them, and QM7 was in
# neither list after the floor change made its climb a real excess.
STORED = ["ld50", "esol", "freesolv", "aqsoldb", "qm7"]


def sig_strength(d):
    """Per-cell evidence strength, as $-\\log_{10}q$ of the better of the two rungs.

    Which rung to take was the open question. The minimum over the two is the right answer and
    not a compromise: a cell is flagged when EITHER rung clears BH, so min(q) < alpha holds for
    exactly the 89 flagged cells of the controlled arm and for no others. Colouring on it
    therefore leaves every clean cell white without a second rule, and a cell where R23 has no
    power is carried by R12 rather than dropped. Averaging q-values would not do that -- an
    average is not a tail probability, and half the cells have only one rung to average.
    """
    q = d[["q_R12", "q_R23"]].min(axis=1, skipna=True)
    # A cell with neither rung tested has no q at all, and must stay NaN: floored to 1e-300 it
    # would come out as the strongest evidence in the panel rather than as no evidence.
    return -np.log10(q.mask(q <= 0, 1e-300))


def sig_norm(s, smax):
    """Map evidence strength onto 0..1: 0 at the flagging threshold, 1 at the panel's strongest.

    The raw range spans two hundred decades, because the positive control is a set of textbook
    constants and every model reproduces it. Linear on that range puts every real cell in the
    first two percent of the ramp, so the scale is sqrt-compressed.
    """
    s0 = -np.log10(SIG_ALPHA)
    u = (np.asarray(s, float) - s0) / max(smax - s0, 1e-9)
    return np.clip(u, 0.0, 1.0) ** SIG_GAMMA
DEEP = {"esol", "freesolv", "lipophilicity", "ld50", "qm7", "qm8", "antiviral", "boilingpoint"}


def sig_ramp(ax, width=0.42, caption="significance of the retentions above the floor",
             right="strongest in the panel", y=None, fs=7.2, fs_small=7.0):
    """The colour key shared by every map in the paper. Fig. 2 is the reference.

    Layout, and it is the same in all four maps (fig2, the zero-shot copy, fig4a and the blinding
    map): the ramp sits at the TOP LEFT of the map axes, its caption on the line above it, `none`
    and `right` on the line below it, flush with the ramp's two ends. Anything else the panel has
    to say goes to the RIGHT of that block, which is where fig2 keeps its `no signal` legend. A
    key that migrates between maps of the same quantity reads as a different key to anyone
    skimming, which is the failure this function exists to make impossible.

    EVERYTHING IS MEASURED IN POINTS, NOT IN AXES FRACTION, and that is the fix rather than a
    refinement. Every map here has a different axes geometry -- fig2 sizes its figure from the
    row count, fig4a is one cell of a gridspec whose height ratios are solved from the same
    count, the blinding map is a plain subplot -- so one fraction cannot mean one distance in
    all three. fig2's original offsets (a 0.028-tall ramp at y=1.035, labels 0.35 ramp-heights
    below it) put the `none` line 0.005 clear of the axes top in fig2, which works, and 0.009
    INSIDE it in fig4a, which prints the label across the first row of cells. Deriving the ramp
    height from the row count instead gave a 30 pt ramp in fig4a against fig2's 10.

    So the axes box is asked for once and the stack is built upward from its top edge in real
    units: 2 pt of air, the `none` line, 1.5 pt, the 9.5 pt ramp, 2 pt, the caption. Nothing can
    land inside the map whatever the panel's proportions are, and the key is the same physical
    object in all four maps.

    `y` is kept in the signature for callers that want the block lifted further, but the default
    of None means "sit on the axes top", which is what every map uses.
    """
    fig = ax.figure
    fig.canvas.draw()
    h_pt = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).height * 72.0
    h_pt = max(h_pt, 1.0)

    def f(pt):
        return pt / h_pt                      # points -> axes fraction, this axes only

    base = (1.0 + f(2.0)) if y is None else y
    y_lab = base                              # bottom of the none/strongest line
    y_bar = y_lab + f(fs_small * 1.15 + 1.5)  # bottom of the ramp
    cb = ax.inset_axes([0.0, y_bar, width, f(9.5)], transform=ax.transAxes)
    cb.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=SIG_CMAP, extent=(0, 1, 0, 1))
    cb.set_xticks([]); cb.set_yticks([])
    for s in cb.spines.values():
        s.set_color("#c2c0bb"); s.set_linewidth(0.5)
    ax.text(0.0, y_bar + f(9.5 + 2.0), caption, fontsize=fs, color=INK2, ha="left",
            va="bottom", transform=ax.transAxes)
    ax.text(0.0, y_lab, "none", fontsize=fs_small, color=INK2, ha="left", va="bottom",
            transform=ax.transAxes)
    ax.text(width, y_lab, right, fontsize=fs_small, color=INK2, ha="right", va="bottom",
            transform=ax.transAxes)
    return cb


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(p + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def panel(ax, letter, title=None):
    ax.text(-0.055, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right", color=INK)
    if title:
        ax.set_title(title, loc="left")


# =========================================================================== fig 0
def fig0(A, L, B):
    """Graphical abstract: what the paper does, in one strip.

    Deliberately number-free. The shapes are drawn from the real data, but every axis is
    qualitative -- a graphical abstract is read in two seconds, and a reader who wants a
    figure with numbers on it should be reading Figs. 2-5 instead.
    """
    fig = plt.figure(figsize=(W, 1.72))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.24, 0.84, 0.96, 0.96], wspace=0.46,
                          left=0.005, right=0.995, top=0.855, bottom=0.155)
    bare = dict(labelleft=False, labelbottom=False, left=False, bottom=False, length=0)
    arrow = dict(arrowstyle="-|>", lw=0.8, color=INK2)

    # (1) the measurement --------------------------------------------------
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("ask for a published value", fontsize=7.4)
    ax.text(0.0, 0.80, "$\\tt{Clc1ccc(Cl)cc1}$", fontsize=7.0, va="top", color=INK2)
    ax.text(0.60, 0.70, "no examples,\nno tools", fontsize=6.2, va="top", color=INK2,
            linespacing=1.25)
    ax.annotate("", xy=(0.44, 0.52), xytext=(0.22, 0.665),
                arrowprops=dict(connectionstyle="arc3,rad=-0.22", **arrow))
    # three digit slots, filled the same way in both rows -- no numerals needed
    for y, lab in ((0.365, "published"), (0.145, "answered")):
        ax.text(0.0, y, lab, fontsize=6.4, va="center", color=INK2)
        for j in range(3):
            ax.add_patch(Rectangle((0.42 + 0.085 * j, y - 0.068), 0.062, 0.136,
                                   fc=REGIME_COLOR["heavy"], ec="none", zorder=3))
    ax.add_patch(Rectangle((0.395, 0.055), 0.30, 0.395, fill=False, lw=0.8,
                           ec=REGIME_COLOR["partial"], zorder=4))
    ax.text(0.72, 0.255, "every digit\nagrees", fontsize=6.4, va="center",
            color=REGIME_COLOR["partial"], linespacing=1.15)

    # (2) it is widespread -------------------------------------------------
    ax = fig.add_subplot(gs[1])
    n_by = (A[A.ds_class == "measured"].groupby("dataset")
            .apply(lambda d: (d.regime.isin(["heavy", "partial", "trace"])).sum(),
                   include_groups=False))
    keys = ["freesolv", "esol", "ld50", "lipophilicity"]
    vals = [n_by.get(k, 0) for k in keys]
    ax.barh(range(len(keys))[::-1], vals, color=[REGIME_COLOR["heavy"]] * 2 +
            [REGIME_COLOR["partial"], REGIME_COLOR["clean"]], height=0.62)
    ax.set_yticks(range(len(keys))[::-1])
    ax.set_yticklabels([DS_SHORT[k] for k in keys], fontsize=6.6)
    ax.set_xlim(0, 24)
    ax.set_xlabel("models that recall it  $\\rightarrow$", labelpad=2.5, fontsize=6.6)
    ax.set_title("not every benchmark", fontsize=7.4)
    ax.grid(visible=False); ax.tick_params(axis="x", **bare); ax.tick_params(axis="y", length=0)

    # (3) reasoning gate ---------------------------------------------------
    ax = fig.add_subplot(gs[2])
    for ds, c in (("ld50", CAT[1]), ("esol", CAT[0]), ("qm8", "#9aa0a6")):
        d = L[(L.dataset == ds) & (L.tag == "gem3flash")].sort_values("reasoning_med")
        if d.empty:
            continue
        ax.plot(np.maximum(d.reasoning_med.to_numpy(), 30.0), d.hit3, "-o", color=c,
                ms=2.6, lw=1.5, label=DS_SHORT[ds])
    ax.set_xscale("log"); ax.set_xlim(25, 9000); ax.set_ylim(-3, 52)
    ax.set_xlabel("more reasoning  $\\rightarrow$", labelpad=2.5, fontsize=6.6)
    ax.set_ylabel("recall  $\\rightarrow$", labelpad=2.5, fontsize=6.6)
    ax.set_title("reasoning unlocks it", fontsize=7.4)
    ax.legend(loc="upper left", handlelength=1.0, borderpad=0.1, labelspacing=0.15,
              fontsize=6.2, handletextpad=0.4)
    ax.grid(visible=False); ax.tick_params(which="both", **bare)

    # (4) mitigation -------------------------------------------------------
    ax = fig.add_subplot(gs[3])
    for _, r in B.sort_values("hit3_L1", ascending=False).iterrows():
        strong = r.hit3_L1 > 8
        c = CAT[0] if strong else "#b9bec4"
        ax.plot([0, 1], [r.hit3_L1, r.hit3_L5], "-o", color=c, lw=1.0, ms=2.4,
                alpha=0.95 if strong else 0.6, zorder=3 if strong else 2)
    ax.set_xlim(-0.30, 1.30); ax.set_ylim(-3, 58); ax.set_xticks([0, 1])
    ax.set_xticklabels(["published\nstructure", "substituted\nstructure"], fontsize=6.4)
    ax.set_ylabel("recall  $\\rightarrow$", labelpad=2.5, fontsize=6.6)
    ax.set_title("blinding removes it", fontsize=7.4)
    ax.grid(visible=False)
    ax.tick_params(axis="y", **bare); ax.tick_params(axis="x", length=0)
    save(fig, "fig0_abstract")


# =========================================================================== fig 1
def fig1(A):
    """Protocol schematic: prompt -> digits -> nested counts -> null -> verdict.

    Panel (b) introduces the palette the rest of the paper uses, so it has to show every class
    and colour each one the way fig2 does. It used to show two hand-entered cells, with the
    clean one drawn in the *partial* colour -- the same blue that means `contaminated` two pages
    later. `A` is the controlled-budget arm; the ladders are read from it.
    """
    use_serif()
    # Vertically tight. The schematic columns are three boxes and a key, all of them fixed-height
    # objects, so the height is set from what they need rather than left at whatever the plot
    # column wanted: 2.30 in against 3.20 before, with the slack taken out between the boxes.
    fig = plt.figure(figsize=(W, 2.30))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.21, 0.73, 1.30], wspace=0.30,
                          left=0.005, right=0.980, top=0.925, bottom=0.035)

    # A text bbox is sized by its longest line, so two boxes with different text are never the
    # same width. These are drawn explicitly instead: the patch spans the column, the text sits
    # inside it, and the height follows from the line count so the geometry below can be solved
    # rather than nudged.
    AX_IN = (0.925 - 0.035) * 2.30          # panel height in inches
    LINE_H = SCHEM_FS * 1.5 / 72 / AX_IN    # one line, in axes fraction
    PAD_H = 0.45 * SCHEM_FS / 72 / AX_IN    # bbox pad, top and bottom

    def text_h(ax, lines):
        """Rendered height of the block, in axes fraction. Measured, not derived.

        Deriving it from the line count is what left half a line of dead space at the foot of
        every box: a run of n lines is (n-1) spacings plus one line BOX, and a line box is not
        one font size -- it carries the ascent and descent of the face. Matplotlib knows the
        answer exactly, so ask it.
        """
        t = ax.text(0, -5, "\n".join(lines), fontsize=SCHEM_FS, va="top", linespacing=1.5)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transAxes.inverted())
        t.remove()
        return bb.height

    def boxed(ax, top, lines, h, x0=0.005, x1=0.995):
        """Rounded box of fixed width with `lines` inside, hung from `top`. Returns its bottom."""
        # The panel is 2.15 x 2.05 in, so one x-unit and one y-unit are almost the same length
        # and mutation_aspect stays at its default; otherwise the corners come out as ellipses.
        ax.add_patch(FancyBboxPatch((x0, top - h), x1 - x0, h,
                                    boxstyle="round,pad=0,rounding_size=0.018",
                                    fc=SCHEM_BOX["fc"], ec=SCHEM_BOX["ec"], lw=SCHEM_BOX["lw"],
                                    clip_on=False, zorder=2))
        ax.text(x0 + 0.035, top - PAD_H, "\n".join(lines), fontsize=SCHEM_FS, va="top",
                linespacing=1.5, zorder=3)
        return top - h

    def head(ax, letter, text, y=1.03):
        """Panel letter and heading on one line: 'a) one call per molecule'.

        `y` is in AXES coords, so a panel whose axes was moved needs it raised by the
        same distance to keep its heading on the shared line: (b) is shifted down by
        1.35 arrow lengths of the figure, which is 0.152 of its own two-thirds height.
        """
        ax.text(0.0, y, f"{letter}) {text}", fontsize=SCHEM_HEAD, fontweight="bold",
                va="bottom", ha="left", transform=ax.transAxes, color=INK)

    # (a) the query --------------------------------------------------------
    # Three boxes, one size, one fill. The middle one used to be unboxed grey italic prose and
    # the last one a blue box, which made the column read as three kinds of statement when it
    # is one: what is asked, how, and how it is scored.
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    head(ax, "a", "zero-shot prediction")
    L1 = ["22 LLMs  ×  12 datasets", "reasoning enabled", "zero-shot prompt", "dataset named"]
    L2 = ["score predictions by the",
          "number of correct significant figures:",
          "$m_1$: number of correct first figures",
          "$m_2$: number of correct first two figures",
          "$m_3$: number of correct first three figures"]
    # Solved rather than nudged: the two boxes and the arrow are centred as a block, and the gap
    # above the arrow equals the gap below it.
    ARROW_L, GAP = 0.078, 0.020
    h1, h2 = (text_h(ax, L1) + 2 * PAD_H), (text_h(ax, L2) + 2 * PAD_H)
    total = h1 + h2 + ARROW_L + 2 * GAP
    top = 1.0 - (1.0 - total) / 2.0
    b1 = boxed(ax, top, L1, h1)
    _a = ax.annotate("", xy=(0.50, b1 - GAP - ARROW_L), xytext=(0.50, b1 - GAP),
                     arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK2))
    _a.arrow_patch.set_zorder(10)
    boxed(ax, b1 - GAP - ARROW_L - GAP, L2, h2)

    # (b) the nested ladder ------------------------------------------------
    # Two thirds of the column height, anchored to the top so its heading sits on the same line
    # as (a)'s and (c)'s. The curves are three points; the extra height was empty axis.
    ax = fig.add_subplot(gs[1])
    _p = ax.get_position()
    ARROW_IN = 0.154                       # one arrow, in inches, at this figure height
    ax.set_position([_p.x0, _p.y0 + _p.height / 3.0 - 1.35 * ARROW_IN / 2.30,
                     _p.width, _p.height * 2.0 / 3.0])
    head(ax, "b", "possible outcomes", y=1.182)
    lv = ["1 s.f.", "2 s.f.", "3 s.f."]
    x = np.arange(3)
    # `clean` gets a darkened line colour because its fill is chosen to be legible as a map
    # cell, not as a 1 pt curve; `contaminated` takes the dark end of the significance ramp,
    # which is where a cell of this shape lands in Fig. 2.
    LINE = {"clean": "#8fa8c4", "contaminated": REGIME_COLOR["heavy"]}
    # The reference is the MEASURED accuracy-matched floor, not a uniform-digit 10% per rung.
    # Digits cluster, so the real 1->2 floor is ~15%, and drawing 10% there made a `trace` cell --
    # which sits ON its floor at that rung by definition -- look elevated.
    # SCHEMATIC. These four ladders are drawn to show what the classes look like; they are not
    # class medians of the data, and the panel is not evidence for anything.
    #
    # Why it is a schematic and not a measurement. Two attempts to plot real medians here were
    # wrong in opposite ways: against one panel-wide floor, `trace` sat above a line it does not
    # clear (q = 0.46 on that rung) while `heavy` was understated; against a per-class floor, the
    # denominator was conditioned on the verdict the classes come from. Both failures have the
    # same root -- the accuracy-matched floor this study tests against is a function of the
    # model's own errors, so it is not constant across cells and cannot be drawn as one line
    # behind several measured ladders.
    #
    # The floor drawn here is the LABEL-ONLY coincidence rate, a property of the published values
    # and of nothing else: given two published values that agree at n figures, how often do they
    # agree at n+1? It cannot be below 10%: the next digit collides with probability sum_d p_d^2,
    # which by Cauchy-Schwarz is at least (sum_d p_d)^2 / 10 = 1/10, with equality only for
    # uniform digits -- any clustering raises it. Measured over the panel it is 10.3-10.7% per
    # rung, and 14.7% on QM8, whose values cluster hardest.
    #
    # A first pass reported 9.2% and even 0.3%, which that bound says is impossible: it had
    # dropped pairs of equal values, and equal values at DIFFERENT molecules are precisely the
    # coincidence being counted -- and a subset of the 3-figure matches, so R23 collapsed.
    #
    # This is the floor to have in mind for the shapes. The accuracy-matched floor the verdicts
    # are tested against is higher (15.2% median), depends on the model's own errors, and is
    # defined in Appendix~\ref{app:nulls}.
    fl, = ax.plot(x, [100, 10, 1], "--", color="#9aa0a6", lw=1.1, zorder=2)
    # TWO shapes, not four. The four ladders drew the retired five-step verdict palette, in which
    # `trace`, `partial` and `heavy` were separate classes. Under the current scheme a cell is
    # flagged when EITHER rung clears BH and is then coloured by how strongly -- so those three
    # names no longer label anything the reader meets in the map, and drawing them here promised
    # a taxonomy the rest of the paper does not deliver. What the panel has to show is the one
    # contrast the metric rests on: a cell that sits on the coincidence floor at both rungs, and
    # a cell that clears it at both.
    # `clean` is drawn a hair ABOVE the floor rather than on it. On it, the 1.4 pt line hides the
    # 1.1 pt dashed one completely, and the legend then names a floor the reader cannot find.
    SCHEMATIC = {"clean":        [100, 12.0, 1.45],    # on the floor at both rungs
                 "contaminated": [100, 38.0, 24.0]}    # off it at both, and by a long way
    for reg in ("contaminated", "clean"):
        ax.plot(x, SCHEMATIC[reg], "-o", color=LINE[reg], ms=3.2, lw=1.4,
                zorder=4 if reg == "contaminated" else 3,
                label="retrieval" if reg == "contaminated" else "clean")
    # (c) no longer carries a class key -- it defines the significance scale instead -- so the
    # two curves have to name themselves here.
    fl.set_label("floor")
    # Lower left, not upper right: the R23 leader ends at (1.62, 30), and an upper-right legend
    # is drawn on top of it. Under the clean ladder the axes is empty from x = 0 to x = 1.
    leg = ax.legend(loc="lower left", bbox_to_anchor=(-0.03, -0.02), fontsize=6.8,
                    frameon=False, handlelength=1.3, labelspacing=0.22, borderaxespad=0.0,
                    handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.annotate("$R_{23}$", xy=(1.62, 30), xytext=(1.18, 52), fontsize=7.4,
                color=REGIME_COLOR["heavy"],
                arrowprops=dict(arrowstyle="-", lw=0.7, color=REGIME_COLOR["heavy"]))
    ax.annotate("$R_{12}$", xy=(0.60, 52), xytext=(0.14, 26), fontsize=7.4,
                color=REGIME_COLOR["heavy"],
                arrowprops=dict(arrowstyle="-", lw=0.7, color=REGIME_COLOR["heavy"]))
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(lv)
    # NOT hit3: this is retention, conditioned on the 1-s.f. matches. Panel (c) splits the
    # classes on hit3, a rate over the whole benchmark, and the two must not be read off the
    # same axis -- a `partial` cell retains 13% of its 1-s.f. matches at three figures while
    # reproducing 6% of the benchmark.
    ax.set_ylim(0.55, 190)
    ax.set_ylabel("retained from 1 s.f. (%)", labelpad=2)
    # No legend here. The classes are colours, and the place that defines colours is the verdict
    # key in (c) -- printing them twice invited the two to disagree, which is how the clean class
    # once appeared in the partial colour.
    ax.grid(axis="x", visible=False); ax.tick_params(length=2)

    # (c) the null, the verdict, and the key for (b) ------------------------
    ax = fig.add_subplot(gs[2]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _p = ax.get_position()
    ax.set_position([_p.x0 - 1.8 * 0.154 / W, _p.y0, _p.width, _p.height])
    head(ax, "c", "when is a retention significant?")
    # Drawn with the same helper as (a), not with a text bbox. The two mechanisms use different
    # padding, so the old text sat a line higher inside its box than (a)'s did -- visible as a
    # mismatch between the columns rather than as anything wrong with either on its own.
    #
    # The box now carries the TEST rather than only the two ratios. Under the retired scheme the
    # panel's job was to name five classes, and the ratios plus "10% would be chance" was enough
    # to motivate them. The verdict is now a significance -- one rung clearing a molecule-blind
    # floor under BH -- and that is three separate decisions (which floor, which test, which
    # multiplicity correction), none of which the reader can guess. Every number here is a
    # property of the labels, so the panel stays a definition and does not become a result.
    LC = ["how many correct first figures survive",
          "to two, and how many of those to three?",
          "$R_{12} = m_2/m_1$      $R_{23} = m_3/m_2$",
          "each retention is tested one-sided against",
          "$\\mathrm{Binom}(m_k,\\ p_0)$, given the observed $m_k$",
          "$p_0$: the best a molecule-blind guesser",
          "can do from the labels of the benchmark",
          "one BH family per run"]
    hc = text_h(ax, LC) + 2 * PAD_H
    bc = boxed(ax, 0.97, LC, hc)
    # "-|>" spends ~0.06 in on the head alone, so an arrow much under 0.078 is head and no shaft.
    # zorder is raised on the PATCH: annotate() applies its own zorder to the Annotation instead.
    _a = ax.annotate("", xy=(0.42, bc - 0.01 - ARROW_L), xytext=(0.42, bc - 0.01),
                     arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK2))
    _a.arrow_patch.set_zorder(10)
    ax.text(0.01, bc - 0.01 - ARROW_L - 0.012, "the map colours each cell by this evidence:",
            fontsize=SCHEM_FS, va="top", fontweight="bold", linespacing=1.5)
    # This used to be a five-row class key -- heavy / partial / trace / clean / no signal -- and
    # those classes are gone. What replaces it is the map's own scale, $-\log_{10}q$ of the
    # better rung, white at the flagging threshold so a clean cell is white without a second
    # rule. The two blanks (`no-signal`, and the star for a cell carried by R12 alone) are NOT
    # repeated here: fig2 carries them in its own legend, and the failure this panel has already
    # had once is a key that disagreed with the map two pages later. One key, in one place.
    # The bold line above is one rendered line tall (~0.081 of this panel's height), so the ramp
    # has to clear bc - ARROW_L - 0.093 or it is drawn straight through the text introducing it.
    ys = bc - 0.01 - ARROW_L - 0.150
    ramp = ax.inset_axes([0.01, ys - 0.021, 0.44, 0.042], transform=ax.transAxes)
    ramp.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=SIG_CMAP, extent=(0, 1, 0, 1))
    ramp.set_xticks([]); ramp.set_yticks([])
    for s in ramp.spines.values():
        s.set_color("#c2c0bb"); s.set_linewidth(0.5)
    ax.text(0.475, ys, "significance of the", fontsize=SCHEM_FS, va="bottom", color=INK2)
    ax.text(0.475, ys - 0.004, "retentions above $p_0$", fontsize=SCHEM_FS, va="top", color=INK2)
    ax.text(0.01, ys - 0.030, "none", fontsize=7.0, va="top", color=INK2)
    ax.text(0.45, ys - 0.030, "strongest", fontsize=7.0, va="top", ha="right", color=INK2)
    save(fig, "fig1_protocol")


# =========================================================================== fig 2
def fig2(A, fname="fig2_map", smax=None):
    """The map, and what the categories in it mean.

    A is the controlled-budget arm (22 models x N benchmarks, the SAME 500 molecules in every
    cell). `fname` is a parameter because this used to be called a second time to draw the
    zero-shot screen as an appendix figure; that arm is retired and the second call is gone.
    The default-vs-budget comparison that used to sit in (b)/(c) has moved to the section that
    argues it (fig4), where it is drawn against the minimum-reasoning arm.

    (b) and (c) instead say what the map's colours ARE: the digit ladder the verdict is read
    off, and the two questions a cell answers -- can this model predict the benchmark at all,
    and does it reproduce the published digits.
    """
    use_serif()
    rows_n = A.tag.nunique()
    order = [d for d in DS_ORDER if d in set(A.dataset)]
    # The legend used to sit in a right-hand column, which cost the map a fifth of the text
    # width. Above the panel it costs one line of height instead, and the cells get that width.
    #
    # (b) and (c) are gone. The digit ladder is Fig.~1b and the class definitions are Fig.~1c, so
    # repeating them here made the map's own figure argue someone else's point twice. What is
    # left is one panel, which is what the figure is for -- and it no longer needs a panel letter.
    fig = plt.figure(figsize=(W, 1.35 + 0.200 * rows_n))
    gs = fig.add_gridspec(1, 1, left=0.005, right=0.995, top=0.905, bottom=0.055)

    tags = A.sort_values("ord").tag.unique().tolist()
    names = A.drop_duplicates("tag").set_index("tag").model.to_dict()

    def grid(col, agg="first"):
        return A.pivot_table(index="tag", columns="dataset", values=col,
                             aggfunc=agg).reindex(index=tags, columns=order)

    A = A.copy()
    A["sig"] = sig_strength(A)
    piv = grid("regime")
    star = grid("decided_at").to_numpy(float)
    hit3 = grid("hit3").to_numpy(float)
    rho = grid("spearman").to_numpy(float)
    m1rate = (grid("rate_m1").to_numpy(float) if "rate_m1" in A.columns
              else np.full(hit3.shape, np.nan))

    # Colour is the evidence, not the class. The classes still name the verdict in every table,
    # but a five-step palette said "partial" and "heavy" were two kinds of thing when the rule
    # separating them is a threshold on a continuous rate; the ramp shows the reader where each
    # cell actually sits, and leaves the clean ones white.
    SIG = grid("sig").to_numpy(float)
    smax = float(np.nanmax(SIG)) if smax is None else float(smax)
    U = sig_norm(SIG, smax)
    code = {"clean": 0, "trace": 1, "partial": 2, "heavy": 3}
    M = piv.map(lambda v: code.get(v, np.nan)).to_numpy(float)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-.5, len(order) - .5); ax.set_ylim(len(tags) - .5, -.5)
    reg = piv.to_numpy(object)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                       fc=SIG_CMAP(U[i, j]), ec="none"))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                if reg[i, j] == "no-signal":
                    # A negative, not a blank: flat fill, and the first-figure rate in the cell
                    # so the reader can see how far short the model fell.
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc=NOSIGNAL, ec="none"))
                    v = m1rate[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.1f}", fontsize=7.8, ha="center", va="center",
                                color=INK2, style="italic")
                else:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////",
                                           ec="#c2c0bb", lw=0.0, alpha=0.9))
                continue
            # The number IS the headline: what fraction of the benchmark this model reproduces
            # to three significant figures. On the two dark fills it has to be white to be read.
            # The number IS the headline. Rank skill used to be drawn as a bar in the same
            # cell and it made the panel unreadable; it now has panel (c) to itself.
            v = hit3[i, j]
            dark = U[i, j] >= 0.62
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                        fontsize=7.8, ha="center", va="center",
                        color="white" if dark else INK)
            if star[i, j] == 2:
                ax.plot(j + 0.40, i - 0.28, marker="*", ms=3.0,
                        mfc="white" if dark else INK, mec="none", zorder=5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([DS_LABEL[d] for d in order], rotation=45, ha="right", fontsize=7.8)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([names[t] for t in tags], fontsize=7.8)
    ax.set_xticks(np.arange(-.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(tags), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=0.8); ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for j, ds in enumerate(order):
        if ds in ("antiviral",) or (j and ds == "qm7"):
            ax.axvline(j - .5, color=INK2, lw=0.9)
    # The key is now a ramp plus the two kinds of blank. It is deliberately unnumbered: the
    # quantity behind it spans two hundred decades, so tick labels would invite a reader to
    # compare two cells by eye on a scale where that comparison is meaningless. What the colour
    # is good for is ordering, and 'none' to 'strongest here' says exactly that.
    present = set(piv.to_numpy(object).ravel())
    # The reference key. `sig_ramp` is the same call in fig4a and in the blinding map, so the
    # three cannot drift apart; the height it picks is measured off this axes rather than
    # derived from `rows_n`, which is what keeps the ramp 9.5 pt in the zero-shot copy too.
    sig_ramp(ax)

    handles = []
    if "no-signal" in present:
        handles.append(Patch(fc=NOSIGNAL, ec="#c2c0bb", lw=0.5,
                             label="no signal  first figure at chance"))
    if "untestable" in present:
        handles.append(Patch(fc=NODATA, ec="#c2c0bb", hatch="////",
                             label="no power  $m_1<15$: no verdict"))
    handles.append(Line2D([], [], marker="*", ls="", color=INK, ms=4,
                          label="decided on the $R_{12}$ rung"))
    ax.legend(handles=handles,
              loc="lower left", bbox_to_anchor=(0.60, 1.045), ncol=1, handlelength=1.1,
              labelspacing=0.32, columnspacing=1.5, fontsize=7.4, borderaxespad=0.0,
              frameon=False)

    save(fig, fname)
    return smax


ARM_ORDER = ["freesolv", "esol", "ld50", "lipophilicity", "qm8", "antiviral", "boilingpoint"]
# =========================================================================== fig 3

def fig3(A, M):
    """What the choice of null does to the verdicts.

    `A` is the MINIMUM-REASONING arm. That is deliberate and it is the only arm in which
    this panel says anything: the comparison is between the conventional global-shuffle
    null and the label-only floor, and they only diverge where a benchmark has few genuine
    flags. On ESOL the minimum arm flags 5 cells against the label floor and 15 against the
    global shuffle; the controlled map flags 15 and 19, where the two nulls nearly agree.
    """
    fig = plt.figure(figsize=(W, 2.30))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.40,
                          left=0.005, right=0.995, top=0.90, bottom=0.16)

    # (a) ESOL: every model against two floors ----------------------------
    e = A[A.dataset == "esol"].sort_values("hit3", ascending=False).copy()
    ax = fig.add_subplot(gs[0])
    panel(ax, "a", "ESOL: the floor decides the verdict")
    x = np.arange(len(e))
    # Two reference lines, because the two obvious analytic guesses are both wrong and it is
    # worth seeing by how much. Uniform digits put a 3-sig coincidence at 1/1000; the measured
    # global-shuffle floor sits 2-3x above it because leading digits cluster, which is exactly
    # why the literature shuffles rather than assuming 0.1%. Conditioning on the model's own
    # 1-sig rate and assuming two further uniform digits gives hit1/100; the measured
    # accuracy-matched floor sits above that too, because per-rung retention is ~17%, not 10%.
    ax.axhline(0.1, color=INK2, lw=0.7, ls=(0, (1, 2)), zorder=1)
    ax.text(len(e) - 0.4, 0.105, "uniform digits, $10^{-3}$", fontsize=5.6, color=INK2,
            ha="right", va="bottom")
    ax.plot(x, 100.0 * e.m1 / e.n_usable / 100.0, "-", lw=0.9, color="#b9bec4",
            label="$\mathsf{hit}_1/100$")
    ax.plot(x, e.chance_hit3, "-", lw=1.3, color=CAT[1], label="global-shuffle floor")
    ax.plot(x, e.mb_chance_hit3, "-", lw=1.3, color=CAT[2], label="accuracy-matched floor")
    # The floor the verdicts actually use, and the point of drawing it beside the other two is
    # that it is FLAT. It is a property of ESOL's label column, so it does not know which model
    # it is standing under; the accuracy-matched line rises under exactly the models it is
    # supposed to be a null for.
    LF = label_floor().get("esol")
    if LF:
        ax.axhline(LF["floor_hit3"], color=CAT[0], lw=1.3,
                   label="label-only floor (used here)")
    fp = (e.p_hit3 < 0.05) & (e.regime == "clean")
    ax.plot(x[~fp.to_numpy()], e.hit3[~fp], "o", ms=3.0, color=REGIME_COLOR["heavy"],
            label="measured 3-sig match")
    ax.plot(x[fp.to_numpy()], e.hit3[fp], "o", ms=4.4, mfc="none", mec=CAT[1], mew=1.1,
            label=f"{int(fp.sum())} false positives")
    ax.set_yscale("log"); ax.set_ylim(0.08, 120)
    ax.set_xlabel(f"{len(e)} models, ranked by measured rate", labelpad=2)
    ax.set_ylabel("3-sig match (%)", labelpad=2)
    ax.set_xticks([]); ax.legend(loc="upper right", handlelength=1.2, labelspacing=0.2)
    ax.grid(axis="x", visible=False); ax.tick_params(length=2)

    # (b) the four floors on the flagged cells ----------------------------
    cells = [("freesolv", "opus5"), ("freesolv", "grok45"), ("esol", "opus5"),
             ("esol", "gem31pro"), ("ld50", "gem31pro")]
    lab = ["FreeSolv\nOpus 5", "FreeSolv\nGrok 4.5", "ESOL\nOpus 5", "ESOL\nGemini 3.1 Pro",
           "LD50\nGemini 3.1 Pro"]
    idx = A.set_index(["dataset", "tag"])
    mi = M.set_index(["dataset", "tag"])
    ax = fig.add_subplot(gs[1])
    panel(ax, "b", "the same cells under four nulls")
    xs = np.arange(len(cells))
    ax.bar(xs, [idx.loc[k, "hit3"] for k in cells], width=0.62, color="#e7eefa",
           edgecolor=REGIME_COLOR["partial"], lw=0.7, zorder=1, label="measured")
    series = [("global shuffle", "chance_hit3", CAT[1]),
              ("accuracy-matched bins", "mb_chance_hit3", CAT[2]),
              ("own-residual resample", "rs_chance_hit3", CAT[3])]
    for name, col, c in series:
        ax.plot(xs, [idx.loc[k, col] for k in cells], "_", color=c, ms=13, mew=1.8,
                label=name, ls="none", zorder=4)
    ax.plot(xs, [mi.loc[k, "smooth_floor"] if k in mi.index else np.nan for k in cells],
            "_", color=INK, ms=13, mew=1.8, label="smooth error, same accuracy", zorder=4)
    LFa = label_floor()
    if LFa:
        ax.plot(xs, [LFa.get(k[0], {}).get("floor_hit3", np.nan) for k in cells],
                "_", color=CAT[0], ms=13, mew=2.2, label="label only (used)", zorder=5)
    ax.set_yscale("log"); ax.set_ylim(0.12, 2200)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=5.4, rotation=32, ha="right",
                                          rotation_mode="anchor")
    ax.set_ylabel("3-sig match (%)", labelpad=2)
    h, l = ax.get_legend_handles_labels()
    o = [l.index("measured")] + [i for i, x in enumerate(l) if x != "measured"]
    ax.legend([h[i] for i in o], [l[i] for i in o], loc="upper left", handlelength=0.9,
              labelspacing=0.18, fontsize=5.9, handletextpad=0.4)
    ax.grid(axis="x", visible=False); ax.tick_params(length=2)

    # (c) R23 against a simulated same-accuracy floor ----------------------
    ax = fig.add_subplot(gs[2])
    panel(ax, "c", "$R_{23}$ survives accuracy matching")
    # The regime column in the simulated-floor file is whatever the classifier said when that
    # file was written, so it is re-joined from the current verdicts rather than trusted.
    cur = A.set_index(["dataset", "tag"]).regime
    m = M[(M.dataset != "boilingpoint")].dropna(subset=["R23_excess_smooth"]).copy()
    m["regime"] = [cur.get((d, t), np.nan) for d, t in zip(m.dataset, m.tag)]
    m = m.dropna(subset=["regime"])
    rng = np.random.default_rng(0)
    counts = {}
    for i, (reg, c) in enumerate((("clean", "#b9bec4"), ("trace", REGIME_COLOR["trace"]),
                                  ("partial", REGIME_COLOR["partial"]),
                                  ("heavy", REGIME_COLOR["heavy"]))):
        d = m[m.regime == reg]
        counts[reg] = len(d)
        ax.scatter(i + rng.uniform(-0.19, 0.19, len(d)), d.R23_excess_smooth, s=9,
                   color=c, zorder=3, edgecolor="white", lw=0.25)
        ax.plot([i - 0.32, i + 0.32], [d.R23_excess_smooth.median()] * 2, "-",
                color=INK, lw=1.4, zorder=4)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{k}\n(n={counts.get(k, 0)})"
                        for k in ("clean", "trace", "partial", "heavy")], fontsize=6.2)
    ax.set_ylabel("$R_{23}$ / floor at the same accuracy", labelpad=2)
    ax.set_ylim(0, 7.8)
    ax.text(0.03, 0.955, "positive control excluded", transform=ax.transAxes,
            fontsize=6.0, color=INK2, va="top")
    ax.grid(axis="x", visible=False); ax.tick_params(length=2)
    save(fig, "fig3_nulls")


# =========================================================================== fig 4
def _fig4_paired_map(ax, A, tags, smax):
    """The map of Fig. 2, every cell split: left at the budget, right at the minimum.

    Only the benchmarks measured in BOTH arms appear, which is what makes the halves a paired
    comparison rather than two maps side by side. The three benchmarks the minimum arm never
    ran are therefore absent by construction, not dropped.
    """
    both = sorted(set(A[A.arm == "t1024"].dataset) & set(A[A.arm == "reg"].dataset),
                  key=lambda d: DS_ORDER.index(d))
    names = A.drop_duplicates("tag").set_index("tag").model.to_dict()
    A = A.copy()
    A["sig"] = sig_strength(A)

    def half(arm):
        return (A[A.arm == arm].pivot_table(index="tag", columns="dataset", values="sig")
                .reindex(index=tags, columns=both).to_numpy(float))

    hi, lo = half("t1024"), half("reg")
    if smax is None:
        smax = float(np.nanmax(hi))
    for i in range(len(tags)):
        for j in range(len(both)):
            for x0, v in ((j - .5, hi[i, j]), (j, lo[i, j])):
                fc = NODATA if not np.isfinite(v) else SIG_CMAP(sig_norm(v, smax))
                ax.add_patch(Rectangle((x0, i - .5), .5, 1, fc=fc, ec="none"))
    ax.set_xlim(-.5, len(both) - .5); ax.set_ylim(len(tags) - .5, -.5)
    ax.set_xticks(range(len(both)))
    ax.set_xticklabels([DS_SHORT[d] for d in both], rotation=45, ha="right", fontsize=7.0)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([names[t] for t in tags], fontsize=6.6)
    ax.set_xticks(np.arange(-.5, len(both), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(tags), 1), minor=True)
    # Grey, not white: this panel carries no numbers, so with a white rule a run of clean cells
    # would have no visible boundaries at all and the reader could not tell nine benchmarks from
    # one wide gap.
    ax.grid(which="minor", color="#e4e2dd", lw=0.7); ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    panel(ax, "a", None)
    # THE KEY IS LAID OUT AS FIG. 2'S, and that is the whole reason it looks like this. Every map
    # in the paper now puts the ramp at the top left with its caption above it and `none` /
    # `strongest` below it, and whatever else the panel has to say on the right of the same
    # block. A reader who has learned to read one key has read them all; a ramp that migrates
    # from above-left to inline-right between two maps of the same quantity is a different key
    # as far as anyone skimming is concerned.
    #
    sig_ramp(ax, width=0.30)
    # Right of the ramp, where fig2 puts its `no signal` legend: what this panel adds to the map.
    ax.text(0.40, 1.035, "left half: 1,024-token budget", transform=ax.transAxes,
            fontsize=7.0, color=INK2, ha="left", va="bottom")
    ax.text(0.72, 1.035, "right half: the endpoint's own minimum", transform=ax.transAxes,
            fontsize=7.0, color=INK2, ha="left", va="bottom")


def fig4(L, A=None, smax=None):
    """What a reasoning budget does to retrieval.

    (a) is the paired arm drawn as the map of Fig. 2, every cell split down the middle: left half
    at the controlled budget, right half at the endpoint's own minimum. It replaces a scatter of
    the same 198 cells against the diagonal, which showed that the budget moves cells but not
    which cells, and it doubles as the statement of which models and benchmarks the paired arm
    covers. Stars are dropped: at this size the half-cells are the information.

    (b--d) are the four-model ladder, which resolves the dose axis the paired arm only has two
    points on.
    """
    fig = plt.figure(figsize=(W, 3.60))
    tags = A.sort_values("ord").tag.unique().tolist() if A is not None else []
    map_h = 0.150 * len(tags) + 0.55
    fig.set_size_inches(W, 2.05 + map_h)
    gs = fig.add_gridspec(2, 3, height_ratios=[map_h, 1.75], hspace=0.42, wspace=0.46,
                          left=0.005, right=0.995, top=0.965, bottom=0.10)
    if A is not None:
        _fig4_paired_map(fig.add_subplot(gs[0, :]), A, tags, smax)
    L = L.copy()
    L["x"] = np.maximum(L.reasoning_med, 30.0)
    # The split is by what the map found in the benchmark, not by chemistry: the panel's claim is
    # that a budget moves the digits only where there are digits to move, so the grouping has to
    # be the map's verdict.
    #
    # QM7 sat with the clean set on the argument that 10 of its 22 map cells have no power, so
    # the detector cannot read it either way. That reasoning belonged to the accuracy-matched
    # floor, under which QM7's excess stayed near 1.5x and its climb looked like the floor
    # rising with the rate. Under the molecule-blind floor the floor is fixed at 1.44% and the
    # climb is a real excess -- 19% at the top of the GPT-5.5 ladder, 13.2x the floor, with both
    # rungs far above theirs -- and the map flags 3 of the 22 cells. A benchmark with three
    # flagged cells and a dose-response belongs with the ones that have something stored.
    groups = [("something stored", STORED),
              ("nothing stored", ["qm8", "lipophilicity", "bace", "caco2", "ppbr"]),
              ("controls", ["boilingpoint", "antiviral"])]
    dsc = {"ld50": CAT[1], "esol": CAT[0], "freesolv": CAT[3], "aqsoldb": CAT[2],
           "qm8": CAT[2], "lipophilicity": CAT[0], "bace": CAT[1], "caco2": CAT[3],
           "ppbr": "#7a7f87", "qm7": "#c9a227",
           "boilingpoint": "#c9a227", "antiviral": CAT[2]}
    for k, (title, dss) in enumerate(groups):
        ax = fig.add_subplot(gs[1, k])
        panel(ax, "bcd"[k], title)
        # A benchmark with no ladder cells must not reach the legend, or the panel advertises a
        # colour that appears nowhere in it.
        dss = [d for d in dss if (L.dataset == d).any()]
        opus_ends = []
        for ds in dss:
            for tag, d in L[L.dataset == ds].groupby("tag"):
                d = d.sort_values("x")
                # Opus 5 is drawn heavier because its line is the panel's sharpest statement and
                # is otherwise unattributable: colour encodes the benchmark, so nothing tells the
                # reader that the two lines starting above 35% at the left edge are one model
                # that reproduced the benchmark before spending a token.
                heavy = tag == "opus5"
                ax.plot(d.x, d.hit3, "-o", ms=2.4 if heavy else 2.0, lw=1.9 if heavy else 1.0,
                        color=dsc[ds], alpha=1.0 if heavy else 0.8,
                        zorder=4 if heavy else 3)
                if heavy:
                    opus_ends.append((float(d.x.iloc[-1]), float(d.hit3.iloc[-1])))
        ax.set_xscale("log"); ax.set_xlim(25, 11000); ax.set_ylim(-3, 95)
        ax.set_xlabel("reasoning tokens emitted", labelpad=2)
        if k == 0:
            ax.set_ylabel("3-sig match (%)", labelpad=2)
        # Opus 5 is a model and the legend is a list of benchmarks, so it cannot go in there
        # without reading as a thirteenth dataset. It is called out in (b) only, with leader
        # lines to its four ladders: (b) is where the heavy line carries an argument -- the same
        # model reciting two benchmarks at 65% before it spends a token and climbing from
        # nothing on the other two. In (c) and (d) the heavy line is drawn but left unlabelled,
        # since nothing there turns on which model it is.
        if k == 0 and opus_ends:
            tx, ty = 2.4e2, 49.6          # right of the Opus 5 ladders, which all end by x=126
            # The label sits over the LD50 fan and the leaders cross it, so both get a white
            # backing: without it a leader reads as one more ladder and the text as data.
            halo = [pe.withStroke(linewidth=1.6, foreground=SURFACE)]
            for ex, ey in opus_ends:
                ax.annotate("", xy=(ex, ey), xytext=(tx, ty), zorder=6,
                            arrowprops=dict(arrowstyle="-", color=INK, lw=0.5,
                                            shrinkA=2.5, shrinkB=2.5,
                                            path_effects=halo))
            ax.text(tx * 1.06, ty, "Claude Opus 5", fontsize=6.0, color=INK,
                    ha="left", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none", alpha=0.88))
        h = [Line2D([], [], color=dsc[d], marker="o", ms=3, label=DS_SHORT[d]) for d in dss]
        ax.legend(handles=h, loc="upper left", handlelength=1.0, labelspacing=0.16,
                  ncol=2 if len(h) > 3 else 1, columnspacing=0.9, fontsize=5.8,
                  bbox_to_anchor=(0.0, 1.02) if k != 2 else (0.0, 0.86))
        ax.tick_params(length=2)

    save(fig, "fig4_ladder")


# =========================================================================== fig 5
def fig5(B, R, L):
    """Structure substitution, which is the one intervention the main text now keeps.

    The other two -- rewriting the SMILES and turning the reasoning budget down -- do not remove
    retrieval, so they are reported in the appendix and their three-way comparison moved with
    them (figS_interventions below). What is left here is the intervention that works and the
    evidence that it works BECAUSE it removes retrieval rather than because it makes the task
    harder.
    """
    from scipy.stats import spearmanr
    fig = plt.figure(figsize=(W, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.0], wspace=0.30,
                          left=0.005, right=0.995, top=0.89, bottom=0.17)
    dsc = {"esol": CAT[0], "freesolv": CAT[1], "ld50": CAT[2]}

    # (a) paired collapse --------------------------------------------------
    ax = fig.add_subplot(gs[0])
    panel(ax, "a", "structure substitution")
    for _, r in B.iterrows():
        ax.plot([0, 1], [r.hit3_L1, r.hit3_L5], "-o", ms=2.6, lw=1.1,
                color=dsc.get(r.dataset, "#888"), alpha=0.9)
    ax.set_xlim(-0.25, 1.25); ax.set_xticks([0, 1])
    ax.set_xticklabels(["published\nSMILES", "substituted\nstring"], fontsize=6.4)
    ax.set_ylabel("3-sig match (%)", labelpad=2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4, color=dsc[d],
                              label=DS_SHORT[d]) for d in dsc],
              loc="center right", handletextpad=0.2, labelspacing=0.18,
              bbox_to_anchor=(1.02, 0.62))
    ax.grid(axis="x", visible=False); ax.tick_params(length=2)

    # (b) dose-response ----------------------------------------------------
    ax = fig.add_subplot(gs[1])
    panel(ax, "b", "the drop scales with the recall")
    # Predictor: the recall measured INSIDE this experiment, not the map's. The map-based
    # predictor needs the model to be in the map's panel, and Kimi K2.6 -- the only cell in the
    # study with nothing memorised, and so the one that anchors the low end -- is not. Against
    # the in-run rate the fit uses every cell of this experiment and does not depend on a panel
    # decision made elsewhere. Both versions are in the appendix.
    d = B.dropna(subset=["hit3_L1"])
    rho, p = spearmanr(d.hit3_L1, d.ratio)
    ax.scatter(d.hit3_L1.clip(lower=0.15), d.ratio, s=22, zorder=3, edgecolor="white", lw=0.4,
               color=[dsc.get(x, "#888") for x in d.dataset])
    lowest = d.nsmallest(1, "hit3_L1")
    if len(lowest):
        ax.scatter(lowest.hit3_L1.clip(lower=0.15), lowest.ratio, s=70, facecolor="none",
                   edgecolor=INK, lw=0.9, zorder=4)
        ax.annotate("least memorised cell\nof this experiment",
                    (float(lowest.hit3_L1.clip(lower=0.15).iloc[0]), float(lowest.ratio.iloc[0])),
                    textcoords="offset points", xytext=(11, 16), fontsize=6.1, color=INK2,
                    linespacing=1.3, arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2))
    ax.set_xscale("log")
    ax.axhline(1.0, color=INK2, lw=0.7, ls="--")
    ax.set_xlabel("3-sig match with the published SMILES (%)", labelpad=2)
    ax.set_ylabel("median error, $\\times$ worse", labelpad=2)
    ax.set_title(f"the drop scales with the recall\n$\\rho={rho:+.2f}$, $p={p:.3f}$, $n={len(d)}$",
                 fontsize=7.2)
    ax.tick_params(length=2)

    save(fig, "fig5_mitigation")


def figS_interventions(B, R, L):
    """The three-way comparison, moved out of the main text with the two failed interventions.

    Same panel as fig5's old (c). It stays in the paper because it is the only place a reader
    can see how far short the other two fall; it is in the appendix because a bar at 102% and a
    bar at 4%-that-comes-back are not results the main argument rests on.
    """
    fig = plt.figure(figsize=(W * 0.62, 1.95))
    gs = fig.add_gridspec(1, 1, left=0.02, right=0.99, top=0.94, bottom=0.22)
    ax = fig.add_subplot(gs[0])
    MIN = 5.0                                    # a cell needs recall before it can lose it
    b = B[B.hit3_L1 >= MIN]
    subst = (100 * b.hit3_L5 / b.hit3_L1).to_numpy()
    Lg = L[L.dataset.isin(["ld50", "esol", "freesolv"])]
    peak = Lg.groupby(["dataset", "tag"]).hit3.max()
    zero = Lg.sort_values("reasoning_med").groupby(["dataset", "tag"]).hit3.first()
    ok = peak[peak >= MIN].index
    supp = (100 * zero[ok] / peak[ok]).to_numpy()
    r = R[R.hit3_can >= MIN]
    rand = (100 * r.hit3_rand / r.hit3_can).to_numpy()
    rows = [("substitute the structure string", subst, CAT[2]),
            ("suppress reasoning", supp, "#c9a227"),
            ("randomise the SMILES", rand, CAT[1])]
    rng = np.random.default_rng(1)
    for i, (lab, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, np.median(v), height=0.46, color=c, zorder=2, alpha=0.85)
        ax.scatter(v, y + rng.uniform(-0.14, 0.14, len(v)), s=8, color=INK, zorder=4,
                   alpha=0.75, lw=0)
        ax.text(1.5, y + 0.33, f"{lab}  ($n={len(v)}$)", fontsize=6.4, va="bottom", color=INK)
    ax.set_yticks([]); ax.set_ylim(-0.5, len(rows) - 0.18)
    ax.set_xlim(0, 105); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("verbatim recall that survives (%)", labelpad=2)
    ax.text(np.median(supp) + 5, len(rows) - 2, "hidden, not removed", fontsize=6.0,
            va="center", color="#8a6d10")
    ax.grid(axis="y", visible=False); ax.tick_params(length=2)
    save(fig, "figS_interventions")


def fig6(G):
    """Does recall buy accuracy? One column per recalled benchmark.

    (top) rank skill against verbatim recall, one point per model. The question the panel
    answers is not "are they correlated" -- they are -- but whether contamination is SUFFICIENT
    for a good score, and it is not: on FreeSolv two thirds of the contaminated cells predict
    worse than the best clean one.

    (bottom) the same models scored only on the molecules they did NOT reproduce, i.e. what the
    benchmark would have said if the recalled rows had never been in it.
    """
    keys = [k for k in ("freesolv", "esol", "ld50") if k in set(G.dataset)]
    fig = plt.figure(figsize=(W, 5.55))
    gs = fig.add_gridspec(2, len(keys), height_ratios=[1.0, 2.05], hspace=0.50, wspace=0.30,
                          left=0.005, right=0.995, top=0.94, bottom=0.07)
    FL = ("heavy", "partial", "trace")

    # (a-c) recall against rank skill, one panel per benchmark ---------------
    for j, dk in enumerate(keys):
        g = G[G.dataset == dk]
        ax = fig.add_subplot(gs[0, j])
        rs = g[["hit3", "rho"]].corr(method="spearman").iloc[0, 1]
        panel(ax, "abc"[j], f"{DS_SHORT[dk]}   $\\rho_s$ = {rs:+.2f}")
        clean = g[g.regime == "clean"]
        best = clean.rho.max() if len(clean) else np.nan
        if np.isfinite(best):
            ax.axhline(best, color=CAT[1], lw=0.9, ls=(0, (3, 2)), zorder=3)
        for reg in ("clean", "trace", "partial", "heavy"):
            d = g[g.regime == reg]
            ax.scatter(d.hit3.clip(lower=0.15), d.rho, s=17,
                       color=REGIME_COLOR.get(reg, "#b9c6d6"), edgecolor="white", lw=0.4,
                       zorder=4, label=reg if j == 0 else None)
        if np.isfinite(best):
            n_worse = int((g[g.regime.isin(FL)].rho < best).sum())
            ax.text(0.97, 0.06, f"{n_worse} of {int(g.regime.isin(FL).sum())} contaminated\n"
                                f"below the best clean model", transform=ax.transAxes,
                    fontsize=5.9, color="#a8441f", va="bottom", ha="right", linespacing=1.35)
        ax.set_xscale("log")
        ax.set_xlabel("verbatim recall (%)", labelpad=2)
        ax.set_ylim(min(0.0, g.rho.min() - 0.05), 1.04)
        if j == 0:
            ax.set_ylabel("rank skill, Spearman $\\rho$", labelpad=2)
            # One shared legend under the row: four categories x three panels is the same
            # legend three times, and inside the axes it lands on the annotation.
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.52, 0.665), ncol=4,
                       fontsize=6.6, handletextpad=0.25, columnspacing=1.6, scatterpoints=1)
        ax.tick_params(length=2)

    # (d) one leaderboard, in full, with and without the recalled molecules --
    dk = "freesolv" if "freesolv" in keys else keys[0]
    g = G[G.dataset == dk].sort_values("mae_clip").reset_index(drop=True)
    rank_res = {t: i for i, t in enumerate(g.sort_values("mae_clip_res").tag)}
    ax = fig.add_subplot(gs[1, :])
    panel(ax, "d", f"{DS_SHORT[dk]}: the leaderboard, and the leaderboard without "
                   f"each model's own recalled molecules")
    y = np.arange(len(g))
    for i, r in g.iterrows():
        c = REGIME_COLOR.get(r.regime, "#b9c6d6")
        ax.plot([r.mae_clip, r.mae_clip_res], [i, i], "-", color=c, lw=1.6, zorder=3,
                solid_capstyle="round")
        ax.plot(r.mae_clip, i, "o", ms=5.0, mfc=c, mec="white", mew=0.6, zorder=5)
        ax.plot(r.mae_clip_res, i, "o", ms=5.0, mfc="white", mec=c, mew=1.2, zorder=4)
        moved = i - rank_res[r.tag]
        if abs(moved) >= 2:
            ax.text(r.mae_clip_res + 0.10, i, f"{'+' if moved > 0 else ''}{moved}",
                    fontsize=5.8, va="center", color="#a8441f", fontweight="bold")
    clean = g[g.regime == "clean"]
    if len(clean):
        b = clean.mae_clip.min()
        ax.axvline(b, color=CAT[1], lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.text(b + 0.04, len(g) - 0.4, "best clean model", fontsize=5.9, color="#a8441f",
                va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}" for m in g.model], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_ylim(len(g) - 0.4, -0.6)
    ax.set_xlabel("MAE (kcal/mol)" if dk == "freesolv" else "MAE (log units)", labelpad=2)
    ax.grid(axis="y", visible=False); ax.tick_params(length=2)
    top = g.iloc[0]
    ax.text(0.985, 0.04,
            f"{top.model}: {top.mae_clip:.2f} $\\to$ {top.mae_clip_res:.2f} kcal/mol.\n"
            f"The best clean model scores {clean.mae_clip.min():.2f} with everything included.",
            transform=ax.transAxes, fontsize=6.2, ha="right", va="bottom", color="#a8441f",
            linespacing=1.35)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=REGIME_COLOR["heavy"], mec="white",
                              ms=5, label="all molecules"),
                       Line2D([], [], marker="o", ls="", mfc="white",
                              mec=REGIME_COLOR["heavy"], mew=1.2, ms=5,
                              label="recalled molecules removed"),
                       Line2D([], [], ls="", marker="$+2$", color="#a8441f", ms=7,
                              label="places lost when they are")],
              loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=6.2, handletextpad=0.4,
              labelspacing=0.3)
    save(fig, "fig6_generalization")


def fig7(D):
    """What a thinking budget buys, paired: same molecules, same prompt, one setting changed.

    This replaces the default-vs-budget panels that used to sit in fig2. Those compared two
    studies with different prompts and different molecule samples, so only the VERDICTS were
    commensurable. Here the rates are.
    """
    fig = plt.figure(figsize=(W, 2.42))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.12], wspace=0.42,
                          left=0.005, right=0.995, top=0.90, bottom=0.17)
    # Driven by the data, not by a literal. The paired arm was extended to AqSolDB and QM7 --
    # the two contaminated benchmarks it was missing, and the two that move most under a budget
    # -- and a hardcoded seven silently dropped them from panels (b) and (c) while the text
    # quoted nine. DS_ORDER is the panel order used everywhere else in this file.
    order = [d for d in DS_ORDER if d in set(D.dataset)]

    # (a) every cell, twice ------------------------------------------------
    ax = fig.add_subplot(gs[0])
    panel(ax, "a", "every cell, measured twice")
    eps = 0.04
    ax.plot([eps, 130], [eps, 130], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    d0, d1 = D[~D.crossed], D[D.crossed]
    ax.scatter(d0.hit3_lo + eps, d0.hit3_hi + eps, s=9, color="#b9c6d6", edgecolor="none",
               zorder=3)
    ax.scatter(d1.hit3_lo + eps, d1.hit3_hi + eps, s=14, color=REGIME_COLOR["heavy"],
               edgecolor="white", lw=0.3, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(eps * 0.8, 160); ax.set_ylim(eps * 0.8, 160)
    ax.set_xlabel("recall at the endpoint minimum (%)", labelpad=2)
    ax.set_ylabel("with 1,024 reasoning tokens (%)", labelpad=2)
    ax.text(0.06, 110, f"{int(D.crossed.sum())} cells cross\nfrom clean to flagged", fontsize=6.2,
            color=REGIME_COLOR["heavy"], va="top", linespacing=1.35)
    ax.tick_params(length=2)

    # (b) flagged share per benchmark, both settings ------------------------
    ax = fig.add_subplot(gs[1])
    panel(ax, "b", "and where they are")
    y = np.arange(len(order))
    lo = [100 * D[D.dataset == d].flag_lo.mean() for d in order]
    hi = [100 * D[D.dataset == d].flag_hi.mean() for d in order]
    ax.barh(y + 0.19, lo, height=0.34, color="#b9c6d6", label="endpoint minimum")
    ax.barh(y - 0.19, hi, height=0.34, color=REGIME_COLOR["partial"], label="1,024 tokens")
    ax.set_yticks(y); ax.set_yticklabels([DS_SHORT[d] for d in order], fontsize=7.0)
    ax.invert_yaxis()
    ax.set_xlabel("cells flagged (%)", labelpad=2)
    ax.set_xlim(0, 104)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55), fontsize=6.2, handlelength=1.0,
              labelspacing=0.25, framealpha=0.95)
    ax.grid(axis="y", visible=False); ax.tick_params(length=2)

    # (c) accuracy everywhere, digits only where something is stored --------
    ax = fig.add_subplot(gs[2])
    panel(ax, "c", "accuracy everywhere, digits not")
    for i, dk in enumerate(order):
        g = D[D.dataset == dk]
        if dk == "boilingpoint":
            continue
        dm = 100 * (1 - g.medae_hi.median() / max(g.medae_lo.median(), 1e-9))
        dh = g.hit3_hi.median() - g.hit3_lo.median()
        recalled = dk in STORED
        c = REGIME_COLOR["heavy"] if recalled else CAT[2]
        ax.scatter(dm, dh, s=26, color=c, edgecolor="white", lw=0.5, zorder=4)
        ax.annotate(DS_SHORT[dk], (dm, dh), textcoords="offset points",
                    xytext=(5, 3 if recalled else -9), fontsize=6.0, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("median error removed by the budget (%)", labelpad=2)
    ax.set_ylabel("change in verbatim recall (pp)", labelpad=2)
    ax.set_xlim(-4, 42)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=REGIME_COLOR["heavy"], ms=4,
                              label="something stored"),
                       Line2D([], [], marker="o", ls="", color=CAT[2], ms=4,
                              label="nothing stored")],
              loc="upper left", handletextpad=0.2, labelspacing=0.2, fontsize=6.2)
    ax.tick_params(length=2)
    save(fig, "fig7_budget_delta")


def load_arms():
    """The classified budget file with BOTH arms kept, for the paired half-cell map."""
    d = drop_extras(_classified("budget_3sig", fallback="budget_3sig_pad_v2.csv"))
    order = [m["tag"] for m in json.load(open(os.path.join(REG, "models.json")))["models"]]
    d["ord"] = d.tag.map({t: i for i, t in enumerate(order)})
    return d


def main():
    L, B, R, M, A, models = load()
    ARMS = load_arms()
    print("figures ->", FIG)
    fig0(A, L, B)
    fig1(A)
    smax = fig2(A)
    fig3(ARMS[ARMS.arm == "reg"], M[M.arm == "reg"] if "arm" in M.columns else M)
    fig4(L, ARMS, smax)
    fig5(B, R, L)
    figS_interventions(B, R, L)
    dp = os.path.join(RES, "reasoning_delta.csv")
    if os.path.exists(dp):
        fig7(pd.read_csv(dp))
    gp = os.path.join(RES, "generalization.csv")
    if os.path.exists(gp):
        fig6(pd.read_csv(gp))
    else:
        print("  (no generalization.csv -- run src/analyze_generalization.py)")


if __name__ == "__main__":
    main()
