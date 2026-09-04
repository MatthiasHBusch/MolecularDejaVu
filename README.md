# Molecular Déjà Vu: Digit-Level Recall of Published Values in Frontier Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Content

Frontier language models are asked, zero-shot, for the published value of a molecular property
for a named benchmark molecule — the real dataset name, the real SMILES, no in-context examples.
A prediction is an *n*-sig **hit** if it rounds to the same *n* significant figures as the
published value. The statistic per (model, benchmark) cell is the **retention ladder**
`R12 = m2/m1`, `R23 = m3/m2` — of the predictions already matching at *n* digits, the fraction
that also match at *n*+1 — with `hit3` reporting the size of the effect. Coincidence loses roughly
a factor of ten per rung; recall does not.

The retentions are tested against a **label-only floor**: the best that any procedure could reach
which sees no molecule-specific information at all, only the benchmark's own label distribution
(`src/label_floor.py`, `results/label_floor.csv`). No model, accuracy or outcome enters it, so it
cannot rise with the effect it bounds — which a null built from the model's own predictions does,
because at the digit level a reproduced value cannot be distinguished from an accurate one. Each
retention is an exact one-sided binomial conditioned on its own denominator, Benjamini–Hochberg
corrected within each run. The full treatment — the floor, the constructions it
replaced and why each fails, power, multiplicity, every control and every limitation — is in the
appendix of the paper.

The headline run is the **controlled map**: 22 models × 12 benchmarks × the same 500 molecules at
a 1,024-token reasoning limit, 264 cells, of which 89 are flagged, 164 clean and 11 no-signal
(`results/budget_3sig_v3.csv`, arm `t1024`). A paired minimum-reasoning arm measures the same
models at the lowest setting each endpoint permits, and the two are never pooled.

Headline findings:

- **Retrieval is concentrated on the most widely redistributed benchmarks.** LD50 is flagged in
  16 of 22 cells, ESOL and AqSolDB in 15, FreeSolv in 14; BACE, Caco-2 and the recency control are
  clean across the panel. Claude Opus 5 reproduces 62.1 % of ESOL to three significant figures.
- **Retrieval is gated by the reasoning level.**
  The same models on the same molecules go from 47 flagged cells at minimum reasoning to 89 at the
  1,024-token limit, so every contamination audit is a lower bound at the setting it was run at.
- **It is not chronology, it is redistribution.** How often a benchmark's molecules appear across
  three open pretraining indexes tracks its strongest retrieval rate (ρ = 0.88); how often the
  benchmark file's own column headers appear does not (ρ = 0.20).
- **Blinding interrupts it, and reorders the leaderboard.** Substituting the structure string
  character by character clears retrieval in most cells and draws the models' errors together;
  the model that leads all three benchmarks unblinded leads none of them blinded.

## Install

```bash
git clone <this repository>
cd <this repository>
pip install -r requirements.txt
```

Python 3.9+. `numpy`, `pandas`, `scipy` and `matplotlib` are all the analysis and the figures
need. `rdkit` is required only by `src/prepare_datasets.py`, which rebuilds `data/screening/`
from the primary sources and is not needed to reproduce anything in the paper.

Julia 1.9+ and an OpenRouter API key are needed **only** to re-run the query campaigns. Nothing
in the reproduction path below touches an API.

## Reproducing the figures

Everything the paper reports is derivable from the files already in `results/`. From the
repository root:

```bash
python src/figures_paper.py
```

writes `fig0_abstract`, `fig1_protocol`, `fig2_map`, `fig3_nulls`, `fig4_ladder`,
`fig5_mitigation`, `fig6_generalization`, `fig7_budget_delta` and `figS_map_zeroshot`
(PDF + 300 dpi PNG) into `paper/figures/`.

| figure | reads |
|---|---|
| fig1 (the protocol) | `results/budget_3sig_v3.csv` |
| fig2 (the map) | `results/budget_3sig_v3.csv` (arm `t1024`), `src/registry/models.json` |
| fig3 (the nulls) | `results/budget_3sig_v3.csv`, `results/smooth_error_null_budget.csv` |
| fig4 (the ladder) | `results/ladder_summary.csv`, `results/budget_3sig_v3.csv` |
| fig5 (mitigation) | `results/blinding_sweep.csv`, `results/randomization_control.csv`, `results/ladder_summary.csv` |
| fig6 (generalization) | `results/generalization.csv` |
| fig7 (budget delta) | `results/reasoning_delta.csv` |

Appendix tables:

```bash
python src/export_appendix_tables.py     # -> paper/tables/*.tex
```

## Regenerating the derived CSVs from the raw per-cell JSON

The raw model replies are the primary data and they are all here. To rebuild the verdict tables
from them rather than trusting the shipped copies (still no API access needed):

```bash
python src/analyze_budget.py --pad              # results/budget/*.json  -> budget_3sig_pad.csv
python src/label_floor.py                       # -> label_floor.csv, the floor the verdicts use
python src/smooth_error_null.py --arm budget    # superseded same-accuracy floors, kept as a comparator
python src/classify.py                          # -> *_v3.csv, the current verdicts
python src/analyze_ladder.py --perm 2000        # results/ladder/*.json -> ladder_summary.csv
python src/analyze_reasoning_delta.py           # -> reasoning_delta.csv
python src/analyze_generalization.py            # -> generalization.csv
python src/analyze_blinding_sweep.py            # results/blinding/*.json -> blinding_sweep.csv
python src/analyze_blinding_map.py              # the same cells on the digit rungs -> blinding_map.csv
python src/first_figure_reference.py            # -> first_figure_reference.csv
python src/detection_limits.py                  # -> detection_limits.csv, the power accounting
python src/corpus_prevalence.py --sample 60     # infini-gram; network, but free
```

`label_floor.py` must run before `classify.py`: the verdicts are read against the floor it
writes. Permutation nulls are the slow part; `--perm 2000` is what the paper used.

## Directory layout

```
data/
  delaney-processed.csv, delaney-randomized.csv, Lipophilicity.csv, qm7.csv
                              primary sources for the benchmarks distributed as files
  antiviral_potency.csv       post-cutoff negative control (ASAP/Polaris/OpenADMET 2025)
  known_boiling_points.csv    positive control (values the models demonstrably know)
  screening/<key>.csv         the 13 prepared benchmarks the analysis reads:
                              mol_id, smiles, value, smiles_random, smiles_blind, value_blind
                              (+ value_affine/nonmono/sine for esol and freesolv)
src/
  figures_paper.py            every figure in the paper
  memodetect.py               the shared detector: digit statistics, the nulls, BH, regimes
  label_floor.py              the label-only floor the verdicts are tested against
  classify.py                 the current verdict scheme -> results/*_v3.csv
  detection_limits.py         power and the bounds behind every clean outcome
  first_figure_reference.py   what a first-figure-only predictor correlates at
  analyze_*.py                one analysis per file; see the table below
  run_*.py, Run_*.jl          the query runners (API access required)
  registry/datasets.json      13 benchmarks (12 in the panel; QM9 was stopped on cost)
                              -- adding one is a data change, not a code change
  registry/models.json        31 models with release date, price and reasoning setting
                              (22 are in the panel; the rest do not honour a reasoning limit)
  lib/                        the shared Julia LLM helper library; LLMs.jl reads the API key
                              from $OPENROUTER_API_KEY
results/
  budget/*.json               the controlled-budget arm, 686 blocks -- the headline data
  ladder/*.json               the controlled reasoning ladder, 257 cells
  blinding/*.json             the L1/L5 blinding arm (`*__t1024.json` is the released one)
  convergence/*.json          the in-context convergence probe
  traces/                     reasoning traces for switcher / non-switcher molecules
  meta/                       reasoning calibration, flex-endpoint support, cost probes
  *.csv                       the derived tables; the ones the figures read are listed above
paper/figures/                output directory for src/figures_paper.py
```

### What each script reads and writes

| script | reads | writes |
|---|---|---|
| `prepare_datasets.py` | registry + sources in `data/` | `data/screening/<key>.csv` |
| `calibrate_reasoning.py` | registry | reasoning setting → registry, `results/meta/reasoning_calibration.json` |
| `run_matrix.jl` → `Run_ZeroShot.jl` | registry, screening CSVs | `results/zs/…json`, `results/usage_log.csv` |
| `Run_Convergence.jl` | screening CSVs | `results/convergence/…json` |
| `run_reasoning_ladder.py` | registry, screening CSVs | `results/ladder/<ds>__<tag>__<level>.json` |
| `run_budget_sweep.py` | registry, screening CSVs | `results/budget/…json` |
| `run_blinding_l1_l5.py` | screening CSVs | `results/blinding/…json` |
| `memodetect.py` | — | the shared detector (imported, not run) |
| `analyze_budget.py` | `results/budget/` | `results/budget_3sig_pad.csv` |
| `smooth_error_null.py` | either arm's cell table | `smooth_error_null{,_budget}.csv` |
| `label_floor.py` | the screening CSVs' label columns | `results/label_floor.csv` — the floor the verdicts use |
| `classify.py` | the cell tables + `label_floor.csv` | `results/*_v3.csv` — **the current verdicts** |
| `detection_limits.py` | `budget_3sig_v3.csv` | `results/detection_limits.csv` — power, bounds on clean cells |
| `first_figure_reference.py` | `results/budget/`, `budget_3sig_v3.csv` | `results/first_figure_reference{,_summary}.csv` |
| `reclassify.py` | same | `results/*_v2.csv` — the scheme `classify.py` replaced, kept for provenance |
| `analyze_ladder.py` | `results/ladder/` | `results/ladder_summary.csv` |
| `analyze_ladder_mechanism.py` | ladder, screening, corpus | `results/ladder_mechanism.csv` |
| `analyze_reasoning_delta.py` | `budget_3sig_v3.csv` (both arms) | `results/reasoning_delta.csv` |
| `analyze_generalization.py` | `results/budget/`, `budget_3sig_v3.csv` | `results/generalization.csv` |
| `analyze_blinding_sweep.py` | `results/blinding/` | `results/blinding_sweep{,_t1024}.csv` |
| `analyze_blinding_map.py` | `results/blinding/` | `results/blinding_map.csv` — the same cells on the digit rungs |
| `audit_l5_prompts.py` | screening CSVs | prints the L5 prompt audit; writes nothing |
| `reblind_datasets.py` | screening CSVs | regenerates only `smiles_blind` (BLIND_MAP v2) |
| `sweep_thresholds.py` | `budget_3sig_v3.csv` | `paper/tables/thresholds.tex` — the constant sweep |
| `analyze_convergence.py` | `results/convergence/` | `results/convergence_<ds>.csv` |
| `corpus_prevalence.py` | screening CSVs | `results/corpus_prevalence.csv` |
| `estimate_cost.py` | `usage_log.csv` | cost projections from measured spend |
| `figures_paper.py` | the analysis CSVs | `paper/figures/fig*` |
| `export_appendix_tables.py` | the analysis CSVs | `paper/tables/*.tex` |

## Re-running the query campaigns

Only needed to collect new data. Set the key first — it is read from the environment and is
never stored in this repository:

```bash
export OPENROUTER_API_KEY=sk-or-...          # PowerShell: $env:OPENROUTER_API_KEY = "sk-or-..."
```

Always dry-run first; it prints the exact prompts and spends nothing.

```bash
julia src/run_matrix.jl pilot --dry
julia src/run_matrix.jl pilot --budget 600
python src/run_reasoning_ladder.py --probe     # costs cents, prices the sweep
python src/run_reasoning_ladder.py --models gem3flash,gem35flash,gem31flashlite \
       --datasets ld50,lipophilicity,boilingpoint --n 60 --iters 3 --threads 32 --budget 25
```

`--budget` is a hard ceiling in USD, checked before every request.

The Julia runners include the shared helper library from `src/lib/` by default; set
`LLM_JULIA_LIB` to point at a copy kept elsewhere.

Every campaign that was run is recoverable from the per-cell JSON: each block records the
command that produced it, its window, its cost and the endpoint settings in its `meta` object.

## Datasets

| key | benchmark | class | source |
|---|---|---|---|
| `esol` | Delaney / ESOL | measured | [Delaney 2004](https://doi.org/10.1021/ci034243x), MoleculeNet |
| `freesolv` | FreeSolv / SAMPL | measured | Mobley & Guthrie 2014, MoleculeNet |
| `lipophilicity` | Lipophilicity | measured | MoleculeNet |
| `bace` | BACE | measured | Subramanian et al. 2016, MoleculeNet |
| `aqsoldb` | AqSolDB | measured | Sorkun et al. 2019 |
| `caco2` | Caco-2 (Wang) | measured | Therapeutics Data Commons |
| `ld50` | LD50 (Zhu) | measured | Therapeutics Data Commons |
| `ppbr` | PPBR (AZ) | measured | Therapeutics Data Commons |
| `qm7` | QM7 | computed | [Rupp et al. 2012](https://doi.org/10.1103/PhysRevLett.108.058301) |
| `qm8` | QM8 | computed | Ramakrishnan et al. 2015 |
| `antiviral` | ASAP/Polaris antiviral potency | recency control | published after every documented cutoff in the panel |
| `boilingpoint` | reference boiling points | positive control | values the models demonstrably know |

## License

MIT — see [LICENSE](LICENSE).
