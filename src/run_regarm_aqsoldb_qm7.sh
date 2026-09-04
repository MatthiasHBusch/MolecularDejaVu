#!/usr/bin/env bash
# Complete the minimum-deliberation arm on AqSolDB and QM7, once the mitigation campaign is done.
#
# WHY THESE TWO. The minimum arm covers 7 of the 12 benchmarks; AqSolDB, BACE, Caco-2, PPBR and
# QM7 are missing. AqSolDB is the one that matters: 15 of its 22 cells are contaminated at a
# 1,024-token budget, so the paired delta there measures the paper's central mechanism on a
# benchmark the zero-shot screen could not test at all. QM7 is included for the cost contrast
# rather than the verdict -- it is the most expensive column of the budget arm ($136, $6.19 per
# cell) and only 1 of its 22 cells is flagged, with 10 untestable. Expect its minimum-arm column
# to be mostly untestable too, because deliberation is what buys accuracy there and accuracy is
# what m2 counts. That is a result about cost, not about contamination; do not read it as one.
#
# The cells draw from the SAME frozen 500-molecule target as the 1,024-token arm (both benchmarks
# are already in results/budget/_target.json), so the comparison is paired by construction. The
# arm tag `reg` keeps the two apart on disk and in analyze_budget.arm_of.
#
# This script waits for the mitigation campaign to finish before spending anything: two sweeps
# against the same endpoints would compete for the same rate limits and make both slower.
set -u

# Interpreter and repository root. Both are overridable; the defaults work from a fresh clone.
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-python}
PANEL="gem25pro,gem31flashlite,gem31pro,gem35flash,gem36flash,gem3flash,gpt41,gpt5,gpt55,grok43,grok45,haiku45,kimik3,llama4mav,luna,minimaxm3,mistrall,opus5,qwen3,sol,sonnet5,terra"
DATASETS=${DATASETS:-aqsoldb,qm7}
WAIT_FOR=${WAIT_FOR:-}          # log file of the run to wait for; empty = start immediately

cd "$ROOT"

if [ -n "$WAIT_FOR" ]; then
    echo "waiting for $(basename "$WAIT_FOR") to reach its analysis banner ..."
    # The mitigation script prints "=== analysis ===" as its last act. Polling the log is more
    # robust than polling for a process name: the campaign is a loop of separate python
    # invocations, so "no python running" is true between every cell.
    while ! grep -q "^=== analysis ===" "$WAIT_FOR" 2>/dev/null; do
        sleep 120
    done
    echo "mitigation campaign finished at $(date '+%H:%M:%S'); starting the minimum arm"
    sleep 30
fi

LOG="results/logs/regarm_$(echo "$DATASETS" | tr ',' '_')_$(date +%Y%m%d_%H%M).log"
echo "=== minimum-deliberation arm: $DATASETS, 500 molecules, 22 models ==="
echo "log: $LOG"

# Projected $29 from the measured per-model $/call of the existing reg arm, scaled by each
# model's own QM7-vs-rest cost ratio in the budget arm. Ceiling set well above it: the ceiling is
# checked before every request, so a runaway stops rather than being discovered on the invoice.
"$PY" src/run_budget_sweep.py \
    --arm registry \
    --datasets "$DATASETS" \
    --models "$PANEL" \
    --n 500 --block 0 --iters 1 \
    --threads 16 --budget 40 2>&1 | tee "$LOG"

echo
echo "=== next ==="
echo "  $PY src/run_budget_sweep.py --verify"
echo "  $PY src/analyze_budget.py --pad --perm 2000"
echo "  $PY src/smooth_error_null.py --arm budget"
echo "  $PY src/classify.py"
echo "  $PY src/analyze_reasoning_delta.py     # the paired table gains two benchmarks"
