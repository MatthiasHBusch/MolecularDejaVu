#!/usr/bin/env bash
# Deep sweep, dataset-major so each benchmark completes across all 30 models before the next
# starts -- run_matrix.jl iterates model-major internally, so the ordering has to come from
# here. Every cell is resumable: re-running skips completed cells, so this can be interrupted.
set -u
cd "$(dirname "$0")/.."
N=${N:-1000}; IT=${IT:-2}; TH=${TH:-32}; BUD=${BUD:-600}
for ds in freesolv esol ld50 qm8 antiviral boilingpoint lipophilicity qm7; do
  echo "################ $ds  $(date +%H:%M:%S) ################"
  julia src/run_matrix.jl deep-mdl --datasets "$ds" --n "$N" --iters "$IT" \
        --threads "$TH" --budget "$BUD"
done
echo "################ DEEP SWEEP COMPLETE $(date +%H:%M:%S) ################"
