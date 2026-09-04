#!/usr/bin/env bash
# Section "Interrupting retrieval", re-run at the controlled map's setting.
#
# Panel and dose are the map's, not the original mitigation experiment's: four models at a
# 1,024-token thinking budget, on the three recalled benchmarks. Both arms below are paired to
# map cells that are already bought, so neither buys a comparison condition:
#
#   randomised SMILES   the canonical t1024 cells ARE the comparison (same 500 molecules)
#   L1 -> L5            L1 is the within-experiment comparison for L5 (same 150 molecules)
#
# Run the audit first. It costs nothing and it is the only check that the L5 arm hides what it
# is supposed to hide:
#
#   python src/audit_l5_prompts.py
#
# Then dry-run both blocks. A mis-templated prompt is otherwise discovered after the matrix has
# been paid for.
set -u

# Interpreter and repository root. Both are overridable; the defaults work from a fresh clone.
PY=${PY:-python}
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"
MODELS=${MODELS:-opus5,grok45,sol,kimik3}
DATASETS=${DATASETS:-esol freesolv ld50}
DRY=${DRY:-}                       # DRY=--dry to print and spend nothing
THREADS=${THREADS:-16}

echo "=== A. randomised SMILES, zero-shot, 500 molecules, 1,024-token budget ==="
# One invocation for all twelve cells: the runner is resumable per cell, so an interrupted run
# is continued by re-issuing the same command. Budget is the hard ceiling, checked before every
# request -- projected $71 from 5,545 measured calls per model, $54 from the one-call probe.
"$PY" src/run_budget_sweep.py \
    --variant random \
    --datasets "$(echo $DATASETS | tr ' ' ',')" \
    --models "$MODELS" \
    --n 500 --block 0 --iters 1 \
    --thinking 1024 --threads "$THREADS" --budget 95 $DRY

echo
echo "=== B. L1 -> L5, 100 shots, 150 test molecules, 1,024-token budget ==="
# Per (benchmark, model) so a stall costs one cell rather than the campaign. The test set of each
# benchmark excludes molecules of the other two: ESOL, FreeSolv and LD50 overlap heavily, and it
# is the IDENTIFICATION step that L5 blocks, so a shared molecule would couple the arms.
#
# --max-tokens is the completion cap, not the thinking budget. It has to be far above 1,024:
# measured here, both GPT-5.6 sol and Kimi K3 emit 3,000-3,800 thinking tokens on L5 against a
# 1,024-token request, because an unreadable structure string is what they deliberate hardest
# about. A trace that eats the cap returns an empty answer, which scores as a miss -- i.e. as a
# successful intervention. That failure mode has to be impossible, not unlikely.
#
# 20,000, not 12,000, and the Moonshot family is why. Kimi K2.6 emitted 9,000-12,000 thinking
# tokens on this kind of prompt and returned NO answer at any setting (EXPERIMENTS_TODO, deferred
# models). Kimi K3 is better behaved -- 433 median in the budget arm, 3,009 on an L5 smoke test --
# but it is from the same family and L5 is the condition that stretches it. A cap that a model can
# reach is a cap that silently manufactures the result this experiment is looking for. Unemitted
# tokens cost nothing, so the headroom is free and there is no reason to be clever about it.
for ds in $DATASETS; do
    others=$(echo $DATASETS | tr ' ' '\n' | grep -v "^${ds}$" | paste -sd,)
    for tag in $(echo "$MODELS" | tr ',' ' '); do
        echo "--- $ds / $tag (test set disjoint from $others) ---"
        "$PY" src/run_blinding_l1_l5.py \
            --dataset "$ds" --tag "$tag" \
            --ntest 150 --shots 100 --levels L1,L5 \
            --reasoning max_tokens:1024 --max-tokens 20000 \
            --threads 10 --budget 25 $DRY
    done
done

echo
echo "=== analysis ==="
echo "  $PY src/run_budget_sweep.py --verify"
echo "  $PY src/analyze_budget.py --pad --perm 2000        # writes an arm column; t1024r is the new one"
echo "  $PY src/analyze_blinding_sweep.py --arm t1024      # never pooled with the legacy arm"
