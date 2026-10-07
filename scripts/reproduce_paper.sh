#!/usr/bin/env bash
# Reproduces the GDSC experiments of the paper:
#   NxtDRP with the MT, MT+PR, MT+EX, MT+PR+EX graphs on the three splitting strategies (IC50),
#   NxtDRP with the maximum concentration (MT+PR+MC) on unseen drugs,
#   NxtDRP MT+PR+EX on AUDRC, and the dummy baselines.
# Results (one csv per split + metrics.json) are saved in results/, the summary in results/summary.csv
#
# Usage: bash scripts/reproduce_paper.sh            (40 splits per experiment, as in the paper)
#        N_TESTS=2 bash scripts/reproduce_paper.sh  (quick check)
#        EXTRA_ARGS="..." adds options to every src/main.py run
set -euo pipefail
cd "$(dirname "$0")/.."

N_TESTS=${N_TESTS:-40}
DEVICE=${DEVICE:-cuda}
RUN="python src/main.py --n_tests ${N_TESTS} --device ${DEVICE} ${EXTRA_ARGS:-}"

python src/data.py --dataset gdsc
python src/data.py --dataset gdsc_auc

for cv in random_split unseen_cell unseen_drug; do
    for omics in none pr ex pr_ex; do
        $RUN --dataset gdsc --model NxtDRP --omics ${omics} --cv_type ${cv}
    done
    $RUN --dataset gdsc_auc --model NxtDRP --omics pr_ex --cv_type ${cv}
done
$RUN --dataset gdsc --model NxtDRPMC --omics pr --cv_type unseen_drug

python src/dummy_models.py --dataset gdsc --n_tests ${N_TESTS}

python src/validation.py evaluate results/*/ --summary results/summary.csv
