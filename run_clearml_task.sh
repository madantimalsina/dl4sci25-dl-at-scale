#!/usr/bin/env bash
# Enqueue a ClearML task for single-GPU training using your repo/branch.
# Usage:
#   ./run_clearml_task.sh [optional:NAME] [optional:QUEUE]
# Environment overrides (optional):
#   PROJECT, BRANCH, SCRIPT, ARGS
set -euo pipefail

PROJECT=${PROJECT:-dl4sci25-dl-at-scale}
NAME=${1:-dl4sci25-dl-at-scale-1gpu-$(date +%Y%m%d-%H%M)}
QUEUE=${2:-muller}

REPO=https://github.com/madantimalsina/dl4sci25-dl-at-scale.git
BRANCH=${BRANCH:-clearml_test}
SCRIPT=${SCRIPT:-train.py}

# Fast sanity config; swap 'short' -> 'bs16_opt' for fuller 1-GPU training
#ARGS=${ARGS:-"--yaml_config=config/ViT.yaml --config=short --num_data_workers=8 --run_num=clearml_1gpu"}
ARGS_KV="yaml_config=config/ViT.yaml config=short num_data_workers=8 run_num=clearml_1gpu"

# NERSC convenience: no-op if modules aren’t used
module load python >/dev/null 2>&1 || true

echo "Submitting ClearML task '$NAME' to queue '$QUEUE'..."
clearml-task \
  --project "$PROJECT" \
  --name "$NAME" \
  --repo "$REPO" \
  --branch "$BRANCH" \
  --script "$SCRIPT" \
  --args $ARGS_KV \
  --queue "$QUEUE"