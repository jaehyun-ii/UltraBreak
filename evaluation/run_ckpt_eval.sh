#!/usr/bin/env bash
# Evaluate a trained UltraBreak patch (universal adversarial image) against a
# target VLM, end-to-end, using the official 2-step pipeline:
#   attack.py  -> target generates responses over the KO SafeBench attack config
#   evaluate.py-> HarmBench judge scores ASR + (KO-aware) NRR
#
# Usage (on the H100, from repo root, inside the container):
#   NVIDIA_VISIBLE_DEVICES=4 docker compose run --rm ultrabreak \
#     bash evaluation/run_ckpt_eval.sh outputs/test/3000.png
#
# Args:
#   $1 CKPT          patch png to evaluate     (default outputs/test/3000.png)
#   $2 MODEL         target model_name         (default Qwen/Qwen2.5-VL-7B-Instruct)
#   $3 ATTACK_CONFIG attack_configs/<name>.csv (default safebench_ko_jailbroken)
set -euo pipefail

CKPT="${1:-outputs/test/3000.png}"
MODEL="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
ATTACK_CONFIG="${3:-safebench_ko_jailbroken}"

if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2; exit 1
fi

# The config's `image` column points every harmful query at outputs/ultrabreak.png,
# so swapping that file IS how a specific checkpoint gets evaluated. Back up any
# existing one so we don't clobber a prior best patch.
mkdir -p outputs
if [[ -f outputs/ultrabreak.png ]]; then
  cp -f outputs/ultrabreak.png "outputs/ultrabreak.prev.png"
fi
cp -f "$CKPT" outputs/ultrabreak.png
echo "[run_ckpt_eval] evaluating $CKPT  (model=$MODEL  config=$ATTACK_CONFIG)"

# 1) target generates responses -> results/<config>/<model>.csv
python evaluation/attack.py \
  --model_name "$MODEL" \
  --attack_config "$ATTACK_CONFIG"

RESULT="results/${ATTACK_CONFIG}/${MODEL}.csv"
echo "[run_ckpt_eval] attack responses -> $RESULT"

# 2) judge -> ASR / NRR (writes <RESULT>_harmbench.csv, prints the metrics)
python evaluation/evaluate.py --attack_result "$RESULT"

echo "[run_ckpt_eval] done. Re-read the 'ASR' / 'Non-Refusal Rate' lines above."
echo "[run_ckpt_eval] BASELINE control: rerun with a blank image to isolate the"
echo "                patch effect:  bash evaluation/run_ckpt_eval.sh images/white.jpeg"
