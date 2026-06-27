#!/bin/bash
export HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/home/jun/comfy-rocm/bin/python
P="/mnt/c/Users/jun/AppData/Local/Temp/claude/C--windows/d8bb1b4b-4485-488d-b253-29f9d55c2cbc/scratchpad"
cd "$P" || exit 1
MODEL="${1:-Qwen/Qwen3-8B}"
LOG="$P/run_$(echo "$MODEL" | tr '/' '_').log"
echo "=== model=$MODEL  gpu=$($PY -c 'import torch;print(torch.cuda.get_device_name(0))') ===" | tee "$LOG"
"$PY" cue_routing_probe.py --model "$MODEL" --attn 2>&1 | tee -a "$LOG"
echo "=== exit=${PIPESTATUS[0]} ===" | tee -a "$LOG"
