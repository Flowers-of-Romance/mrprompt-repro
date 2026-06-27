#!/bin/bash
P="/mnt/c/Users/jun/AppData/Local/Temp/claude/C--windows/d8bb1b4b-4485-488d-b253-29f9d55c2cbc/scratchpad"
for v in comfy-rocm irodori-venv heartmula chatterbox-venv HeartMuLa-Studio/venv anima-train/venv anima-prompt-pipeline/venv smart-comfyui-gallery/venv llama-env acestep; do
  py="/home/jun/$v/bin/python"
  [ -x "$py" ] || continue
  echo "===== $v ====="
  "$py" "$P/env_check.py" 2>&1 | grep -E "torch|device|hip|transformers|accelerate" || echo "(no torch/tf)"
done
