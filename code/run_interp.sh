#!/bin/bash
# LEVER A + B interpretability run (GPU only, no API). Logs to ~/mdrp-repro/.
SC=/mnt/c/Users/jun/AppData/Local/Temp/claude/C--windows/d8bb1b4b-4485-488d-b253-29f9d55c2cbc/scratchpad
PY=/home/jun/comfy-rocm/bin/python
B=~/mdrp-repro
cd "$SC"
FILT='FutureWarning|UserWarning|warnings.warn|sysfs|Loading checkpoint|attn_output|TheRock'

echo "### LEVER A: cue-routing null + 16-22 localization, n=29 ###" > "$B/interp_A.log"
$PY run_probe_ext.py 2>&1 | grep -vE "$FILT" >> "$B/interp_A.log"
echo "[A done]" >> "$B/interp_A.log"

echo "### LEVER B: per-character bridge, n=17 ###" > "$B/interp_B.log"
$PY bridge_percharacter.py 2>&1 | grep -vE "$FILT" >> "$B/interp_B.log"
echo "[B done]" >> "$B/interp_B.log"

echo "ALL DONE" >> "$B/interp_A.log"
