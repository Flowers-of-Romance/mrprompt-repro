#!/bin/bash
source ~/.openai_env
cd /mnt/c/Users/jun/AppData/Local/Temp/claude/C--windows/d8bb1b4b-4485-488d-b253-29f9d55c2cbc/scratchpad
B=~/mdrp-repro
echo "### judging definitive run ###"
~/mdrp-repro/venv/bin/python judge.py "$B/generations_clean.jsonl" "$B/scores_clean.jsonl" || exit 1
echo "### analysis (N=100, all 7, uniform thinking) ###"
/home/jun/comfy-rocm/bin/python analyze.py "$B/scores_clean.jsonl"
