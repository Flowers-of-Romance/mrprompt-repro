#!/bin/bash
source ~/.openai_env
cd /mnt/c/Users/jun/AppData/Local/Temp/claude/C--windows/d8bb1b4b-4485-488d-b253-29f9d55c2cbc/scratchpad
B=~/mdrp-repro
echo "### judging thinking-on generations ###"
~/mdrp-repro/venv/bin/python judge.py "$B/generations_think.jsonl" "$B/scores_think.jsonl" || exit 1
echo "### merging ###"
~/mdrp-repro/venv/bin/python merge_scores.py || exit 1
echo "### analysis (thinking-ON: base/card off, mrprompt-family on) ###"
/home/jun/comfy-rocm/bin/python analyze.py "$B/scores_v2.jsonl"
