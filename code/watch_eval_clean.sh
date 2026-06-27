#!/bin/bash
# Wait for the definitive gen_clean.py run to finish, then auto-run judge -> analyze.
SC=/mnt/c/Users/jun/AppData/Local/Temp/claude/C--windows/d8bb1b4b-4485-488d-b253-29f9d55c2cbc/scratchpad
B=~/mdrp-repro
LOG=$B/watch_eval_clean.log
cd "$SC"

echo "[watcher] waiting for gen_clean.py to finish..." > "$LOG"
while pgrep -f gen_clean.py >/dev/null; do sleep 120; done
LINES=$(wc -l < "$B/generations_clean.jsonl")
echo "[watcher] gen_clean exited. generations_clean.jsonl = $LINES lines (target 700)" >> "$LOG"

if [ "$LINES" -lt 700 ]; then
  echo "[watcher] INCOMPLETE (<700). Not judging. Re-run gen_clean.py to resume." >> "$LOG"
  exit 1
fi

source ~/.openai_env
echo "[watcher] ### judging definitive run ###" >> "$LOG"
"$B/venv/bin/python" judge.py "$B/generations_clean.jsonl" "$B/scores_clean.jsonl" >> "$LOG" 2>&1 || { echo "[watcher] judge FAILED" >> "$LOG"; exit 1; }
echo "[watcher] ### analysis (N=100, all 7, uniform thinking) ###" >> "$LOG"
/home/jun/comfy-rocm/bin/python analyze.py "$B/scores_clean.jsonl" >> "$LOG" 2>&1
echo "[watcher] DONE" >> "$LOG"
