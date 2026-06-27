#!/bin/bash
set -e
mkdir -p ~/mdrp-repro && cd ~/mdrp-repro
if [ ! -d CharacterEval ]; then
  echo "cloning CharacterEval (shallow)..."
  git clone --depth 1 https://github.com/morecry/CharacterEval.git 2>&1 | tail -3
fi
echo "=== top-level ==="
ls -la CharacterEval | head -30
echo "=== dirs (depth<=2) ==="
find CharacterEval -maxdepth 2 -type d | head -30
echo "=== data files (json/jsonl/csv) with sizes ==="
find CharacterEval -maxdepth 3 -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.csv" -o -name "*.tsv" \) -printf "%10s  %p\n" 2>/dev/null | sort -k2 | head -30
echo "=== any README mention of data download / huggingface / gdrive ==="
grep -rIEl 'huggingface|drive.google|download|\.zip' CharacterEval/README* 2>/dev/null | head
