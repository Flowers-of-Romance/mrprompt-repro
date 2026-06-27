#!/bin/bash
echo "=== candidate dirs (steering / blog source) ==="
ls -d ~/poptones ~/flowers* ~/*steer* ~/*activation* ~/*poptones* 2>/dev/null
echo "=== vector files .pt/.npz/.npy/.safetensors (steer/vec/emot/lang/direction) ==="
find ~ -maxdepth 6 -type f \( -iname "*.pt" -o -iname "*.npz" -o -iname "*.npy" -o -iname "*.safetensors" \) 2>/dev/null | grep -iE "steer|vec|emot|lang|direct|diff" | head -40
echo "=== code/data (py/json/jsonl/csv) mentioning steering/contrast/200 ==="
find ~ -maxdepth 6 -type f \( -iname "*.py" -o -iname "*.json" -o -iname "*.jsonl" -o -iname "*.csv" \) 2>/dev/null | grep -iE "steer|activation|contrast|diff.?mean|pairs" | head -40
echo "=== blog/post dirs (poptones/flowers) ==="
find ~ -maxdepth 5 -type d 2>/dev/null | grep -iE "poptones|flowers|activation-steer" | head -20
