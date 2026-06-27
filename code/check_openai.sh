#!/bin/bash
for py in /home/jun/comfy-rocm/bin/python /home/jun/irodori-venv/bin/python /home/jun/heartmula/bin/python /usr/bin/python3; do
  [ -x "$py" ] || continue
  v=$("$py" -c 'import openai;print(openai.__version__)' 2>/dev/null)
  if [ -n "$v" ]; then echo "$py : openai $v"; else echo "$py : MISSING"; fi
done
echo "--- pip available for a fresh venv? ---"
python3 -c 'import venv; print("venv module ok")' 2>&1
