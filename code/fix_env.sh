#!/bin/bash
# Pull the real OPENAI_API_KEY assignment out of ~/.bashrc (which has a non-interactive
# guard that stops `bash -c` from reaching it) into a guard-free ~/.openai_env that any
# script can `source`. The key value is never printed.
raw=$(grep -E 'OPENAI_API_KEY=' ~/.bashrc 2>/dev/null | grep -vE '^[[:space:]]*#' | tail -1)
if [ -z "$raw" ]; then echo "NO assignment found in ~/.bashrc"; exit 1; fi
assign=$(printf '%s\n' "$raw" | sed -E 's/^[[:space:]]*export[[:space:]]+//')
printf 'export %s\n' "$assign" > ~/.openai_env
chmod 600 ~/.openai_env
source ~/.openai_env
echo "value_len=${#OPENAI_API_KEY}"
[ -z "$OPENAI_API_KEY" ] && { echo "still empty after copy"; exit 1; }
code=$(curl -s -o /tmp/oa.json -w "%{http_code}" https://api.openai.com/v1/models \
       -H "Authorization: Bearer $OPENAI_API_KEY")
echo "http=$code"
echo "gpt-4.1 family visible to this key:"
grep -oE '"id"[: ]*"gpt-4\.1[^"]*"' /tmp/oa.json | sed -E 's/.*"(gpt[^"]*)"/  \1/' | sort -u
echo "judge (gpt-4.1-mini) present: $(grep -c 'gpt-4.1-mini' /tmp/oa.json)"
