#!/bin/bash
F=~/.openai_env
echo "exists: $([ -e "$F" ] && echo yes || echo NO)"
echo "bytes:  $(wc -c < "$F" 2>/dev/null || echo 0)"
echo "has-export-line: $(grep -c 'OPENAI_API_KEY=' "$F" 2>/dev/null || echo 0)"
source "$F" 2>/dev/null
echo "value_len after source: ${#OPENAI_API_KEY}"
