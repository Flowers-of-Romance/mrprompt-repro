#!/bin/bash
# sources the key (guard-free file) then runs the repro venv python on the given script+args
source ~/.openai_env
exec ~/mdrp-repro/venv/bin/python "$@"
