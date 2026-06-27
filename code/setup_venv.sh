#!/bin/bash
set -e
cd ~/mdrp-repro
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q openai
./venv/bin/python -c 'import openai; print("openai", openai.__version__)'
