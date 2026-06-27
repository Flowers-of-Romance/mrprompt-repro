#!/bin/bash
PDF=/mnt/c/Users/jun/AppData/Local/Temp/claude/C--windows/d8bb1b4b-4485-488d-b253-29f9d55c2cbc/scratchpad/mdrp.pdf
OUT=/tmp/mdrp.txt
: > "$OUT"
if command -v pdftotext >/dev/null 2>&1; then
  echo "[using pdftotext]"
  pdftotext "$PDF" "$OUT"
else
  echo "[no pdftotext; trying python pdf libs]"
  for py in /home/jun/comfy-rocm/bin/python /home/jun/irodori-venv/bin/python /home/jun/heartmula/bin/python /usr/bin/python3; do
    [ -x "$py" ] || continue
    "$py" - "$PDF" "$OUT" <<'PY'
import sys
pdf, out = sys.argv[1], sys.argv[2]
t = ""
for mod in ("fitz", "pypdf", "pdfminer.high_level"):
    try:
        if mod == "fitz":
            import fitz; d = fitz.open(pdf); t = "".join(p.get_text() for p in d)
        elif mod == "pypdf":
            from pypdf import PdfReader; t = "".join((p.extract_text() or "") for p in PdfReader(pdf).pages)
        else:
            from pdfminer.high_level import extract_text; t = extract_text(pdf)
        if t:
            print("  extracted with", mod, "chars", len(t)); break
    except Exception as e:
        print("  fail", mod, repr(e)[:70])
open(out, "w").write(t)
PY
    [ -s "$OUT" ] && break
  done
fi
echo "chars in OUT: $(wc -c < "$OUT")"
echo "=== all URLs ==="
grep -oE 'https?://[^ )>"]+' "$OUT" | sort -u
echo "=== anon / repo / data lines ==="
grep -iE 'anonym|4open|github|zenodo|repositor|huggingface|/r/|requirements|release' "$OUT" | sed 's/^ *//' | head -40
