#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

python scripts/make_primary_flank400_figures.py

pandoc paper_draft/full_draft_current.md \
  --from markdown \
  --to html5 \
  --standalone \
  --citeproc \
  --resource-path="$PROJECT_ROOT" \
  --css="$PROJECT_ROOT/paper_draft/two_column_paper.css" \
  --embed-resources \
  -o paper_draft/full_draft_current.html

if command -v weasyprint >/dev/null 2>&1; then
  weasyprint \
    paper_draft/full_draft_current.html \
    paper_draft/full_draft_current.pdf
elif python -c 'import weasyprint' >/dev/null 2>&1; then
  python -m weasyprint \
    paper_draft/full_draft_current.html \
    paper_draft/full_draft_current.pdf
else
  echo "WeasyPrint is required. Install it with: pip install weasyprint==61.1 pydyf==0.10.0" >&2
  exit 1
fi

pdfinfo paper_draft/full_draft_current.pdf | head -20
