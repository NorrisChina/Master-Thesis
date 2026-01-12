#!/usr/bin/env bash
set -euo pipefail

# Build thesis PDF from a clean state
cd "$(dirname "$0")/.."
latexmk -C
latexmk -pdf -pdflatex='pdflatex -interaction=nonstopmode -file-line-error' thesis.tex
