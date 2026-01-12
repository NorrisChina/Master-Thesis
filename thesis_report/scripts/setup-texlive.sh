#!/usr/bin/env bash
set -euo pipefail

# Install LaTeX tools and TUD dependencies
sudo apt-get update
sudo apt-get install -y \
  latexmk \
  biber \
  texlive-publishers \
  texlive-latex-extra \
  texlive-fonts-extra

echo "TeX Live setup complete."
