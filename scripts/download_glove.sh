#!/usr/bin/env bash
# Download GloVe 6B (Wikipedia + Gigaword, 400K words) and extract the 300-d vectors.
set -euo pipefail
mkdir -p data
cd data
if [ ! -f glove.6B.300d.txt ]; then
  curl -L -o glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
  unzip -o glove.6B.zip glove.6B.300d.txt
  rm glove.6B.zip
fi
echo "ready: data/glove.6B.300d.txt"
