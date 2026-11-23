#!/usr/bin/env bash
# Download GloVe 6B 300-d vectors (Wikipedia + Gigaword, 400K words) to data/glove.6B.300d.txt.
# Tries Stanford first, then the gensim-data mirror on GitHub (same vectors, gzipped,
# with a "400000 300" header line, which is stripped here).
set -euo pipefail
mkdir -p data
cd data
OUT=glove.6B.300d.txt
if [ -f "$OUT" ]; then echo "ready: data/$OUT"; exit 0; fi

if curl -fL --connect-timeout 10 -o glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip; then
  unzip -o glove.6B.zip "$OUT" && rm glove.6B.zip
else
  echo "Stanford server unreachable; using the gensim-data mirror on GitHub"
  rm -f glove.6B.zip
  curl -fL -o glove-300.gz \
    https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz
  gunzip -c glove-300.gz | tail -n +2 > "$OUT"
  rm glove-300.gz
fi
echo "ready: data/$OUT ($(wc -l < "$OUT") vectors)"
