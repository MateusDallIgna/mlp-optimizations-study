#!/usr/bin/env bash
# Downloads the Fashion-MNIST dataset into data/ (raw IDX files, uncompressed).
# Usage: ./scripts/get_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"

FILES=(
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
)

# Primary source (Zalando repository) and fallback (Zalando S3 bucket).
MIRRORS=(
  "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
  "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
)

mkdir -p "$DATA_DIR"

for file in "${FILES[@]}"; do
  if [[ -s "$DATA_DIR/$file" ]]; then
    echo "skip  $file (already present)"
    continue
  fi

  downloaded=0
  for mirror in "${MIRRORS[@]}"; do
    url="$mirror/$file.gz"
    if curl -fsSL --retry 3 --connect-timeout 15 -o "$DATA_DIR/$file.gz" "$url"; then
      gzip -df "$DATA_DIR/$file.gz"
      echo "ok    $file"
      downloaded=1
      break
    fi
    rm -f "$DATA_DIR/$file.gz"
    echo "warn  $file not available at $mirror, trying next mirror"
  done

  if [[ $downloaded -eq 0 ]]; then
    echo "error: could not download $file" >&2
    exit 1
  fi
done

echo "Fashion-MNIST ready in $DATA_DIR"
