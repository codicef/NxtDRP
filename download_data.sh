#!/usr/bin/env bash
# Downloads the preprocessed raw data from the GitHub release into data/raw/
set -euo pipefail

RELEASE_URL="https://github.com/codicef/NxtDRP/releases/download/dataset"
cd "$(dirname "$0")/data"

for archive in raw_nxtdrp_data.zip ccle_nxtdrp_data.zip; do
    echo "Downloading ${archive}..."
    curl -fL -o "${archive}" "${RELEASE_URL}/${archive}"
    unzip -o "${archive}"
    rm "${archive}"
done

echo "Raw data available in $(pwd)/raw"
