#!/usr/bin/env bash
# Download the official MolE baseline artifacts from the Zenodo record linked in
# https://github.com/rolayoalarcon/MolE.
#
# Usage:
#   bash scripts/mole/download_mole_baseline.sh [dest_dir]
#
# Examples:
#   bash scripts/mole/download_mole_baseline.sh
#   bash scripts/mole/download_mole_baseline.sh downloads/mole
#
# Environment:
#   FORCE=1             Redownload even if checksum-valid files already exist
#   AXEL_CONNECTIONS=16 Parallel connections when axel is available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Required by scripts/AGENTS.md conventions.
source "${SCRIPT_DIR}/../common.sh"

RECORD_ID="10803099"
DEST_DIR="${1:-${MOLE_DEST_DIR:-downloads/mole}}"
FORCE="${FORCE:-0}"
AXEL_CONNECTIONS="${AXEL_CONNECTIONS:-16}"

mkdir -p "${DEST_DIR}"

URL_BASE="https://zenodo.org/api/records/${RECORD_ID}/files"

declare -A URLS=(
    [config.yaml]="${URL_BASE}/config.yaml/content"
    [model.pth]="${URL_BASE}/model.pth/content"
)

declare -A MD5S=(
    [config.yaml]="1192c76113766e0d742bc34cb91c81d8"
    [model.pth]="3d084daa5f75a0bde23caf7a24a43f8d"
)

md5_of() {
    md5sum "$1" | awk '{print $1}'
}

verify_file() {
    local path="$1"
    local filename
    filename="$(basename "${path}")"
    [[ -f "${path}" ]] || return 1
    [[ "$(md5_of "${path}")" == "${MD5S[$filename]}" ]]
}

download_with_axel() {
    local url="$1"
    local out="$2"
    axel -q -n "${AXEL_CONNECTIONS}" -o "${out}" "${url}"
}

download_with_curl() {
    local url="$1"
    local out="$2"
    curl -L --retry 3 --fail --continue-at - -o "${out}" "${url}"
}

download_file() {
    local filename="$1"
    local out="${DEST_DIR}/${filename}"
    local tmp="${out}.part"

    if [[ "${FORCE}" != "1" ]] && verify_file "${out}"; then
        echo "[skip] ${filename} already present with expected checksum"
        return 0
    fi

    rm -f "${tmp}"
    echo "[download] ${filename}"

    if command -v axel >/dev/null 2>&1; then
        download_with_axel "${URLS[$filename]}" "${tmp}"
    else
        download_with_curl "${URLS[$filename]}" "${tmp}"
    fi

    mv "${tmp}" "${out}"

    if ! verify_file "${out}"; then
        echo "[error] checksum mismatch for ${filename}" >&2
        echo "        expected: ${MD5S[$filename]}" >&2
        echo "        actual:   $(md5_of "${out}")" >&2
        exit 1
    fi

    echo "[ok] ${filename} $(du -h "${out}" | cut -f1) md5=$(MD5S[$filename])"
}

echo "=========================================="
echo "  MolE baseline downloader"
echo "=========================================="
echo "  Source record: ${RECORD_ID}"
echo "  Destination:   ${DEST_DIR}"
echo ""

download_file config.yaml
download_file model.pth

echo ""
echo "Done."
echo "Files:"
ls -lh "${DEST_DIR}/config.yaml" "${DEST_DIR}/model.pth"
