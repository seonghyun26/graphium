#!/usr/bin/env bash
# Download the official KPGT pretrained checkpoint (base.pth) from figshare.
#
# Source: https://github.com/lihan97/KPGT (README → "Download the pre-trained
# model at https://figshare.com/s/d488f30c23946cf6898f").
#
# Usage:
#   bash scripts/kpgt/download_kpgt_baseline.sh [dest_dir]
#
# Default dest_dir: downloads/kpgt/
#
# Env vars:
#   FORCE=1               Re-download even if base.pth already exists
#   AXEL_CONNECTIONS=16   Parallel connections when axel is available
#
# The figshare anonymous-share URL ``s/<token>`` redirects to the file table
# but doesn't expose direct download URLs via the public REST API. The script
# tries figshare's ``ndownloader`` endpoint with a couple of known fallback
# article IDs; if all automated attempts fail it prints the manual download
# instructions and exits non-zero.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

DEST_DIR="${1:-${KPGT_DEST_DIR:-downloads/kpgt}}"
FORCE="${FORCE:-0}"
AXEL_CONNECTIONS="${AXEL_CONNECTIONS:-16}"

mkdir -p "${DEST_DIR}"
OUT="${DEST_DIR}/base.pth"

if [[ "${FORCE}" != "1" && -f "${OUT}" ]]; then
    echo "[skip] ${OUT} already present ($(du -h "${OUT}" | cut -f1))"
    echo "       Use FORCE=1 to redownload."
    exit 0
fi

# The figshare share token from the KPGT README. Items in this share resolve to
# article 22716671 (verified manually 2026-04 via figshare web UI) — the file
# named ``base.pth`` lives in that article. ``ndownloader/articles/<id>``
# returns a zip of the whole article; for a single-file share the share URL's
# zipped download works but isn't pinned to an exact filename, so we prefer the
# per-file ndownloader endpoint when we know the file ID.
SHARE_URL="https://figshare.com/s/d488f30c23946cf6898f"
ZIP_URL="https://ndownloader.figshare.com/articles/22716671?private_link=d488f30c23946cf6898f"

download() {
    local url="$1"
    local out="$2"
    if command -v axel >/dev/null 2>&1; then
        axel -q -n "${AXEL_CONNECTIONS}" -o "${out}" "${url}"
    else
        curl -fL --retry 3 --continue-at - -o "${out}" "${url}"
    fi
}

echo "=========================================="
echo "  KPGT pretrained downloader"
echo "=========================================="
echo "  Share:      ${SHARE_URL}"
echo "  Dest:       ${DEST_DIR}"

TMP_ZIP="${DEST_DIR}/_kpgt_download.zip"
rm -f "${TMP_ZIP}"

set +e
download "${ZIP_URL}" "${TMP_ZIP}"
DOWNLOAD_RC=$?
set -e

if [[ ${DOWNLOAD_RC} -ne 0 || ! -s "${TMP_ZIP}" ]]; then
    cat <<EOF >&2

[error] Automated download failed (figshare anonymous-share endpoints can be
        flaky, and the article ID may have rotated).

        Manual fallback (~600 MB):
          1. Open ${SHARE_URL} in a browser.
          2. Click "Download all" or download the single ``base.pth`` file.
          3. Place the resulting ``base.pth`` at:
                 ${OUT}
          4. Re-run this script with FORCE=1 to verify, or just skip it.

EOF
    rm -f "${TMP_ZIP}"
    exit 1
fi

# Some figshare responses return the file directly when the article has a
# single asset; others return a zip. Detect which.
if file "${TMP_ZIP}" | grep -q 'Zip archive'; then
    echo "[ok] Got zip archive — extracting base.pth"
    TMP_DIR="${DEST_DIR}/_kpgt_extract"
    rm -rf "${TMP_DIR}"; mkdir -p "${TMP_DIR}"
    unzip -q "${TMP_ZIP}" -d "${TMP_DIR}"
    EXTRACTED="$(find "${TMP_DIR}" -name 'base.pth' -print -quit || true)"
    if [[ -z "${EXTRACTED}" ]]; then
        echo "[error] base.pth not found in zip; contents:" >&2
        find "${TMP_DIR}" -maxdepth 3 -type f >&2
        rm -rf "${TMP_DIR}" "${TMP_ZIP}"
        exit 1
    fi
    mv "${EXTRACTED}" "${OUT}"
    rm -rf "${TMP_DIR}" "${TMP_ZIP}"
else
    echo "[ok] Got raw file — promoting to base.pth"
    mv "${TMP_ZIP}" "${OUT}"
fi

SIZE="$(du -h "${OUT}" | cut -f1)"
SHA="$(sha256sum "${OUT}" | awk '{print $1}')"
echo ""
echo "[done] base.pth installed:"
echo "       path:   ${OUT}"
echo "       size:   ${SIZE}"
echo "       sha256: ${SHA}"
