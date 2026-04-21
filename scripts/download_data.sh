#!/usr/bin/env bash
# Download pre-training datasets from a Hugging Face dataset repo created by
# scripts/upload_data.sh, then extract each archive into its target location
# under <target-root>/ (default /home/shpark/prj-molrepr/).
#
# Reads MANIFEST.json from the repo to drive archive → extract-path mapping,
# so it stays in sync with whatever the uploader included.
#
# Usage:
#   bash scripts/download_data.sh <repo-id> [flags]
#
# <repo-id> is the HF repo, e.g. "hyunnnnnnnn/graphium-pretrain-data".
#
# Flags:
#   --only LIST        Subset of groups {toymix,bbbc047,esmc}. Default: all.
#   --target-root DIR  Where to untar. Default: /home/shpark/prj-molrepr.
#                      Configs under graphium/expts/hydra-configs/ hardcode
#                      this path in ~114 YAMLs; change only if you sed the
#                      configs afterwards.
#   --stage-dir DIR    Where to store tar files during download. Default:
#                      /tmp/hf_download_stage. Needs ~14 GB peak (largest tar).
#   --keep-stage       Don't delete tars after extraction. Useful for retry.
#   --list             Print the manifest + plan, then exit.
#   --apply            Actually download + extract. Without this, runs dry-run.
#   -h, --help         Show this help.
#
# Examples:
#   bash scripts/download_data.sh hyunnnnnnnn/graphium-pretrain-data --list
#   bash scripts/download_data.sh hyunnnnnnnn/graphium-pretrain-data --apply
#   bash scripts/download_data.sh hyunnnnnnnn/graphium-pretrain-data \
#       --only toymix,esmc --target-root /data/molrepr --apply
#
# Prereqs:
#   - `hf` CLI available (conda activate graphium).
#   - For a public repo, login is not required. For a private repo, `hf auth login`.
#
# Pairs with: scripts/upload_data.sh

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Arg parsing ──────────────────────────────────────────────────────────────
ONLY="toymix,bbbc047,esmc"
TARGET_ROOT="/home/shpark/prj-molrepr"
STAGE_DIR="/tmp/hf_download_stage"
KEEP_STAGE=0
LIST_ONLY=0
APPLY=0
REPO_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)        ONLY="$2"; shift 2 ;;
        --target-root) TARGET_ROOT="$2"; shift 2 ;;
        --stage-dir)   STAGE_DIR="$2"; shift 2 ;;
        --keep-stage)  KEEP_STAGE=1; shift ;;
        --list)        LIST_ONLY=1; shift ;;
        --apply)       APPLY=1; shift ;;
        -h|--help)     sed -n '1,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)            echo "Unknown flag: $1" >&2; exit 2 ;;
        *)
            if [[ -z "${REPO_ID}" ]]; then
                REPO_ID="$1"
            else
                echo "Unexpected positional arg: $1" >&2; exit 2
            fi
            shift ;;
    esac
done

if [[ -z "${REPO_ID}" ]]; then
    echo "Error: missing <repo-id>. Run with --help for usage." >&2
    exit 2
fi

# ── Prereq checks ────────────────────────────────────────────────────────────
if ! command -v hf >/dev/null; then
    echo "Error: 'hf' CLI not found. Activate the graphium env:" >&2
    echo "  conda activate graphium" >&2
    exit 2
fi
if ! command -v python >/dev/null; then
    echo "Error: python not on PATH (needed to parse MANIFEST.json)." >&2
    exit 2
fi

IFS=',' read -ra SELECTED <<< "${ONLY}"
in_selected () {
    local g=$1
    for s in "${SELECTED[@]}"; do
        [[ "${s}" == "${g}" ]] && return 0
    done
    return 1
}

# ── Fetch MANIFEST.json ──────────────────────────────────────────────────────
mkdir -p "${STAGE_DIR}"
MANIFEST="${STAGE_DIR}/MANIFEST.json"

echo ">>> fetching MANIFEST.json from ${REPO_ID}"
hf download --repo-type dataset "${REPO_ID}" MANIFEST.json \
    --local-dir "${STAGE_DIR}" >/dev/null

if [[ ! -s "${MANIFEST}" ]]; then
    echo "Error: MANIFEST.json missing or empty in repo ${REPO_ID}" >&2
    exit 1
fi

# Parse MANIFEST.json into TSV: group<TAB>archive<TAB>extract_to
ENTRIES=$(python - <<PY
import json, sys
with open("${MANIFEST}") as f:
    m = json.load(f)
for a in m.get("archives", []):
    print(f"{a['group']}\t{a['archive']}\t{a['extract_to']}")
PY
)

if [[ -z "${ENTRIES}" ]]; then
    echo "Error: MANIFEST has no archives." >&2; exit 1
fi

# ── Filter by --only and build plan ──────────────────────────────────────────
declare -a PLAN_GROUP PLAN_ARCH PLAN_EXTRACT
while IFS=$'\t' read -r group arch extract; do
    in_selected "${group}" || continue
    PLAN_GROUP+=("${group}")
    PLAN_ARCH+=("${arch}")
    PLAN_EXTRACT+=("${extract}")
done <<< "${ENTRIES}"

if [[ ${#PLAN_ARCH[@]} -eq 0 ]]; then
    echo "Nothing matches --only=${ONLY}." >&2; exit 1
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Download plan ==="
echo "Repo        : ${REPO_ID}"
echo "Groups      : ${ONLY}"
echo "Target root : ${TARGET_ROOT}"
echo "Stage dir   : ${STAGE_DIR}"
echo ""
for i in "${!PLAN_ARCH[@]}"; do
    grp="${PLAN_GROUP[$i]}"
    arch="${PLAN_ARCH[$i]}"
    extract="${PLAN_EXTRACT[$i]}"
    printf "  [%-7s] %s  ->  %s/%s\n" "${grp}" "${arch}" "${TARGET_ROOT}" "${extract}"
done
echo ""

if [[ ${LIST_ONLY} -eq 1 ]]; then
    exit 0
fi
if [[ ${APPLY} -eq 0 ]]; then
    echo "(dry-run — pass --apply to actually download + extract)"
    exit 0
fi

# ── Download + extract loop ──────────────────────────────────────────────────
for i in "${!PLAN_ARCH[@]}"; do
    grp="${PLAN_GROUP[$i]}"
    arch="${PLAN_ARCH[$i]}"
    extract="${PLAN_EXTRACT[$i]}"
    stage_tar="${STAGE_DIR}/${arch}"
    mkdir -p "$(dirname "${stage_tar}")"

    echo ""
    echo ">>> [${grp}] download ${arch} -> ${stage_tar}"
    if [[ -f "${stage_tar}" ]]; then
        echo "    already downloaded, skipping"
    else
        hf download --repo-type dataset "${REPO_ID}" "${arch}" \
            --local-dir "${STAGE_DIR}" >/dev/null
        if [[ ! -f "${stage_tar}" ]]; then
            echo "Error: expected file not downloaded: ${stage_tar}" >&2
            exit 1
        fi
    fi

    # The archive contains a top-level dir equal to the source's basename;
    # extract into the parent of extract-to so paths match the original layout.
    dest_parent="${TARGET_ROOT}/$(dirname "${extract}")"
    mkdir -p "${dest_parent}"

    echo ">>> [${grp}] extract -> ${dest_parent}/"
    # Auto-detect gzip compression
    if [[ "${arch}" == *.gz ]]; then
        tar -xzf "${stage_tar}" -C "${dest_parent}"
    else
        tar -xf "${stage_tar}" -C "${dest_parent}"
    fi

    if [[ ${KEEP_STAGE} -eq 0 ]]; then
        rm -f "${stage_tar}"
    fi
done

# Clean up empty stage sub-dirs if we deleted everything
if [[ ${KEEP_STAGE} -eq 0 ]]; then
    find "${STAGE_DIR}" -type d -empty -delete 2>/dev/null || true
fi

echo ""
echo "=== Download complete ==="
echo ""
echo "Verify a few paths:"
for extract in "${PLAN_EXTRACT[@]}"; do
    path="${TARGET_ROOT}/${extract}"
    if [[ -e "${path}" ]]; then
        size=$(du -sh "${path}" 2>/dev/null | awk '{print $1}')
        echo "  OK    ${path}  (${size})"
    else
        echo "  MISS  ${path}"
    fi
done
