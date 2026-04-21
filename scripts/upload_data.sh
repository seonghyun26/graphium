#!/usr/bin/env bash
# Upload pre-training datasets (toymix, bbbc047, esmc) to a public Hugging Face
# dataset repo.  Mirrors the grouping of scripts/move_pretrain_data.sh but
# transports via HF Hub (git-LFS) instead of rsync/ssh.
#
# Each source directory is tarred into a single archive in a staging dir and
# pushed to the repo under {group}/{archive}.tar.  A MANIFEST.json is uploaded
# at the end; the companion scripts/download_data.sh reads it to decide where
# to extract each tar on the target machine.
#
# Usage:
#   bash scripts/upload_data.sh <repo-id> [flags]
#
# <repo-id> is the HF repo, e.g. "hyunnnnnnnn/graphium-pretrain-data".
#
# Flags:
#   --only LIST       Subset of {toymix,bbbc047,esmc}. Default: all three.
#   --stage-dir DIR   Staging dir for tar files. Default: /tmp/hf_upload_stage.
#                     Needs ~40 GB free across the selected groups.
#   --create          Create the repo on HF (public) if it doesn't exist.
#   --compress        Gzip tars (via pigz if available). Default: no compression
#                     — datacache contents are largely binary and don't shrink.
#   --keep-stage      Don't delete staged tars after upload. Useful for retry.
#   --list            Print planned archives with sizes, then exit.
#   --apply           Actually tar + upload. Without this, runs dry-run.
#   -h, --help        Show this help.
#
# Examples:
#   bash scripts/upload_data.sh hyunnnnnnnn/graphium-pretrain-data --list
#   bash scripts/upload_data.sh hyunnnnnnnn/graphium-pretrain-data --create --apply
#   bash scripts/upload_data.sh hyunnnnnnnn/graphium-pretrain-data --only esmc --apply
#
# Prereqs (verified before upload):
#   - `hf` CLI installed in active env: `conda activate graphium`
#   - Logged in: `hf auth login`
#   - ~40 GB free at $STAGE_DIR and disk at $HOME for LFS cache
#
# Pairs with: scripts/download_data.sh (reads MANIFEST.json and extracts tars).

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO_ROOT=$(pwd)                       # .../prj-molrepr/graphium
PARENT_ROOT=$(cd .. && pwd)            # .../prj-molrepr

# ── Source paths ─────────────────────────────────────────────────────────────
# Each entry:  <group>|<abs-source-path>|<archive-name>|<optional:glob-filter>|<extract-to-under-prj-molrepr>
# archive-name is where the tar lands inside the HF repo.
# extract-to tells the download script where to untar on the target machine.

SPECS=(
    # toymix: raw CSVs + splits (~44 MB)
    "toymix|${REPO_ROOT}/data/graphium/neurips2023/small-dataset/|toymix/small-dataset.tar||graphium/data/graphium/neurips2023/small-dataset"
    # toymix: featurized datacache (~8.7 GB, many small pickles — tar critical)
    "toymix|${PARENT_ROOT}/datacache/neurips2023-small/|toymix/datacache_neurips2023-small.tar||datacache/neurips2023-small"

    # bbbc047: raw cell-painting embeddings (~2.1 GB, 3 CSVs)
    "bbbc047|${PARENT_ROOT}/data/bbbc047/|bbbc047/data.tar||data/bbbc047"
    # bbbc047: joint toymix + bbbc047 datacache (~14 GB)
    "bbbc047|${PARENT_ROOT}/datacache/toymix_bbbc047/|bbbc047/datacache_toymix_bbbc047.tar||datacache/toymix_bbbc047"

    # esmc: DTI CSVs filtered to dti_esmc_* (~4.6 GB)
    "esmc|${REPO_ROOT}/data/dti-processed/|esmc/dti-processed-esmc.tar|dti_esmc_*|graphium/data/dti-processed"
    # esmc: per-target residue embeddings (~41 MB, 5k files — tar critical)
    "esmc|${REPO_ROOT}/data/protein-esmc/|esmc/protein-esmc.tar||graphium/data/protein-esmc"
    # esmc: joint toymix + DTI + ESM-C datacache (~11 GB)
    "esmc|${PARENT_ROOT}/datacache/toymix-dti-esmc-v2/|esmc/datacache_toymix-dti-esmc-v2.tar||datacache/toymix-dti-esmc-v2"
)

# ── Arg parsing ──────────────────────────────────────────────────────────────
ONLY="toymix,bbbc047,esmc"
STAGE_DIR="/tmp/hf_upload_stage"
CREATE=0
COMPRESS=0
KEEP_STAGE=0
LIST_ONLY=0
APPLY=0
REPO_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)       ONLY="$2"; shift 2 ;;
        --stage-dir)  STAGE_DIR="$2"; shift 2 ;;
        --create)     CREATE=1; shift ;;
        --compress)   COMPRESS=1; shift ;;
        --keep-stage) KEEP_STAGE=1; shift ;;
        --list)       LIST_ONLY=1; shift ;;
        --apply)      APPLY=1; shift ;;
        -h|--help)    sed -n '1,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)           echo "Unknown flag: $1" >&2; exit 2 ;;
        *)
            if [[ -z "${REPO_ID}" ]]; then
                REPO_ID="$1"
            else
                echo "Unexpected positional arg: $1" >&2; exit 2
            fi
            shift ;;
    esac
done

if [[ ${LIST_ONLY} -eq 0 && -z "${REPO_ID}" ]]; then
    echo "Error: missing <repo-id>. Run with --help for usage." >&2
    exit 2
fi

# ── Prereq checks ────────────────────────────────────────────────────────────
if [[ ${APPLY} -eq 1 ]]; then
    if ! command -v hf >/dev/null; then
        echo "Error: 'hf' CLI not found. Activate the graphium env:" >&2
        echo "  conda activate graphium" >&2
        exit 2
    fi
    if ! hf auth whoami >/dev/null 2>&1; then
        echo "Error: not logged in. Run 'hf auth login' and retry." >&2
        exit 2
    fi
fi

IFS=',' read -ra SELECTED <<< "${ONLY}"
in_selected () {
    local g=$1
    for s in "${SELECTED[@]}"; do
        [[ "${s}" == "${g}" ]] && return 0
    done
    return 1
}

# Pick tar compression
if [[ ${COMPRESS} -eq 1 ]]; then
    if command -v pigz >/dev/null; then
        TAR_COMPRESS=(-I pigz); ARCHIVE_EXT=".gz"
    else
        TAR_COMPRESS=(-z);      ARCHIVE_EXT=".gz"
    fi
else
    TAR_COMPRESS=();            ARCHIVE_EXT=""
fi

# ── Build plan ───────────────────────────────────────────────────────────────
declare -a PLAN_GROUP PLAN_SRC PLAN_ARCH PLAN_FILTER PLAN_EXTRACT

for spec in "${SPECS[@]}"; do
    IFS='|' read -r group src arch filter extract <<< "${spec}"
    filter="${filter:-}"
    in_selected "${group}" || continue

    if [[ ! -e "${src%/}" ]]; then
        echo "WARN: source missing, skipping: ${src}" >&2
        continue
    fi
    PLAN_GROUP+=("${group}")
    PLAN_SRC+=("${src}")
    PLAN_ARCH+=("${arch}${ARCHIVE_EXT}")
    PLAN_FILTER+=("${filter}")
    PLAN_EXTRACT+=("${extract}")
done

if [[ ${#PLAN_SRC[@]} -eq 0 ]]; then
    echo "Nothing to do." >&2; exit 1
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=== Upload plan ==="
echo "Repo        : ${REPO_ID:-(not provided)}"
echo "Groups      : ${ONLY}"
echo "Stage dir   : ${STAGE_DIR}"
echo "Compression : $([[ ${COMPRESS} -eq 1 ]] && echo 'gzip' || echo 'none')"
echo ""
for i in "${!PLAN_SRC[@]}"; do
    src="${PLAN_SRC[$i]}"
    flt="${PLAN_FILTER[$i]}"
    grp="${PLAN_GROUP[$i]}"
    arch="${PLAN_ARCH[$i]}"
    if [[ -n "${flt}" ]]; then
        size=$(du -ch -- "${src}"${flt} 2>/dev/null | tail -n1 | awk '{print $1}')
    else
        size=$(du -sh -- "${src}" 2>/dev/null | awk '{print $1}')
    fi
    printf "  [%-7s] %-5s  %s  ->  %s\n" "${grp}" "${size:-?}" "${src}" "${arch}"
done
echo ""

if [[ ${LIST_ONLY} -eq 1 ]]; then
    exit 0
fi

if [[ ${APPLY} -eq 0 ]]; then
    echo "(dry-run — pass --apply to actually tar + upload)"
    exit 0
fi

# ── Create repo if requested ─────────────────────────────────────────────────
if [[ ${CREATE} -eq 1 ]]; then
    echo ">>> Creating dataset repo ${REPO_ID} (public, exist-ok)"
    hf repo create "${REPO_ID}" --repo-type dataset --exist-ok
fi

mkdir -p "${STAGE_DIR}"
MANIFEST="${STAGE_DIR}/MANIFEST.json"

# Start MANIFEST.json
{
    echo "{"
    echo "  \"version\": 1,"
    echo "  \"repo_id\": \"${REPO_ID}\","
    echo "  \"archives\": ["
} > "${MANIFEST}"
first=1

# ── Tar + upload loop ────────────────────────────────────────────────────────
for i in "${!PLAN_SRC[@]}"; do
    src="${PLAN_SRC[$i]}"
    grp="${PLAN_GROUP[$i]}"
    arch="${PLAN_ARCH[$i]}"
    flt="${PLAN_FILTER[$i]}"
    extract="${PLAN_EXTRACT[$i]}"
    stage_tar="${STAGE_DIR}/${arch}"
    mkdir -p "$(dirname "${stage_tar}")"

    echo ""
    echo ">>> [${grp}] tar ${src} -> ${stage_tar}"
    if [[ -f "${stage_tar}" ]]; then
        echo "    already staged, skipping tar step"
    else
        # Use tar -C to strip the path; the archive contains the dir contents
        # without the source prefix, so download-side can extract into the
        # target `extract-to` directly.
        src_parent=$(dirname "${src%/}")
        src_base=$(basename "${src%/}")
        if [[ -n "${flt}" ]]; then
            # Filtered archive: only include files matching the glob
            ( cd "${src_parent}" && \
              find "${src_base}" -maxdepth 1 -type f -name "${flt}" -print0 | \
              tar -c --null -T - "${TAR_COMPRESS[@]}" -f "${stage_tar}" )
        else
            tar -c "${TAR_COMPRESS[@]}" -f "${stage_tar}" -C "${src_parent}" "${src_base}"
        fi
        echo "    staged: $(du -sh "${stage_tar}" | awk '{print $1}')"
    fi

    echo ">>> [${grp}] upload -> ${REPO_ID}:${arch}"
    hf upload --repo-type dataset "${REPO_ID}" "${stage_tar}" "${arch}"

    # Append to MANIFEST
    [[ ${first} -eq 0 ]] && echo "," >> "${MANIFEST}"
    first=0
    cat >> "${MANIFEST}" <<EOF
    {
      "group": "${grp}",
      "archive": "${arch}",
      "extract_to": "${extract}",
      "source_basename": "${src_base}"
    }
EOF

    if [[ ${KEEP_STAGE} -eq 0 ]]; then
        rm -f "${stage_tar}"
    fi
done

# Close MANIFEST.json and upload it last
{
    echo ""
    echo "  ]"
    echo "}"
} >> "${MANIFEST}"

echo ""
echo ">>> upload MANIFEST.json"
hf upload --repo-type dataset "${REPO_ID}" "${MANIFEST}" "MANIFEST.json"

echo ""
echo "=== Upload complete ==="
echo "Repo: https://huggingface.co/datasets/${REPO_ID}"
echo "On the target machine, pull with:"
echo "    bash scripts/download_data.sh ${REPO_ID} --apply"
