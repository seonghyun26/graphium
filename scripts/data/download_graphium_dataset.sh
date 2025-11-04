# ============================================
# Download Graphium ToyMix datasets
# Destination: ./data/graphium/neurips2023/small-dataset/
# Sources: https://doi.org/10.5281/zenodo.10797794
# ============================================

cd ../../

set -euo pipefail

ROOT="./data/graphium"
SMALL="${ROOT}/neurips2023/small-dataset"   # ToyMix (QM9, Tox21, ZINC12k)
LARGE="${ROOT}/neurips2023/large-dataset"   # LargeMix (PCQM4M_G25_N4, PCBA1328, L1000)

mkdir -p "$SMALL" "$LARGE"

echo "==> Downloading zips from Zenodo…"

# Files listed on the Zenodo record:
#   qm9.zip, Tox21.zip, ZINC12k.zip, pcqm4m_g25_n4.zip, pcba_1328.zip, l1000_vcap.zip, l1000_mcf7.zip
# (MD5 hashes taken from the Zenodo page.)

declare -A URLS=(
  # small (ToyMix)
#   [${SMALL}/qm9.zip]="https://zenodo.org/records/10797794/files/qm9.zip?download=1"
#   [${SMALL}/Tox21.zip]="https://zenodo.org/records/10797794/files/Tox21.zip?download=1"
#   [${SMALL}/ZINC12k.zip]="https://zenodo.org/records/10797794/files/ZINC12k.zip?download=1"
  # large (LargeMix)
#   [${LARGE}/pcqm4m_g25_n4.zip]="https://zenodo.org/records/10797794/files/pcqm4m_g25_n4.zip?download=1"
#   [${LARGE}/pcba_1328.zip]="https://zenodo.org/records/10797794/files/pcba_1328.zip?download=1"
  [${LARGE}/l1000_vcap.zip]="https://zenodo.org/records/10797794/files/l1000_vcap.zip?download=1"
  [${LARGE}/l1000_mcf7.zip]="https://zenodo.org/records/10797794/files/l1000_mcf7.zip?download=1"
)


declare -A MD5S=(
  # small (ToyMix)
#   [${SMALL}/qm9.zip]="03be6960c1d97e4cc08dfe4b6623825d"
#   [${SMALL}/Tox21.zip]="f4baa8570a0913bb9efc763b1eb66c60"
#   [${SMALL}/ZINC12k.zip]="73fc6e5c67d4e6e25b59ec0d7433d769"
  # large (LargeMix)
#   [${LARGE}/pcqm4m_g25_n4.zip]="5a60b50ff3ad4594f978ecb8ee482070"
#   [${LARGE}/pcba_1328.zip]="7ae4c2f871fe86b07f4ec161023caa62"
  [${LARGE}/l1000_vcap.zip]="c43d36c69888df8c182919537d68c6a3"
  [${LARGE}/l1000_mcf7.zip]="22d1b7276a834fd60fbf00742eed82f5"
)

for OUT in "${!URLS[@]}"; do
  URL="${URLS[$OUT]}"
  echo "  -> ${OUT##*/}"
  curl -L --retry 3 --fail -o "$OUT" "$URL"
  # Verify md5
  echo "${MD5S[$OUT]}  $(basename "$OUT")" | (cd "$(dirname "$OUT")" && md5sum -c -)
done

echo "==> Unzipping…"
# Unzip ToyMix (small)
# for Z in "$SMALL"/*.zip; do
#   echo "  -> ${Z##*/}"
#   unzip -o -q "$Z" -d "$SMALL"
# done

# Unzip LargeMix (large)
# for Z in "$LARGE"/*.zip; do
#   echo "  -> ${Z##*/}"
#   unzip -o -q "$Z" -d "$LARGE"
# done

echo "==> Done."
echo "Directory tree:"
# echo "  ${SMALL}/   # contains ToyMix raws (QM9, Tox21, ZINC12k)"
echo "  ${LARGE}/   # contains LargeMix raws (PCQM4M_G25_N4, PCBA1328, L1000)"