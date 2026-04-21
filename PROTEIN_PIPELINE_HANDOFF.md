# Protein-ESM2 Embedding Pipeline — Handoff for Remote Server

## Goal

Build a dataset that maps **SMILES (drug molecules) -> ESM2 protein embeddings** for molecule pre-training. The final output is ~63,405 rows of (SMILES, normalized 2560-dim embedding) used alongside the rxrx3 microscopy dataset (SMILES -> 384-dim OpenPhenom embeddings, 63,405 samples).

## Pipeline Overview (5 stages)

```
Stage 1: Collect DTI data        → protein-drug.csv (1.79M rows)
Stage 2: Extract ESM2 embeddings → individual .pt files per protein (5,222 files)
Stage 3: Consolidate embeddings  → protein-esm2.parquet (5,222 × 2562)
Stage 4: Merge DTI + embeddings  → tdcdti-esm2.parquet (1.62M × 2563)
Stage 5: Filter + normalize      → tdcdti-esm2-filtered-*.parquet/pt (63,405 × 2560)
```

---

## Stage 1: Collect Drug-Target Interaction Data

**Source**: TDC (Therapeutics Data Commons) DTI datasets
**Output**: `data/protein-drug.csv` (~1.79M rows)

Columns:
- `Drug_ID` — compound identifier
- `Drug` — SMILES string of the drug molecule
- `Target_ID` — protein identifier (e.g., "AAK1", "P45877", "Q9UBY0")
- `Target` — full amino acid sequence of the protein
- `Y` — binding affinity value
- `dti_dataset` — source dataset name (e.g., "davis", "bindingdb_patent")
- `Year` — publication year (may be NaN)

This is a many-to-many relationship: one protein binds many drugs, one drug may bind multiple proteins.

**How it was done on the original server**: Downloaded from TDC Python package, concatenated multiple DTI datasets (davis, KIBA, BindingDB variants), deduplicated, saved as CSV.

---

## Stage 2: Extract ESM2 Protein Embeddings

**Input**: Unique protein sequences from Stage 1 (5,222 proteins)
**Model**: `esm2_t36_3B_UR50D` (ESM2 3B parameter model)
**Output**: One `.pt` file per protein in `data/protein-esm2/embedding/`

Each `.pt` file contains a dict with key `mean_representations` -> `{36: tensor([2560])}` (mean-pooled embedding from layer 36).

**Script**: `embedd_protein.py`

Key details:
- Uses `esm-extract` CLI tool (from `fair-esm` package)
- Command per protein: `esm-extract esm2_t36_3B_UR50D <fasta_file> <output_dir> --repr_layers 36 --include mean`
- Multi-GPU parallel processing via thread-based queue (GPUManager class)
- Batch size configurable (default 32 proteins per GPU batch)
- Creates intermediate FASTA files in `data/protein-esm2/fasta/`
- Skips proteins that already have embeddings (idempotent)
- SMILES validation via RDKit before processing
- Caches valid_smiles list and unique_proteins dict as `.pkl` files

**Dependencies**: `torch`, `fair-esm` (provides `esm-extract`), `pandas`, `numpy`, `tqdm`, `rdkit` (optional)

**GPU requirements**: ESM2 3B model needs ~12GB VRAM. Can run on multiple GPUs in parallel.

**Runtime**: ~2-4 hours for 5,222 proteins on 2 GPUs.

---

## Stage 3: Consolidate Individual Embeddings

**Input**: 5,222 individual `.pt` files from Stage 2
**Output**:
- `protein-esm2.csv` (5222 rows × 2562 cols) — uploaded to HuggingFace
- `protein-esm2.parquet` (same data, compressed)
- `protein-esm2.pt` — dict with keys: `protein_id` (list), `smiles` (list of sequences, misleadingly named), `embedding` (numpy array [5222, 2560])

**How it was done**: In `dataset.ipynb` — the CSV was downloaded from HuggingFace (`hyunnnnnnnn/rxrx3_smiles_embedding`), then converted to parquet and .pt.

Columns: `protein_id`, `sequence`, `feature_0` through `feature_2559`

Note: The `protein-esm2.pt` file stores sequences under the key name `"smiles"` (historical naming). The actual SMILES are added in Stage 4.

---

## Stage 4: Merge DTI + Embeddings

**Input**:
- `protein-esm2.csv` (5,222 proteins with embeddings)
- `protein-drug.csv` (1.79M DTI interactions)

**Output**: `tdcdti-esm2.parquet` (1,620,095 rows × 2563 cols)

**Merge logic** (from `dataset.ipynb`):
```python
# Join on both protein_id AND sequence to avoid mismatches
df_out = df_csv.merge(
    map_df,                                    # protein-drug data (Target_ID, Target, SMILES)
    left_on=["protein_id", "sequence"],        # from protein-esm2
    right_on=["Target_ID", "Target"],          # from protein-drug
    how="left",
    validate="m:m",                            # many-to-many (1 protein -> many drugs)
)
```

Result: Each of the 5,222 proteins gets paired with all its drug SMILES. One protein can appear hundreds of times (one row per drug). Total: 1,620,095 rows, 764,618 unique SMILES.

Columns: `protein_id`, `sequence`, `feature_0`..`feature_2559`, `SMILES`

---

## Stage 5: Filter + Normalize (Final Dataset)

**Input**: `tdcdti-esm2.parquet` (1.62M rows)
**Output** (in `graphium/data/dti/`):
- `tdcdti-esm2-filtered-l2.parquet` (~319 MB) — L2-normalized embeddings
- `tdcdti-esm2-filtered-l2.pt` (~655 MB) — same as above in PyTorch format
- `tdcdti-esm2-filtered-zscore.parquet` (~319 MB) — Z-score normalized
- `tdcdti-esm2-filtered-zscore.pt` (~655 MB) — same in PyTorch format
- `tdcdti-esm2-filtered-zscore-stats.pt` (~22 KB) — mean/std vectors for denormalization

**Script**: `create_protein_dataset.py`

### Filtering strategy (seed=42, target=63,405 rows):

**Phase 1** — Sample 1 random row per protein (5,222 rows). Guarantees all proteins appear.
**Phase 2** — From remaining 1.61M rows, select 58,183 rows with novel SMILES (not in Phase 1), deduplicated by SMILES.

Result: 63,405 rows, 5,222 proteins (all represented), ~62,585 unique SMILES.

### Normalization:

**L2 normalization**: Each embedding vector divided by its L2 norm → unit vectors (norm=1).
```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_l2 = embeddings / norms
```

**Z-score normalization**: Per-feature standardization → each feature has mean=0, std=1.
```python
mean = embeddings.mean(axis=0)  # shape [2560]
std = embeddings.std(axis=0)    # shape [2560]
embeddings_zscore = (embeddings - mean) / std
```
The mean/std vectors are saved separately for denormalization at inference time.

### Output .pt format:
```python
{
    "embeddings": torch.FloatTensor([63405, 2560]),  # normalized embeddings
    "smiles": list[str],                              # 63405 SMILES strings
    "protein_id": list[str],                          # 63405 protein IDs
}
```

### Output .parquet format:
Columns: `protein_id`, `SMILES`, `feature_0`..`feature_2559` (no `sequence` column — dropped to save space).

---

## File Inventory

### Source/intermediate files:
| File | Size | Description |
|---|---|---|
| `data/protein-drug.csv` | ~1.3 GB | Raw DTI pairs from TDC |
| `data/protein-esm2/embedding/*.pt` | ~11 KB each | Individual protein embeddings (5,222 files) |
| `data/protein-esm2/fasta/*.fasta` | tiny | Intermediate FASTA files |
| `graphium/data/dti/protein-esm2.csv` | ~159 MB | Consolidated embeddings (5222 × 2562) |
| `graphium/data/dti/protein-esm2.parquet` | ~124 MB | Same, parquet format |
| `graphium/data/dti/protein-esm2.pt` | ~162 MB | Same, PyTorch format |
| `graphium/data/dti/tdcdti-esm2.parquet` | ~217 MB | Full merged DTI+embeddings (1.62M rows) |

### Final output files (what matters for training):
| File | Size | Description |
|---|---|---|
| `tdcdti-esm2-filtered-l2.parquet` | 319 MB | L2-normalized, 63405 rows |
| `tdcdti-esm2-filtered-l2.pt` | 655 MB | Same, PyTorch dict |
| `tdcdti-esm2-filtered-zscore.parquet` | 319 MB | Z-score normalized |
| `tdcdti-esm2-filtered-zscore.pt` | 655 MB | Same, PyTorch dict |
| `tdcdti-esm2-filtered-zscore-stats.pt` | 22 KB | mean/std for denorm |

---

## Reproducing on a New Server

### What you already have (transferred):
- `protein-esm2.csv` — the 5,222 consolidated protein embeddings
- `tdcdti-esm2.parquet` — the 1.62M merged DTI+embedding rows

### What you need to do:
1. **If starting from scratch** (no transferred files): Run Stages 1-5
2. **If you have `tdcdti-esm2.parquet`**: Run Stage 5 only (`create_protein_dataset.py`)
3. **If you have the filtered output files**: Nothing — ready for training

### Quick reproduction (Stage 5 only):

Place `tdcdti-esm2.parquet` at `graphium/data/dti/tdcdti-esm2.parquet`, then run:

```bash
python create_protein_dataset.py
```

This skips Stage 0 (embedding verification) if the individual `.pt` files aren't present. You may need to comment out `step0_verify_embeddings()` or make it conditional.

### Full reproduction (all stages):

```bash
# Stage 1: Get DTI data (requires tdc package)
# pip install PyTDC
# Script needed to download and merge TDC datasets into protein-drug.csv

# Stage 2: Extract embeddings (requires GPU + fair-esm)
# pip install fair-esm
cd tdc-tdi-protein/
python embedd_protein.py

# Stage 3-4: Consolidate and merge (use dataset.ipynb or write equivalent script)
# Loads individual .pt files → protein-esm2.csv/parquet
# Merges with protein-drug.csv → tdcdti-esm2.parquet

# Stage 5: Filter and normalize
python create_protein_dataset.py
```

---

## Key Parameters

- **ESM2 model**: `esm2_t36_3B_UR50D` — 36 layers, 3B params, trained on UniRef50
- **Embedding layer**: 36 (final layer)
- **Pooling**: Mean pooling over sequence tokens
- **Embedding dimension**: 2560
- **Random seed**: 42 (for filtering reproducibility)
- **Target dataset size**: 63,405 rows (matches rxrx3 microscopy dataset)
- **HuggingFace repo**: `hyunnnnnnnn/rxrx3_smiles_embedding`

---

## Scripts Reference

### `embedd_protein.py` (Stage 2)
- Reads `data/protein-drug.csv`
- Validates SMILES with RDKit
- Extracts unique proteins
- Runs ESM2 embedding extraction in parallel across GPUs
- Outputs individual `.pt` files to `data/protein-esm2/embedding/`

### `dataset.ipynb` (Stages 3-4)
- Downloads/loads `protein-esm2.csv` from HuggingFace
- Converts to parquet and .pt formats
- Merges with `protein-drug.csv` on (protein_id, sequence)
- Produces `tdcdti-esm2.parquet` (1.62M rows)

### `create_protein_dataset.py` (Stage 5)
- Verifies embeddings against individual .pt files (Step 0)
- Loads `tdcdti-esm2.parquet` (Step 1)
- Filters to 63,405 rows with protein-stratified sampling (Step 2)
- L2 and z-score normalizes, saves parquet + pt files (Steps 3-4)
- Reloads and verifies all outputs (Step 5)
