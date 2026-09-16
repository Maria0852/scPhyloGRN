# scPhyloGRN

scPhyloGRN infers ranked, directed transcription-factor (TF)–target candidates from a gene-by-cell expression matrix and an aligned cell-lineage kernel. This repository contains the publication-facing training code, lineage-kernel builder, downstream analysis utilities, and reproducible examples for the manuscript **“Lineage-aware inference of developmental gene regulatory networks from lineage-resolved single-cell data with scPhyloGRN.”**

## Installation

Python 3.8 is used for the manuscript workflow. Create the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate scPhyloGRN
```

Alternatively, create a virtual environment and install the pip requirements:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Repository layout

```text
src/
  run_scPhyloGRN.py                    Main training and prediction CLI
  scPhyloGRN_model.py                  Reusable model and graph utilities
  build_scPhyloGRN_lineage_kernel.py   Newick-to-kernel CLI
  analyze_scPhyloGRN_results.py        Ranked-edge summary and plotting CLI
  validate_scPhyloGRN_predictions.py   External-reference validation CLI
data/                                  Public input data included in this release
examples/
  quickstart/                          Executed pretrained-inference tutorial
  training_demo/                       Small synthetic end-to-end training example
```

## Required inputs

The training CLI requires three aligned inputs:

1. **Expression matrix (`--expr`)**: CSV with genes in rows, cells in columns, and gene identifiers in the first column.
2. **Reference edges (`--ref`)**: CSV containing `gene_i,gene_j`, `tf,target`, `src,dst`, or `gene1,gene2`. The first column is interpreted as the regulator/source.
3. **Lineage kernel (`--lineage-k`)**: NumPy `.npy` file containing a finite, non-negative cell-by-cell matrix. Its row and column order must match the expression-matrix column order.

Duplicate genes are reduced to their first occurrence. Reference edges with genes absent from the expression matrix are removed. The training, validation, and test split is performed at the directed-edge level; genes may occur in more than one partition.

## End-to-end workflow

### 1. Build a lineage kernel from a Newick tree

```bash
python src/build_scPhyloGRN_lineage_kernel.py \
  --newick path/to/lineage_tree.newick \
  --expr path/to/expression.csv \
  --out path/to/lineage_K.npy \
  --tau 4.0 \
  --k-top 32
```

### 2. Train scPhyloGRN and export ranked edges

```bash
python src/run_scPhyloGRN.py \
  --expr path/to/expression.csv \
  --ref path/to/reference_edges.csv \
  --lineage-k path/to/lineage_K.npy \
  --outdir results/main/scPhyloGRN \
  --seed 42
```

Inspect all options with `python src/run_scPhyloGRN.py --help`.

The main output is a timestamped `predicted_edges_*.csv` file with regulator (`gene_i`), target (`gene_j`), uncalibrated logit/probability, and temperature-scaled probability (`prob_T`). Scores indicate model support for candidate edges; they are not effect sizes or activation/repression signs.

### 3. Summarize ranked predictions

```bash
python src/analyze_scPhyloGRN_results.py \
  --input results/main/scPhyloGRN/predicted_edges_YYYYMMDD_HHMMSS.csv \
  --outdir results/main/scPhyloGRN/analysis
```

### 4. Validate against external resources

This optional step downloads external reference resources and therefore requires network access:

```bash
python src/validate_scPhyloGRN_predictions.py \
  --input results/main/scPhyloGRN/analysis/top1000_edges.csv \
  --outdir results/main/scPhyloGRN/validation_report
```

## Examples

- [`examples/training_demo/scPhyloGRN_training_demo.ipynb`](examples/training_demo/scPhyloGRN_training_demo.ipynb) generates compact synthetic inputs, trains the actual model for a few epochs, and inspects the exported edge scores. It is a functional smoke test and interface tutorial, not a biological benchmark.
- [`examples/quickstart/scPhyloGRN_quickstart.ipynb`](examples/quickstart/scPhyloGRN_quickstart.ipynb) is a standalone inference tutorial for scoring gene pairs with the packaged trained decoder and fixed encoder embeddings. Its manifest verifies the bundled model assets before loading. See [`examples/quickstart/START_HERE_ZH.md`](examples/quickstart/START_HERE_ZH.md) for Chinese instructions.

## Reproducibility notes

- `--seed` controls Python, NumPy, and PyTorch random-number generators.
- Test labels are used only for final evaluation, not model or hyperparameter selection.
- GPU execution is used when CUDA is available; the synthetic training demo is configured for CPU execution.
- Record the Git commit, release tag, input checksums, and command line when reproducing manuscript results.
- The versioned Zenodo record should contain the source tree, example inputs, archived pretrained weights, and the exact notebooks referenced above.

## Troubleshooting

- **Input not found:** check the paths supplied to the CLI.
- **Kernel shape mismatch:** ensure the kernel is square and matches the number and order of expression-matrix columns.
- **Validation metric error:** each validation/test partition must contain both positive and negative edges.
- **Out of memory:** reduce `--batch-size`, `--proj-dim`, `--out-dim`, or the number of genes.

## Availability

Source code is hosted at <https://github.com/Maria0852/scPhyloGRN>. The manuscript cites the versioned Zenodo archive at <https://doi.org/10.5281/zenodo.19914743>. Update the repository release and Zenodo version together so that the archived source, notebooks, and packaged quickstart assets remain identical.
