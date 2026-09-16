# scPhyloGRN standalone CPU quickstart

Open quickstart.ipynb from this extracted directory. It uses the actual trained
embryo1 decoder with fixed encoder embeddings. No training, full expression
matrix, lineage kernel, GPU, or parent repository is needed at runtime.

For beginners, start with notebook section 0 (environment and cell execution).
It uses 28 short code cells, each at most 10 lines, with separate explanations.
Figures show query direction and computed scores; inference is explained in text.
See START_HERE_ZH.md for a Chinese step-by-step guide.

Install requirements-quickstart.txt in the intended environment if needed.
On this cluster, execute only within a Slurm allocation; in the repository use
`sbatch run_quickstart.sbatch`. The portable notebook also works without Slurm
on a personal computer. All runtime paths are relative to this directory.

Files: quickstart.ipynb, quickstart_utils.py, inference_utils.py, tutorial_plots.py,
requirements-quickstart.txt, START_HERE_ZH.md, and model/quickstart/ (decoder weights, embeddings,
metadata, checksums, gene list, example pairs, expected values, validation).

The notebook explains loading, input editing, CPU prediction, independent
example checks, score interpretation, ranking, CSV export and common errors.
Figures and prediction tables are saved in a new timestamped results/ subfolder.
Full graph reconstruction remains in inference_demo.ipynb in the full repository.
The small ZIP does not include the full encoder checkpoint or original data.

In the full repository, run_quickstart.sbatch first packages the current tutorial
on the allocated CPU node, then executes an extracted copy outside the repository.
This reuses existing environments; it is not a clean-install validation of every
dependency combination permitted by the requirements file.

The manifest binds embeddings, weights and metadata with checksums. Metadata
records the source checkpoint/input hashes and training context. Validation
compares CPU scoring with GPU encoder/decoder inference and archived reference
predictions. This verifies computation, not independent biological accuracy.

No public DOI or redistribution licence has been assigned to this draft package.
Confirm the original data/model redistribution terms before public Zenodo upload.
