# Pretrained inference quickstart

Open `scPhyloGRN_quickstart.ipynb` from this directory. The tutorial uses the
packaged trained decoder and fixed encoder embeddings, so it runs on a CPU and
does not require the full training data or a lineage kernel.

Install its lightweight dependencies if needed:

```bash
python -m pip install -r requirements-quickstart.txt
```

The notebook verifies `decoder.pt`, `embeddings.npy`, and `metadata.json`
against `model/quickstart/manifest.json` before loading them. Do not edit these
assets independently. See `START_HERE_ZH.md` for Chinese instructions and
`QUICKSTART.md` for the complete package description and limitations.
