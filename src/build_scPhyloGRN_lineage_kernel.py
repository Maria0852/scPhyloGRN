#!/usr/bin/env python
"""Build a scPhyloGRN cell-lineage kernel from a Newick tree."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scPhyloGRN_model import build_lineage_K_from_newick


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newick", required=True, help="Input lineage tree in Newick format.")
    parser.add_argument(
        "--expr",
        required=True,
        help="Gene-by-cell expression CSV; column names must match Newick leaf names.",
    )
    parser.add_argument("--out", default="lineage_K.npy", help="Output .npy or .npz path.")
    parser.add_argument("--tau", type=float, default=4.0, help="Lineage-distance decay coefficient.")
    parser.add_argument(
        "--k-top", "--k_top", dest="k_top", type=int, default=32,
        help="Number of nearest lineage leaves retained per cell.",
    )
    return parser


def main() -> None:
    """Read aligned cells, construct the kernel, and save it to disk."""
    args = build_parser().parse_args()
    output_path = Path(args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading expression: {args.expr}")
    expr = pd.read_csv(args.expr, index_col=0)
    cell_names = expr.columns.astype(str).tolist()
    print(f"[expr] cells={len(cell_names)}, genes={expr.shape[0]}")

    print(f"[build] constructing lineage kernel (tau={args.tau}, k_top={args.k_top})")
    kernel = build_lineage_K_from_newick(
        newick_path=args.newick,
        cell_names=cell_names,
        tau=args.tau,
        k_top=args.k_top,
    ).astype(np.float32)

    if output_path.suffix.lower() == ".npz":
        np.savez(output_path, K=kernel)
    else:
        np.save(output_path, kernel)
    print(f"[done] saved lineage kernel with shape {kernel.shape} to {output_path}")


if __name__ == "__main__":
    main()
