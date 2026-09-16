#!/usr/bin/env python3
"""Summarize and visualize ranked scPhyloGRN edge predictions.

The command reads a prediction table produced by ``run_scPhyloGRN.py``,
exports the highest-scoring edges, plots the score distribution, and draws
small directed subnetworks for the most frequently represented regulators.
No reference labels are used by this post-processing utility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def read_predictions(path: Path) -> pd.DataFrame:
    """Load a prediction table and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction table not found: {path}")
    predictions = pd.read_csv(path)
    required = {"gene_i", "gene_j", "prob_T"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    return predictions


def plot_score_histogram(predictions: pd.DataFrame, output: Path) -> None:
    """Plot the distribution of calibrated candidate-edge probabilities."""
    plt.figure(figsize=(6, 4))
    plt.hist(predictions["prob_T"], bins=100, color="steelblue", edgecolor="white")
    plt.xlabel("Calibrated probability (prob_T)")
    plt.ylabel("Count")
    plt.title("Distribution of predicted edge probabilities")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_regulator_subnetwork(
    predictions: pd.DataFrame,
    regulator: str,
    edge_limit: int,
    output: Path,
) -> None:
    """Draw the highest-scoring outgoing edges for one regulator."""
    subset = (
        predictions.loc[predictions["gene_i"] == regulator]
        .sort_values("prob_T", ascending=False)
        .head(edge_limit)
    )
    if subset.empty:
        return

    graph = nx.DiGraph()
    for row in subset.itertuples(index=False):
        graph.add_edge(row.gene_i, row.gene_j, weight=float(row.prob_T))

    positions = nx.spring_layout(graph, seed=42, k=0.8)
    edge_weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    node_sizes = [500 if node == regulator else 300 for node in graph.nodes()]
    node_colors = ["tomato" if node == regulator else "skyblue" for node in graph.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, positions, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(graph, positions, font_size=8)
    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        edge_color=edge_weights,
        edge_cmap=plt.cm.Blues,
        width=2,
    )
    plt.title(f"{regulator} predicted regulatory subnetwork (top {edge_limit})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def analyze_predictions(input_csv: Path, outdir: Path, top_n: int, edges_per_tf: int) -> None:
    """Run all summaries and write them beneath ``outdir``."""
    if top_n < 1 or edges_per_tf < 1:
        raise ValueError("top_n and edges_per_tf must be positive integers")
    outdir.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(input_csv)
    print(f"[info] loaded {len(predictions):,} predicted edges")

    histogram_path = outdir / "prob_T_hist.png"
    plot_score_histogram(predictions, histogram_path)

    top_edges = predictions.sort_values("prob_T", ascending=False).head(top_n)
    top_path = outdir / f"top{top_n}_edges.csv"
    top_edges.to_csv(top_path, index=False)

    regulator_counts = top_edges["gene_i"].value_counts()
    for regulator in regulator_counts.head(3).index:
        plot_regulator_subnetwork(
            predictions,
            regulator,
            edges_per_tf,
            outdir / f"{regulator}_subnetwork.png",
        )

    summary = pd.Series(
        {
            "edge_count": len(predictions),
            "mean_prob_T": predictions["prob_T"].mean(),
            "median_prob_T": predictions["prob_T"].median(),
            "max_prob_T": predictions["prob_T"].max(),
            "edges_prob_T_gt_0.9": int((predictions["prob_T"] > 0.9).sum()),
            f"top_{top_n}_mean_prob_T": top_edges["prob_T"].mean(),
        },
        name="value",
    )
    summary.to_csv(outdir / "prediction_summary.csv", header=True)
    print(f"[done] outputs written to {outdir}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="scPhyloGRN predicted_edges CSV")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory for tables and plots")
    parser.add_argument("--top-n", type=int, default=1000, help="Number of top edges to export")
    parser.add_argument("--edges-per-tf", type=int, default=30, help="Edges shown per regulator subnetwork")
    return parser


def main() -> None:
    """Parse command-line arguments and run the analysis."""
    args = build_parser().parse_args()
    analyze_predictions(args.input, args.outdir, args.top_n, args.edges_per_tf)


if __name__ == "__main__":
    main()
