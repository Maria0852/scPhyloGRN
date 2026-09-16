#!/usr/bin/env python3
"""Compare scPhyloGRN predictions with external regulatory resources.

This optional, network-dependent utility downloads mouse TF–target resources,
reports exact directed-edge overlap, and performs Enrichr analyses for predicted
targets. External databases can change over time; archive the downloaded files
when using this script for a reproducible analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


TRRUST_URL = "https://www.grnpedia.org/trrust/download/mouse_trrust_rawdata.txt"
REGNETWORK_URL = "https://regnetworkweb.org/download/human_mus_musculus.txt"


def read_predictions(path: Path) -> pd.DataFrame:
    """Load ranked edges and validate columns required for overlap analysis."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction table not found: {path}")
    predictions = pd.read_csv(path)
    required = {"gene_i", "gene_j", "prob_T"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    return predictions


def download_reference_edges() -> pd.DataFrame:
    """Download and combine directed mouse edges from TRRUST and RegNetwork."""
    tables: list[pd.DataFrame] = []

    try:
        trrust = pd.read_csv(
            TRRUST_URL,
            sep="\t",
            header=None,
            names=["gene_i", "gene_j", "mode", "reference"],
        )
        tables.append(trrust[["gene_i", "gene_j"]])
        print(f"[reference] TRRUST: {len(trrust):,} rows")
    except Exception as error:  # network and upstream formatting can both fail
        print(f"[warning] TRRUST download failed: {error}")

    try:
        regnetwork = pd.read_csv(
            REGNETWORK_URL,
            sep="\t",
            header=None,
            names=["gene_i", "gene_j", "type", "source"],
        )
        regnetwork = regnetwork.loc[
            regnetwork["type"].str.contains("TF", na=False), ["gene_i", "gene_j"]
        ]
        tables.append(regnetwork)
        print(f"[reference] RegNetwork: {len(regnetwork):,} TF rows")
    except Exception as error:
        print(f"[warning] RegNetwork download failed: {error}")

    if not tables:
        raise RuntimeError("No external reference resource could be downloaded")
    return pd.concat(tables, ignore_index=True).astype(str).drop_duplicates()


def run_enrichment(
    genes: Iterable[str],
    gene_sets: list[str],
    output: Path,
    organism: str,
) -> pd.DataFrame:
    """Run Enrichr and save a compact result table.

    Returns an empty table when Enrichr is unavailable so that edge-overlap
    results remain usable independently of the enrichment service.
    """
    unique_genes = sorted(set(map(str, genes)))
    if not unique_genes:
        return pd.DataFrame()
    try:
        import gseapy as gp

        enrichment = gp.enrichr(
            gene_list=unique_genes,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,
            cutoff=0.05,
        )
        keep = ["Gene_set", "Term", "Overlap", "Adjusted P-value", "Combined Score"]
        results = enrichment.results.loc[:, keep]
        results.to_csv(output, index=False)
        return results
    except Exception as error:
        print(f"[warning] enrichment failed for {output.name}: {error}")
        return pd.DataFrame()


def validate_predictions(
    input_csv: Path,
    outdir: Path,
    top_n: int,
    top_tfs: int,
    organism: str,
) -> None:
    """Run directed-edge overlap and target-set enrichment analyses."""
    if top_n < 1 or top_tfs < 1:
        raise ValueError("top_n and top_tfs must be positive integers")
    outdir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(input_csv).sort_values("prob_T", ascending=False).head(top_n)
    references = download_reference_edges()
    references.to_csv(outdir / "downloaded_reference_edges.csv", index=False)

    predicted_edges = set(zip(predictions["gene_i"].astype(str), predictions["gene_j"].astype(str)))
    reference_edges = set(zip(references["gene_i"], references["gene_j"]))
    matched_edges = sorted(predicted_edges.intersection(reference_edges))
    match_rate = len(matched_edges) / len(predicted_edges) if predicted_edges else 0.0
    pd.DataFrame(matched_edges, columns=["gene_i", "gene_j"]).to_csv(
        outdir / "matched_edges.csv", index=False
    )

    global_enrichment = run_enrichment(
        predictions["gene_j"],
        ["GO_Biological_Process_2023", "KEGG_2021_Mouse"],
        outdir / "enrichment_results.csv",
        organism,
    )

    regulator_counts = predictions["gene_i"].value_counts()
    regulator_summaries: list[tuple[str, int, pd.DataFrame]] = []
    for regulator in regulator_counts.head(top_tfs).index:
        targets = predictions.loc[predictions["gene_i"] == regulator, "gene_j"].drop_duplicates()
        targets.to_csv(outdir / f"{regulator}_targets.txt", index=False, header=False)
        enrichment = run_enrichment(
            targets,
            ["GO_Biological_Process_2023"],
            outdir / f"{regulator}_enrichment.csv",
            organism,
        )
        regulator_summaries.append((str(regulator), len(targets), enrichment))

    with (outdir / "validation_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("=== scPhyloGRN external validation summary ===\n")
        handle.write(f"Evaluated predicted edges: {len(predicted_edges):,}\n")
        handle.write(f"Downloaded reference edges: {len(reference_edges):,}\n")
        handle.write(f"Exact directed matches: {len(matched_edges):,} ({match_rate:.2%})\n\n")
        if not global_enrichment.empty:
            handle.write("[Global enrichment: first 10 rows]\n")
            handle.write(global_enrichment.head(10).to_string(index=False))
            handle.write("\n\n")
        handle.write("[Top regulators]\n")
        for regulator, target_count, enrichment in regulator_summaries:
            handle.write(f"{regulator}: {target_count} targets")
            if not enrichment.empty:
                handle.write(f"; top term: {enrichment.iloc[0]['Term']}")
            handle.write("\n")

    print(f"[done] matched {len(matched_edges):,}/{len(predicted_edges):,} edges")
    print(f"[done] outputs written to {outdir}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Ranked scPhyloGRN edge CSV")
    parser.add_argument("--outdir", required=True, type=Path, help="Validation output directory")
    parser.add_argument("--top-n", type=int, default=1000, help="Highest-scoring edges to evaluate")
    parser.add_argument("--top-tfs", type=int, default=3, help="Regulators for per-TF enrichment")
    parser.add_argument("--organism", default="Mouse", help="Enrichr organism name")
    return parser


def main() -> None:
    """Parse command-line arguments and run validation."""
    args = build_parser().parse_args()
    validate_predictions(args.input, args.outdir, args.top_n, args.top_tfs, args.organism)


if __name__ == "__main__":
    main()
