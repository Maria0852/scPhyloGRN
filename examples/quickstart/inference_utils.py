"""Self-contained inference helpers for the scPhyloGRN Zenodo demonstration.

The model definitions mirror the architecture used by
``train_grn5_lineageaware_v4.py``. This module deliberately performs no work at
import time. The accompanying notebook enforces execution inside a Slurm
allocation before importing this module.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _zscore_genes(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp_min(1e-8)
    return (x - mean) / std


def _local_topk(similarity: torch.Tensor, k: int) -> torch.Tensor:
    gene_count = similarity.size(0)
    k = min(max(int(k), 1), gene_count)
    indices = torch.topk(
        similarity, k=k, dim=1, largest=True, sorted=False
    ).indices
    adjacency = torch.zeros_like(similarity)
    rows = (
        torch.arange(gene_count, device=similarity.device)
        .unsqueeze(1)
        .expand_as(indices)
    )
    adjacency[rows, indices] = 1.0
    adjacency.fill_diagonal_(0.0)
    return torch.maximum(adjacency, adjacency.T)


def pearson_topk_graph(
    expression: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    standardized = _zscore_genes(expression)
    similarity = (standardized @ standardized.T) / expression.size(1)
    adjacency = _local_topk(torch.clamp(similarity, min=0.0), k)
    return adjacency, adjacency


def lineage_corr_topk(
    expression: torch.Tensor, lineage_kernel: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    xk = expression @ lineage_kernel
    numerator = xk @ expression.T
    scale = (xk * expression).sum(dim=1).clamp_min(1e-8).sqrt()
    denominator = scale.view(-1, 1) * scale.view(1, -1)
    similarity = (numerator / denominator).nan_to_num(0.0)
    adjacency = _local_topk(torch.clamp(similarity, min=0.0), k)
    return adjacency, adjacency


def lineage_smooth_multi(
    expression: torch.Tensor,
    lineage_kernel: Optional[torch.Tensor],
    scales: Sequence[int],
    alpha: float,
    restandardize: bool,
) -> torch.Tensor:
    if lineage_kernel is None or alpha <= 0.0 or not scales:
        return expression
    smoothed = []
    for scale in scales:
        if int(scale) < 1:
            continue
        powered = lineage_kernel
        for _ in range(int(scale) - 1):
            powered = powered @ lineage_kernel
        smoothed.append(expression @ powered)
    if not smoothed:
        return expression
    aggregate = (
        torch.stack(smoothed, dim=0).mean(dim=0)
        if len(smoothed) > 1
        else smoothed[0]
    )
    result = (1.0 - alpha) * expression + alpha * aggregate
    return _zscore_genes(result) if restandardize else result


def gene_lineage_distance(
    expression: torch.Tensor, lineage_kernel: torch.Tensor
) -> torch.Tensor:
    standardized = _zscore_genes(expression)
    xk = standardized @ lineage_kernel
    numerator = xk @ standardized.T
    left = (xk * standardized).sum(dim=1).clamp_min(1e-8).sqrt()
    right = (
        (standardized @ lineage_kernel * standardized)
        .sum(dim=1)
        .clamp_min(1e-8)
        .sqrt()
    )
    correlation = (
        numerator / (left.view(-1, 1) * right.view(1, -1))
    ).nan_to_num(0.0).clamp(-1.0, 1.0)
    distance = 1.0 - correlation.abs()
    return distance / distance.max().clamp_min(1e-8)


class GraphMHALineage(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        beta: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        # Attribute names intentionally match train_grn5_lineageaware_v4.py so
        # that its state_dict can be loaded with strict=True.
        self.dim = dim
        self.nh = n_heads
        self.dk = dim // n_heads
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency_mask: torch.Tensor,
        gene_distance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gene_count, dim = hidden.shape
        query = self.Wq(hidden).view(gene_count, self.nh, self.dk)
        key = self.Wk(hidden).view(gene_count, self.nh, self.dk)
        value = self.Wv(hidden).view(gene_count, self.nh, self.dk)
        scores = (
            torch.einsum("ihd,jhd->hij", query, key)
            / math.sqrt(self.dk)
        )
        mask = adjacency_mask > 0
        mask = mask | torch.eye(
            gene_count, dtype=torch.bool, device=hidden.device
        )
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        if gene_distance is not None:
            scores = (
                scores
                - torch.clamp(self.beta, min=0.0)
                * gene_distance.unsqueeze(0)
            )
        attention = self.dropout(torch.softmax(scores, dim=-1))
        context = (
            torch.einsum("hij,jhd->ihd", attention, value)
            .contiguous()
            .view(gene_count, dim)
        )
        return self.out(context)


class GTBlockLineage(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.attn = GraphMHALineage(dim, n_heads, dropout, beta)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * 4.0)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency_mask: torch.Tensor,
        gene_distance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden = hidden + self.attn(
            self.ln1(hidden), adjacency_mask, gene_distance
        )
        return hidden + self.ffn(self.ln2(hidden))


class DualGraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        out_dim: int,
        n_heads: int,
        depth: int,
        dropout: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim, bias=False)
        self.raw_blocks = nn.ModuleList(
            [
                GTBlockLineage(proj_dim, n_heads, dropout, beta=0.0)
                for _ in range(depth)
            ]
        )
        self.lin_blocks = nn.ModuleList(
            [
                GTBlockLineage(proj_dim, n_heads, dropout, beta=beta)
                for _ in range(depth)
            ]
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(proj_dim, out_dim, bias=True)

    def forward(
        self,
        expression: torch.Tensor,
        raw_mask: torch.Tensor,
        lineage_mask: Optional[torch.Tensor],
        gene_distance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        initial = self.proj(expression)
        raw_hidden = initial
        for block in self.raw_blocks:
            raw_hidden = block(raw_hidden, raw_mask, None)
        if lineage_mask is None:
            hidden = raw_hidden
        else:
            lineage_hidden = initial
            for block in self.lin_blocks:
                lineage_hidden = block(
                    lineage_hidden, lineage_mask, gene_distance
                )
            gate = torch.sigmoid(self.gate)
            hidden = gate * raw_hidden + (1.0 - gate) * lineage_hidden
        return self.out(hidden)


class StrongDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        eps_cross: float = 1e-3,
    ) -> None:
        super().__init__()
        self.Wb = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim)
            / math.sqrt(embedding_dim)
        )
        self.eps_cross = eps_cross
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, hidden: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        source = hidden[edges[:, 0]]
        target = hidden[edges[:, 1]]
        bilinear_core = torch.sum(
            source * (target @ self.Wb.T),
            dim=1,
            keepdim=True,
        )
        bilinear = bilinear_core
        if self.eps_cross > 0:
            bilinear = bilinear + self.eps_cross * torch.sum(
                (source @ self.Wb) * target,
                dim=1,
                keepdim=True,
            )
        features = torch.cat(
            [
                source,
                target,
                source - target,
                source * target,
                bilinear,
            ],
            dim=1,
        )
        return self.mlp(features).view(-1)


class DecoderNoBilinear(nn.Module):
    def __init__(
        self, embedding_dim: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, hidden: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        source = hidden[edges[:, 0]]
        target = hidden[edges[:, 1]]
        features = torch.cat(
            [source, target, source - target, source * target], dim=1
        )
        return self.mlp(features).view(-1)


class DecoderLinear(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(
        self, hidden: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        source = hidden[edges[:, 0]]
        target = hidden[edges[:, 1]]
        return self.fc(torch.cat([source, target], dim=1)).view(-1)


def build_models(
    model_config: Dict[str, object],
) -> Tuple[DualGraphTransformerEncoder, nn.Module]:
    encoder = DualGraphTransformerEncoder(
        in_dim=int(model_config["in_dim"]),
        proj_dim=int(model_config["proj_dim"]),
        out_dim=int(model_config["out_dim"]),
        n_heads=int(model_config["heads"]),
        depth=int(model_config["depth"]),
        dropout=float(model_config["dropout"]),
        beta=float(model_config.get("beta_lineage_att", 0.2)),
    )
    decoder_name = str(model_config.get("decoder", "strong"))
    hidden_dim = int(
        model_config.get(
            "decoder_hidden_dim",
            max(128, 2 * int(model_config["proj_dim"])),
        )
    )
    if decoder_name == "linear":
        decoder: nn.Module = DecoderLinear(int(model_config["out_dim"]))
    elif decoder_name == "no_bilinear":
        decoder = DecoderNoBilinear(
            int(model_config["out_dim"]),
            hidden_dim,
            float(model_config["dropout"]),
        )
    elif decoder_name == "strong":
        decoder = StrongDecoder(
            int(model_config["out_dim"]),
            hidden_dim,
            float(model_config["dropout"]),
        )
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_name}")
    return encoder, decoder


def load_checkpoint(path: Path) -> Dict[str, object]:
    """Load a trusted local checkpoint on CPU.

    PyTorch checkpoints use pickle internally. Do not call this function on
    untrusted files.
    """

    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required = {
        "encoder_state_dict",
        "decoder_state_dict",
        "model_config",
        "preprocessing_config",
        "gene_names",
        "feature_names",
        "temperature",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            "Checkpoint is missing required fields: " + ", ".join(missing)
        )
    return checkpoint


def load_lineage_kernel(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return pd.read_csv(path, index_col=0).to_numpy(dtype=np.float32)


def prepare_inputs(
    expression_df: pd.DataFrame,
    lineage_kernel_array: Optional[np.ndarray],
    checkpoint: Dict[str, object],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    device = device or torch.device("cpu")
    expected_genes = [str(value) for value in checkpoint["gene_names"]]
    expected_features = [
        str(value) for value in checkpoint["feature_names"]
    ]
    expression_df = expression_df.copy()
    expression_df.index = expression_df.index.astype(str)
    expression_df.columns = expression_df.columns.astype(str)
    # Match training read_expression: keep the first occurrence of a gene.
    # Otherwise .loc[expected_genes] expands duplicate rows and corrupts indices.
    expression_df = expression_df[~expression_df.index.duplicated(keep="first")]
    missing_genes = sorted(set(expected_genes).difference(expression_df.index))
    if missing_genes:
        raise ValueError(
            f"Expression matrix is missing {len(missing_genes)} checkpoint genes; "
            f"first missing values: {missing_genes[:5]}"
        )
    if expression_df.columns.tolist() != expected_features:
        raise ValueError(
            "Expression feature order does not match checkpoint feature_names."
        )
    expression_df = expression_df.loc[expected_genes, expected_features]
    expression = torch.from_numpy(
        expression_df.to_numpy(dtype=np.float32)
    ).to(device)
    if expression.shape[1] != int(checkpoint["model_config"]["in_dim"]):
        raise ValueError(
            "Expression feature count does not match model_config['in_dim']."
        )

    config = checkpoint["preprocessing_config"]
    uses_lineage = bool(config.get("uses_lineage", True))
    raw_only_encoder = bool(config.get("raw_only_encoder", False))

    lineage_kernel: Optional[torch.Tensor] = None
    if uses_lineage and lineage_kernel_array is not None:
        lineage_kernel = torch.from_numpy(
            np.asarray(lineage_kernel_array, dtype=np.float32)
        ).to(device)
        expected_shape = (expression.shape[1], expression.shape[1])
        if tuple(lineage_kernel.shape) != expected_shape:
            raise ValueError(
                f"Lineage kernel shape {tuple(lineage_kernel.shape)} does not "
                f"match expected shape {expected_shape}."
            )
        lineage_kernel = lineage_kernel / lineage_kernel.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

    scales = [int(value) for value in config.get("smooth_scales", [])]
    alpha = float(config.get("smooth_alpha", 0.0))
    expression_model = lineage_smooth_multi(
        expression,
        lineage_kernel,
        scales,
        alpha,
        bool(config.get("restandardize_after_smooth", False)),
    )
    k_top = int(config.get("k_top", 10))
    _, raw_adjacency = pearson_topk_graph(expression, k_top)
    smooth_adjacency = None
    if lineage_kernel is not None and alpha > 0.0 and scales:
        _, smooth_adjacency = pearson_topk_graph(
            expression_model, k_top
        )
    graph_from = str(config.get("graph_from", "union"))
    if graph_from == "raw" or smooth_adjacency is None:
        main_adjacency = raw_adjacency
    elif graph_from == "smooth":
        main_adjacency = smooth_adjacency
    elif graph_from == "union":
        main_adjacency = torch.maximum(
            raw_adjacency, smooth_adjacency
        )
    else:
        raise ValueError(f"Unsupported graph_from value: {graph_from}")

    lineage_adjacency = None
    distance = None
    if lineage_kernel is not None:
        _, lineage_adjacency = lineage_corr_topk(
            expression, lineage_kernel, k_top
        )
        distance = gene_lineage_distance(expression, lineage_kernel)
    if raw_only_encoder:
        lineage_adjacency = None
        distance = None
    return expression_model, main_adjacency, lineage_adjacency, distance


def predict_pairs(
    checkpoint: Dict[str, object],
    expression_df: pd.DataFrame,
    lineage_kernel_array: Optional[np.ndarray],
    pairs_df: pd.DataFrame,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    device = device or torch.device("cpu")
    encoder, decoder = build_models(checkpoint["model_config"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    expression, raw_mask, lineage_mask, distance = prepare_inputs(
        expression_df, lineage_kernel_array, checkpoint, device=device
    )
    genes = [str(value) for value in checkpoint["gene_names"]]
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    required_columns = {"gene_i", "gene_j"}
    if not required_columns.issubset(pairs_df.columns):
        raise ValueError("Pairs file must contain gene_i and gene_j columns.")
    missing_pair_genes = sorted(
        {
            str(value)
            for column in ["gene_i", "gene_j"]
            for value in pairs_df[column]
            if str(value) not in gene_to_index
        }
    )
    if missing_pair_genes:
        raise ValueError(
            "Pairs contain genes absent from checkpoint: "
            + ", ".join(missing_pair_genes[:10])
        )
    edge_indices = torch.tensor(
        [
            [gene_to_index[str(source)], gene_to_index[str(target)]]
            for source, target in zip(
                pairs_df["gene_i"], pairs_df["gene_j"]
            )
        ],
        dtype=torch.long,
    ).to(device)
    with torch.no_grad():
        hidden = encoder(
            expression, raw_mask, lineage_mask, distance
        )
        logits = decoder(hidden, edge_indices)
        probabilities = torch.sigmoid(logits)
        temperature = max(float(checkpoint["temperature"]), 1e-6)
        calibrated = torch.sigmoid(logits / temperature)
    result = pairs_df[["gene_i", "gene_j"]].copy()
    result["logit"] = logits.detach().cpu().numpy()
    result["probability"] = probabilities.detach().cpu().numpy()
    result["calibrated_probability"] = calibrated.detach().cpu().numpy()
    return result


def train_toy_checkpoint(
    expression_df: pd.DataFrame,
    lineage_kernel_array: np.ndarray,
    output_path: Path,
    epochs: int = 160,
    seed: int = 42,
) -> Tuple[Dict[str, object], float]:
    """Train a tiny synthetic checkpoint solely for the notebook smoke test."""

    set_seed(seed)
    genes = expression_df.index.astype(str).tolist()
    features = expression_df.columns.astype(str).tolist()
    model_config: Dict[str, object] = {
        "in_dim": len(features),
        "proj_dim": 16,
        "out_dim": 16,
        "heads": 4,
        "depth": 1,
        "dropout": 0.0,
        "beta_lineage_att": 0.2,
        "decoder": "strong",
        "decoder_hidden_dim": 32,
    }
    preprocessing_config: Dict[str, object] = {
        "k_top": 3,
        "graph_from": "union",
        "smooth_scales": [1, 2],
        "smooth_alpha": 0.2,
        "restandardize_after_smooth": False,
        "uses_lineage": True,
        "raw_only_encoder": False,
    }
    metadata_only: Dict[str, object] = {
        "format_version": 1,
        "model_config": model_config,
        "preprocessing_config": preprocessing_config,
        "gene_names": genes,
        "feature_names": features,
        "temperature": 1.0,
        "seed": seed,
        "intended_use": "Technical demonstration only; not biological inference.",
    }
    expression, raw_mask, lineage_mask, distance = prepare_inputs(
        expression_df,
        lineage_kernel_array,
        metadata_only,
        device=torch.device("cpu"),
    )
    encoder, decoder = build_models(model_config)
    edges = torch.tensor(
        [
            [source, target]
            for source in range(len(genes))
            for target in range(len(genes))
            if source != target
        ],
        dtype=torch.long,
    )
    positive_names = {
        ("GENE_A", "GENE_B"),
        ("GENE_C", "GENE_D"),
        ("GENE_E", "GENE_G"),
        ("GENE_H", "GENE_C"),
    }
    labels = torch.tensor(
        [
            float((genes[source], genes[target]) in positive_names)
            for source, target in edges.tolist()
        ],
        dtype=torch.float32,
    )
    positive_count = labels.sum().clamp_min(1.0)
    negative_count = len(labels) - positive_count
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=negative_count / positive_count
    )
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-2,
        weight_decay=1e-4,
    )
    final_loss = math.nan
    for _ in range(int(epochs)):
        encoder.train()
        decoder.train()
        hidden = encoder(
            expression, raw_mask, lineage_mask, distance
        )
        logits = decoder(hidden, edges)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    checkpoint = {
        **metadata_only,
        "encoder_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in encoder.state_dict().items()
        },
        "decoder_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in decoder.state_dict().items()
        },
        "training_epochs": int(epochs),
        "final_training_loss": final_loss,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint, final_loss
