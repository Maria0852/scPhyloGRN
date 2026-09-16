#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train scPhyloGRN and export ranked directed TF–target candidates.

The command consumes a gene-by-cell expression matrix, reference regulatory
edges, and a cell-by-cell lineage kernel aligned to the expression columns.
It constructs expression- and lineage-informed gene graphs, trains the
dual-stream encoder and directed decoder, evaluates held-out candidate edges,
and exports scores for all retained candidates.
"""

import argparse
import gc
import json
import math
import os
import random
from typing import Optional, List, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

gc.collect()
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

# Publication-facing scPhyloGRN entry point.
REPO_ROOT = Path(__file__).resolve().parents[1]

def resolve_input_path(path: Optional[str]) -> Optional[str]:
    """Resolve an input path from the working directory or repository root."""
    if not path:
        return path
    p = os.path.expanduser(path)
    if os.path.isabs(p) or os.path.exists(p):
        return p
    p2 = str(REPO_ROOT / p)
    return p2 if os.path.exists(p2) else p

def resolve_output_dir(path: Optional[str]) -> str:
    """Resolve and return the requested output directory."""
    if not path:
        return str(REPO_ROOT / "results" / "main" / "scPhyloGRN")
    p = os.path.expanduser(path)
    if os.path.isabs(p):
        return p
    return str(REPO_ROOT / p)

# -----------------------------
# Utils
# -----------------------------

def sigmoid_np(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Numerically stable sigmoid with optional temperature."""
    zT = z / max(T, 1e-6)
    out = np.empty_like(zT, dtype=np.float32)
    pos = zT >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-zT[pos]))
    ez = np.exp(zT[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def set_seed(seed: int = 42):
    """Seed Python, NumPy, and PyTorch and request deterministic cuDNN behavior."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_name_list(text: Optional[str]) -> List[str]:
    """Parse a comma-separated list, removing surrounding whitespace."""
    vals = []
    for tok in str(text or "").split(","):
        tok = tok.strip()
        if tok:
            vals.append(tok)
    return vals


def load_name_list(text: Optional[str], path: Optional[str]) -> List[str]:
    """Combine names supplied inline and by file while preserving order."""
    vals = parse_name_list(text)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                tok = line.strip()
                if tok:
                    vals.append(tok)
    out = []
    seen = set()
    for x in vals:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_expression(path: str) -> pd.DataFrame:
    """Load and sanitize a gene-by-cell expression matrix from CSV."""
    df = pd.read_csv(path)
    if df.shape[1] > 1 and not np.issubdtype(df.iloc[:,0].dtype, np.number):
        df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").fillna(0.0)
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]
    return df


def read_ref_edges(path: str) -> pd.DataFrame:
    """Load directed reference edges and standardize columns to gene_i/gene_j."""
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    for a,b in [('gene_i','gene_j'),('tf','target'),('src','dst'),('gene1','gene2')]:
        if a in cols and b in cols:
            return df[[a,b]].rename(columns={a:'gene_i', b:'gene_j'}).astype(str)
    out = df.iloc[:, :2].copy(); out.columns = ['gene_i','gene_j']; return out.astype(str)


def align_genes(X_df: pd.DataFrame, ref_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict expression and reference edges to their shared gene set."""
    genes = X_df.index.astype(str)
    m = ref_df['gene_i'].isin(genes) & ref_df['gene_j'].isin(genes)
    ref2 = ref_df[m].drop_duplicates().copy()
    keep = sorted(set(ref2['gene_i']) | set(ref2['gene_j']))
    return X_df.loc[keep], ref2


def zscore_genes(X: torch.Tensor) -> torch.Tensor:
    """Standardize every gene across cells, returning a tensor shaped G×C."""
    mu = X.mean(dim=1, keepdim=True)
    sd = X.std(dim=1, keepdim=True).clamp_min(1e-8)
    return (X - mu) / sd


def local_topk(S: torch.Tensor, k: int) -> torch.Tensor:
    """Convert a square similarity matrix into a symmetric top-k adjacency."""
    G = S.size(0); k = min(int(k), G)
    idx = torch.topk(S, k=k, dim=1, largest=True, sorted=False).indices
    A = torch.zeros_like(S)
    rows = torch.arange(G, device=S.device).unsqueeze(1).expand_as(idx)
    A[rows, idx] = 1.0
    A.fill_diagonal_(0.0)
    A = torch.maximum(A, A.T)
    return A


def gcn_norm(A_bin: torch.Tensor) -> torch.Tensor:
    """Apply symmetric GCN normalization after adding self-loops."""
    G = A_bin.size(0)
    A_hat = A_bin.clone(); A_hat.fill_diagonal_(1.0)
    deg = A_hat.sum(dim=1).clamp_min(1e-8)
    D_inv_sqrt = torch.pow(deg, -0.5)
    return (A_hat * D_inv_sqrt.view(G,1)) * D_inv_sqrt.view(1,G)


def pearson_topk_graph(X: torch.Tensor, k: int):
    """Construct normalized and binary top-k Pearson gene graphs."""
    Z = zscore_genes(X)
    S = (Z @ Z.T) / X.size(1)
    S = torch.clamp(S, min=0)
    A_bin = local_topk(S, k)
    return gcn_norm(A_bin), A_bin


def lineage_corr_topk(X: torch.Tensor, K: torch.Tensor, k: int):
    """Construct normalized and binary lineage-weighted gene graphs."""
    XK = X @ K; N = XK @ X.T
    denom = ((XK * X).sum(dim=1).clamp_min(1e-8).sqrt().view(-1,1) * ((XK * X).sum(dim=1).clamp_min(1e-8).sqrt().view(1,-1)))
    S = (N / denom).nan_to_num(0.0)
    S = torch.clamp(S, min=0)
    A_bin = local_topk(S, k); return gcn_norm(A_bin), A_bin


def lineage_smooth_multi(X: torch.Tensor, K: Optional[torch.Tensor], scales: List[int], alpha: float, restandardize: bool) -> torch.Tensor:
    """Blend expression with lineage-kernel smoothing over one or more scales."""
    if (K is None) or (alpha <= 0) or (not scales): return X
    feats=[]
    for s in scales:
        s = int(s)
        if s < 1: continue
        Kp = K
        for _ in range(s-1): Kp = Kp @ K
        feats.append(X @ Kp)
    Xs = torch.stack(feats, 0).mean(0) if len(feats)>1 else feats[0]
    Xn = (1-alpha)*X + alpha*Xs
    if restandardize:
        mu = Xn.mean(dim=1, keepdim=True); sd = Xn.std(dim=1, keepdim=True).clamp_min(1e-8)
        Xn = (Xn - mu) / sd
    return Xn

# -----------------------------
# Gene-space lineage distance from K
# -----------------------------

@torch.no_grad()
def compute_gene_distance_from_lineage(X: torch.Tensor, K: torch.Tensor,
                                       block: int = 512, cache_path: str = None) -> torch.Tensor:
    """
    Compute D_gene ∈ [0,1] (gene-space lineage distance)
    with auto cache validation and recompute if mismatch.

    Args:
        X : (G×C) expression tensor
        K : (C×C) lineage kernel
        block : chunk size for memory safety
        cache_path : optional .npy cache path
    """
    G, C = X.shape

    # --- Step 1. 尝试加载缓存 ---
    if cache_path and os.path.exists(cache_path):
        try:
            D_cached = np.load(cache_path)
            if D_cached.shape == (G, G):
                print(f"[lineage] load precomputed D_gene from {cache_path}")
                return torch.from_numpy(D_cached).to(X.device)
            else:
                print(f"[lineage] cached D_gene shape {D_cached.shape} != ({G},{G}); recomputing...")
        except Exception as e:
            print(f"[lineage] failed to load cached D_gene ({e}); recomputing...")

    # --- Step 2. 正式计算 ---
    print(f"[lineage] computing D_gene (block={block}) ...")
    Xz = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True).clamp_min(1e-8))
    outs = []
    for s in range(0, G, block):
        e = min(G, s + block)
        XK = Xz[s:e] @ K
        N = XK @ Xz.T
        denom = (
            (XK * Xz[s:e]).sum(dim=1).clamp_min(1e-8).sqrt().view(-1, 1)
            * ((Xz @ K * Xz).sum(dim=1).clamp_min(1e-8).sqrt().view(1, -1))
        )
        corrK = (N / denom).nan_to_num(0.0).clamp(-1.0, 1.0)
        outs.append(1.0 - corrK.abs())
        torch.cuda.empty_cache()

    D = torch.cat(outs, dim=0)
    D = D / (D.max() + 1e-8)

    # --- Step 3. 保存缓存 ---
    if cache_path:
        try:
            np.save(cache_path, D.cpu().numpy())
            print(f"[lineage] saved D_gene to {cache_path}")
        except Exception as e:
            print(f"[warn] failed to save D_gene cache ({e})")

    return D


# -----------------------------
# scPhyloGRN attention and encoder blocks
# -----------------------------

class LineageAwareGraphAttention(nn.Module):
    """Masked multi-head gene attention penalized by lineage distance."""
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1, beta: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim; self.nh = n_heads; self.dk = dim // n_heads
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))

    def forward(self, H: torch.Tensor, A_mask: torch.Tensor, D_gene: Optional[torch.Tensor] = None):
        G, D = H.size()
        q = self.Wq(H).view(G, self.nh, self.dk)
        k = self.Wk(H).view(G, self.nh, self.dk)
        v = self.Wv(H).view(G, self.nh, self.dk)
        scores = torch.einsum('ihd,jhd->hij', q, k) / math.sqrt(self.dk)
        M = (A_mask > 0); M = M | torch.eye(G, device=H.device, dtype=torch.bool)
        scores = scores.masked_fill(~M.unsqueeze(0), float('-inf'))
        if D_gene is not None:
            scores = scores - torch.clamp(self.beta, min=0.0) * D_gene.unsqueeze(0)
        att = torch.softmax(scores, dim=-1)
        att = self.dropout(att)
        ctx = torch.einsum('hij,jhd->ihd', att, v).contiguous().view(G, D)
        return self.out(ctx)

class LineageAwareTransformerBlock(nn.Module):
    """Transformer block combining lineage-aware attention and an MLP."""
    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1, beta: float = 0.0):
        super().__init__()
        self.attn = LineageAwareGraphAttention(dim, n_heads=n_heads, dropout=dropout, beta=beta)
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        h = int(dim*mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim,h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h,dim), nn.Dropout(dropout))
    def forward(self, H, A_mask, D_gene=None):
        x = H + self.attn(self.ln1(H), A_mask, D_gene)
        x = x + self.ffn(self.ln2(x)); return x

class scPhyloGRNEncoder(nn.Module):
    """Dual-stream encoder for raw and lineage-informed gene graphs."""
    def __init__(self, in_dim: int, proj_dim: int = 128, out_dim: int = 128, n_heads: int = 4, depth: int = 2, dropout: float = 0.1, beta: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim, bias=False)
        self.raw_blocks = nn.ModuleList([LineageAwareTransformerBlock(proj_dim, n_heads=n_heads, dropout=dropout, beta=0.0) for _ in range(depth)])
        self.lin_blocks = nn.ModuleList([LineageAwareTransformerBlock(proj_dim, n_heads=n_heads, dropout=dropout, beta=beta) for _ in range(depth)])
        self.gate = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(proj_dim, out_dim, bias=True)
    def forward(self, X_in, A_raw_mask, A_lin_mask=None, D_gene=None):
        H0 = self.proj(X_in)
        H_raw = H0
        for blk in self.raw_blocks: H_raw = blk(H_raw, A_raw_mask, None)
        if A_lin_mask is None:
            H = H_raw
        else:
            H_lin = H0
            for blk in self.lin_blocks: H_lin = blk(H_lin, A_lin_mask, D_gene)
            g = torch.sigmoid(self.gate); H = g*H_raw + (1-g)*H_lin
        return self.out(H)

# -----------------------------
# scPhyloGRN decoder
# -----------------------------

class scPhyloGRNDecoder(nn.Module):
    """Directed edge decoder with bilinear and element-wise interactions."""
    def __init__(self, emb_dim: int, hid: int = 256, dropout: float = 0.3, eps_cross: float = 1e-3):
        super().__init__()
        self.Wb = nn.Parameter(torch.randn(emb_dim, emb_dim) / math.sqrt(emb_dim))
        self.eps_cross = eps_cross
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*4 + 1, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid//2, 1)
        )
    def forward(self, H: torch.Tensor, edges_ij: torch.Tensor) -> torch.Tensor:
        i = edges_ij[:,0]; j = edges_ij[:,1]
        hi = H[i]; hj = H[j]
        # safe bilinear: hiᵀ W hj = sum(hi * (Wᵀ hj))
        bilinear_core = torch.sum(hi * (hj @ self.Wb.T), dim=1, keepdim=True)
        if self.eps_cross > 0:
            # tiny cross-noise to mimic implicit regularization from big matmul
            noise = self.eps_cross * torch.sum((hi @ self.Wb) * hj, dim=1, keepdim=True)
            bilinear = bilinear_core + noise
        else:
            bilinear = bilinear_core
        z = torch.cat([hi, hj, hi-hj, hi*hj, bilinear], dim=1)
        return self.mlp(z).view(-1)


class scPhyloGRNDecoderNoBilinear(nn.Module):
    """Ablation decoder that omits the bilinear interaction term."""
    def __init__(self, emb_dim: int, hid: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid // 2, 1)
        )

    def forward(self, H: torch.Tensor, edges_ij: torch.Tensor) -> torch.Tensor:
        i = edges_ij[:, 0]; j = edges_ij[:, 1]
        hi = H[i]; hj = H[j]
        z = torch.cat([hi, hj, hi - hj, hi * hj], dim=1)
        return self.mlp(z).view(-1)


class scPhyloGRNDecoderLinear(nn.Module):
    """Linear directed-edge decoder used for ablation experiments."""
    def __init__(self, emb_dim: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim * 2, 1)

    def forward(self, H: torch.Tensor, edges_ij: torch.Tensor) -> torch.Tensor:
        i = edges_ij[:, 0]; j = edges_ij[:, 1]
        hi = H[i]; hj = H[j]
        z = torch.cat([hi, hj], dim=1)
        return self.fc(z).view(-1)

# -----------------------------
# Priors & helpers
# -----------------------------

def direction_prior_loss(edges_b: torch.Tensor, logits: torch.Tensor, X_raw: torch.Tensor,
                         t_vec: torch.Tensor, top_p: float, margin: float) -> torch.Tensor:
    """Penalize high-confidence edges that oppose expression pseudotime order."""
    if t_vec is None or edges_b.numel() == 0:
        return torch.zeros((), device=X_raw.device)
    p = max(1, int(edges_b.size(0) * top_p))
    sel = torch.topk(logits.detach(), k=p, largest=True).indices
    ij = edges_b[sel]
    i_idx, j_idx = ij[:,0], ij[:,1]
    t0 = (t_vec - t_vec.mean()) / (t_vec.std().clamp_min(1e-8))
    xi = (X_raw[i_idx] - X_raw[i_idx].mean(dim=1, keepdim=True)) / (X_raw[i_idx].std(dim=1, keepdim=True).clamp_min(1e-8))
    xj = (X_raw[j_idx] - X_raw[j_idx].mean(dim=1, keepdim=True)) / (X_raw[j_idx].std(dim=1, keepdim=True).clamp_min(1e-8))
    rho_i = (xi * t0).sum(dim=1) / (xi.norm(dim=1) * t0.norm())
    rho_j = (xj * t0).sum(dim=1) / (xj.norm(dim=1) * t0.norm())
    return F.relu(rho_j - rho_i + margin).mean()

@torch.no_grad()
def pairwise_corr_for_edges(X: torch.Tensor, edges_ij: torch.Tensor, K: Optional[torch.Tensor]):
    """Calculate ordinary or lineage-weighted correlation for selected edges."""
    i = edges_ij[:,0]; j = edges_ij[:,1]
    xi = X[i]; xj = X[j]
    if K is None:
        xi = (xi - xi.mean(dim=1, keepdim=True)) / (xi.std(dim=1, keepdim=True).clamp_min(1e-8))
        xj = (xj - xj.mean(dim=1, keepdim=True)) / (xj.std(dim=1, keepdim=True).clamp_min(1e-8))
        num = (xi * xj).sum(dim=1); den = xi.norm(dim=1) * xj.norm(dim=1)
        return (num / den).nan_to_num(0.0).clamp(-1.0, 1.0)
    else:
        xiK = xi @ K
        num = (xiK * xj).sum(dim=1)
        den = ((xiK * xi).sum(dim=1).clamp_min(1e-8).sqrt() * (((xj @ K) * xj).sum(dim=1).clamp_min(1e-8).sqrt()))
        return (num / den).nan_to_num(0.0).clamp(-1.0, 1.0)

# -----------------------------
# Data builders
# -----------------------------

def build_candidate_edges(A_bin: torch.Tensor) -> np.ndarray:
    """Convert an undirected adjacency matrix into both directed orientations."""
    i,j = torch.nonzero(torch.triu(A_bin, diagonal=1), as_tuple=True)
    und = torch.stack([i,j],1).cpu().numpy()
    src = np.concatenate([und[:,0], und[:,1]]); dst = np.concatenate([und[:,1], und[:,0]])
    return np.stack([src,dst],1)

def make_labels(edges: np.ndarray, ref_df: pd.DataFrame, gene_names: List[str]) -> np.ndarray:
    """Label directed candidates using the supplied reference edge set."""
    idx = {g:i for i,g in enumerate(gene_names)}
    pos = set((idx[a], idx[b]) for a,b in zip(ref_df['gene_i'], ref_df['gene_j']) if a in idx and b in idx)
    return np.array([(i,j) in pos for i,j in edges], dtype=np.float32)

def split_indices(n: int, seed: int, ratios=(0.7, 0.1, 0.2), y: Optional[np.ndarray] = None):
    """Return reproducible train, validation, and test indices.

    When labels are supplied, stratification is used so that binary metrics are
    defined in each partition. A clear error is raised when the data set is too
    small to support that split.
    """
    idx = np.arange(n)
    if y is None:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    try:
        tr, held = train_test_split(
            idx, test_size=ratios[1] + ratios[2], random_state=seed, stratify=y
        )
        val_fraction = ratios[1] / (ratios[1] + ratios[2])
        va, te = train_test_split(
            held, test_size=1.0 - val_fraction, random_state=seed + 1, stratify=y[held]
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create stratified train/validation/test partitions. "
            "Provide more positive and negative candidate edges or use a larger k_top."
        ) from exc
    return np.asarray(tr), np.asarray(va), np.asarray(te)


def _has_both_classes(y: np.ndarray, idx: np.ndarray) -> bool:
    """Return whether a subset contains positive and negative examples."""
    if idx.size == 0:
        return False
    vals = np.unique(y[idx])
    return vals.size >= 2


def split_by_tf(edges_keep: np.ndarray, seed: int, ratios=(0.7, 0.1, 0.2),
                y_keep: Optional[np.ndarray] = None, max_tries: int = 128):
    """Split edges by source TF, falling back when class balance is impossible."""
    rng = np.random.RandomState(seed)
    tfs = np.unique(edges_keep[:, 0])
    n = len(tfs)
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    order = tfs.copy()
    for _ in range(max_tries):
        rng.shuffle(order)
        tf_tr = set(order[:n_tr])
        tf_va = set(order[n_tr:n_tr + n_va])
        tf_te = set(order[n_tr + n_va:])
        tr = np.where(np.isin(edges_keep[:, 0], list(tf_tr)))[0]
        va = np.where(np.isin(edges_keep[:, 0], list(tf_va)))[0]
        te = np.where(np.isin(edges_keep[:, 0], list(tf_te)))[0]
        if y_keep is None:
            return tr, va, te
        if _has_both_classes(y_keep, tr) and _has_both_classes(y_keep, va) and _has_both_classes(y_keep, te):
            return tr, va, te
    print(f"[split] warning: tf_holdout could not find fully mixed splits after {max_tries} tries; falling back to random split")
    return split_indices(len(edges_keep), seed=seed, ratios=ratios, y=y_keep)

def sample_negatives(edges: np.ndarray, y: np.ndarray, ratio: float, seed: int):
    """Retain all positives and sample negatives at the requested ratio."""
    rng = np.random.RandomState(seed); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    n_neg=int(len(pos)*max(0.0,ratio)); pick = neg if n_neg>=len(neg) else rng.choice(neg, size=n_neg, replace=False)
    keep=np.concatenate([pos,pick]); rng.shuffle(keep); return keep


def sample_negatives_hard_streaming(
    edges: np.ndarray,
    y: np.ndarray,
    X_raw: torch.Tensor,
    K: Optional[torch.Tensor],
    ratio: float,
    seed: int,
    chunk_size: int = 20000,
    pool_mult: int = 3,
):
    """Sample high-correlation negative edges without materializing all scores."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = int(n_pos * max(0.0, ratio))
    if n_neg <= 0:
        keep = pos_idx.copy()
        rng.shuffle(keep)
        return keep

    pool_size = int(max(n_neg * pool_mult, 1000))
    device = X_raw.device
    pool_sims = torch.empty((0,), device="cpu", dtype=torch.float32)
    pool_ids = torch.empty((0,), device="cpu", dtype=torch.int64)

    def update_pool(new_sims_cpu: np.ndarray, new_ids_cpu: np.ndarray):
        nonlocal pool_sims, pool_ids
        pool_sims = torch.cat([pool_sims, torch.from_numpy(new_sims_cpu.astype(np.float32))], dim=0)
        pool_ids = torch.cat([pool_ids, torch.from_numpy(new_ids_cpu.astype(np.int64))], dim=0)
        if pool_sims.numel() > pool_size:
            topv, topi = torch.topk(pool_sims, k=pool_size, largest=True, sorted=False)
            pool_sims = topv
            pool_ids = pool_ids[topi]

    for st in range(0, len(neg_idx), chunk_size):
        ed = min(len(neg_idx), st + chunk_size)
        sel = neg_idx[st:ed]
        e_chunk = torch.from_numpy(edges[sel].astype(np.int64)).to(device)
        with torch.no_grad():
            sim_chunk = pairwise_corr_for_edges(X_raw, e_chunk, K).abs()
        update_pool(sim_chunk.detach().cpu().numpy(), sel.astype(np.int64))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    hard_ids = pool_ids.numpy()
    if len(hard_ids) == 0:
        pick = neg_idx if n_neg >= len(neg_idx) else rng.choice(neg_idx, size=n_neg, replace=False)
    elif n_neg >= len(hard_ids):
        pick = hard_ids
    else:
        pick = rng.choice(hard_ids, size=n_neg, replace=False)
    keep = np.concatenate([pos_idx, pick])
    rng.shuffle(keep)
    return keep

# -----------------------------
# Calibration & ECE
# -----------------------------

class _TempScaler(nn.Module):
    """Learn a positive scalar temperature for post-hoc calibration."""
    def __init__(self, init_T: float = 1.0):
        super().__init__(); self.logT = nn.Parameter(torch.tensor([math.log(init_T+1e-8)], dtype=torch.float32))
    def forward(self, z): T=torch.exp(self.logT).clamp(1e-4,100.0); return z/T

def temperature_calibrate(logits_val: np.ndarray, y_val: np.ndarray, max_iter=200) -> float:
    """Fit a scalar temperature on validation logits."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler=_TempScaler(1.0).to(device); opt=torch.optim.LBFGS(scaler.parameters(), lr=0.5, max_iter=max_iter, line_search_fn='strong_wolfe')
    xv=torch.from_numpy(logits_val.astype(np.float32)).to(device); yv=torch.from_numpy(y_val.astype(np.float32)).to(device)
    bce=nn.BCEWithLogitsLoss()
    def closure(): opt.zero_grad(); loss=bce(scaler(xv).view(-1), yv.view(-1)); loss.backward(); return loss
    opt.step(closure); return float(torch.exp(scaler.logT).item())

def ece(probs: np.ndarray, y: np.ndarray, n_bins=15):
    """Compute expected calibration error over equal-width probability bins."""
    bins=np.linspace(0,1,n_bins+1); e=0.0; N=len(probs)
    for i in range(n_bins):
        m=(probs>=bins[i])&(probs<bins[i+1])
        if m.sum()==0: continue
        e+=abs(probs[m].mean()-y[m].mean())*(m.sum()/N)
    return float(e)

# -----------------------------
# CLI with stable defaults
# -----------------------------

def build_parser():
    """Create the command-line parser for training and inference."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expr', required=True, help='Gene-by-cell expression CSV; first column contains gene names.')
    p.add_argument('--ref', required=True, help='Reference edge CSV containing gene_i and gene_j columns.')
    p.add_argument('--config', default=None)
    p.add_argument('--outdir', default='results/main/scPhyloGRN', help='Directory for predictions and cached distances.')
    # lineage kernels, smoothing, graphs
    p.add_argument('--lineage-k', '--lineage_k_path', dest='lineage_k_path', required=True,
                   help='Cell-by-cell lineage kernel (.npy), ordered as expression columns.')
    p.add_argument('--lineage_depth_path', type=str, default=None)
    p.add_argument('--depth_encoding', choices=['none','sin'], default='sin')
    p.add_argument('--beta_lineage_att', type=float, default=0.2)
    p.add_argument('--smooth_scales', type=str, default='1,2')
    p.add_argument('--smooth_alpha', type=float, default=0.2)
    p.add_argument('--restandardize_after_smooth', action='store_true')
    p.add_argument('--k_top', type=int, default=300)
    p.add_argument('--graph_from', choices=['raw','smooth','union'], default='union')
    # model dims
    p.add_argument('--proj_dim', type=int, default=128)
    p.add_argument('--out_dim', type=int, default=128)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    # opt
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--batch_size', type=int, default=8192)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--neg_ratio', type=float, default=1.0)
    p.add_argument('--neg_sampler', choices=['random', 'hard'], default='random')
    p.add_argument('--hard_chunk_size', type=int, default=20000)
    p.add_argument('--hard_pool_mult', type=int, default=3)
    p.add_argument('--alpha_eval', type=float, default=1.0)
    p.add_argument('--class_balance', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split_mode', choices=['random', 'tf_holdout'], default='random')
    p.add_argument('--exclude_source_tfs', type=str, default='')
    p.add_argument('--exclude_source_tfs_file', type=str, default='')
    # regularizers
    p.add_argument('--lambda_lineage', type=float, default=0.1)
    p.add_argument('--lc_top_p', type=float, default=0.3)
    p.add_argument('--lambda_dir', type=float, default=0.2)
    p.add_argument('--dir_margin', type=float, default=0.05)
    p.add_argument('--dir_top_p', type=float, default=0.3)
    p.add_argument('--lineage_time_path', type=str, default=None)
    p.add_argument('--export_explain', action='store_true')
    p.add_argument('--raw_only_encoder', action='store_true')
    p.add_argument('--no_bilinear', action='store_true')
    p.add_argument('--linear_decoder', action='store_true')
    p.add_argument('--no_calibration', action='store_true')
    p.add_argument('--model_select', choices=['auprc', 'mean', 'min'], default='auprc')
    return p


def maybe_load_config(args):
    """Overlay arguments with keys from an optional JSON configuration file."""
    cfg_path = resolve_input_path(args.config)
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path,'r') as f: cfg=json.load(f)
        for k,v in cfg.items():
            if hasattr(args,k): setattr(args,k,v)
        args.config = cfg_path
        print(f"[config] loaded {args.config}")


def parse_smooth_scales(text: str) -> List[int]:
    """Parse a comma-separated sequence of positive smoothing powers."""
    vals = []
    for t in str(text).split(','):
        t = t.strip()
        if not t:
            continue
        try:
            v = int(t)
            if v > 0:
                vals.append(v)
        except ValueError:
            continue
    return vals

# -----------------------------
# Main
# -----------------------------

def main():
    """Run the complete scPhyloGRN training, evaluation, and export workflow."""
    parser = build_parser(); args = parser.parse_args(); maybe_load_config(args)
    args.expr = resolve_input_path(args.expr)
    args.ref = resolve_input_path(args.ref)
    args.lineage_k_path = resolve_input_path(args.lineage_k_path)
    args.lineage_depth_path = resolve_input_path(args.lineage_depth_path)
    args.lineage_time_path = resolve_input_path(args.lineage_time_path)
    args.exclude_source_tfs_file = resolve_input_path(args.exclude_source_tfs_file)
    args.outdir = resolve_output_dir(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    print('[config] using scPhyloGRN default configuration')
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) data load
    if not os.path.exists(args.expr):
        raise FileNotFoundError(f"expr not found: {args.expr}")
    if not os.path.exists(args.ref):
        raise FileNotFoundError(f"ref not found: {args.ref}")
    if not args.lineage_k_path:
        raise ValueError("lineage_k_path is required for scPhyloGRN.")
    if not os.path.exists(args.lineage_k_path):
        raise FileNotFoundError(f"lineage_k_path not found: {args.lineage_k_path}")
    X_df = read_expression(args.expr)
    ref_df = read_ref_edges(args.ref)
    X_df, ref_df = align_genes(X_df, ref_df)
    genes = X_df.index.astype(str).tolist(); G, C = X_df.shape
    print(f"[data] genes={G}, cells={C}, positives(after align)={len(ref_df)}")
    X_raw = torch.from_numpy(X_df.values.astype(np.float32)).to(device)
    smooth_scales = parse_smooth_scales(args.smooth_scales)
    print(
        f"[ablation] graph_from={args.graph_from} smooth_alpha={args.smooth_alpha} "
        f"smooth_scales={smooth_scales} "
        f"beta_lineage_att={args.beta_lineage_att} lambda_lineage={args.lambda_lineage} "
        f"class_balance={args.class_balance} neg_ratio={args.neg_ratio} "
        f"neg_sampler={args.neg_sampler} split_mode={args.split_mode} "
        f"raw_only_encoder={args.raw_only_encoder} no_bilinear={args.no_bilinear} "
        f"linear_decoder={args.linear_decoder} model_select={args.model_select}"
    )

    # 2) lineage matrices
    print(f"[lineage] load K from {args.lineage_k_path}")
    K_array = np.load(args.lineage_k_path)
    if isinstance(K_array, np.lib.npyio.NpzFile):
        if "K" not in K_array.files:
            raise ValueError("A lineage .npz archive must contain an array named 'K'.")
        K_array = K_array["K"]
    K_array = np.asarray(K_array, dtype=np.float32)
    if K_array.ndim != 2 or K_array.shape[0] != K_array.shape[1]:
        raise ValueError(f"The lineage kernel must be square; received shape {K_array.shape}.")
    if K_array.shape[0] != C:
        raise ValueError(
            f"Lineage kernel size {K_array.shape[0]} does not match the {C} expression columns."
        )
    if not np.isfinite(K_array).all() or (K_array < 0).any():
        raise ValueError("The lineage kernel must contain finite, non-negative values.")
    if (K_array.sum(axis=1) <= 0).any():
        raise ValueError("Every lineage-kernel row must have a positive sum.")
    K = torch.from_numpy(K_array).to(device)
    K = K / K.sum(dim=1, keepdim=True).clamp_min(1e-8)
    D_gene = compute_gene_distance_from_lineage(
        X_raw,
        K,
        block=512,
        cache_path=os.path.join(args.outdir, 'D_gene_cached.npy'),
    )

    # 3) smoothing + graph build
    can_use_smooth = (args.smooth_alpha > 0.0) and (len(smooth_scales) > 0)
    X_model = X_raw
    if can_use_smooth:
        X_model = lineage_smooth_multi(
            X_raw, K, smooth_scales, alpha=args.smooth_alpha,
            restandardize=args.restandardize_after_smooth
        )
        print(f"[smooth] enabled; alpha={args.smooth_alpha}, scales={smooth_scales}")
    else:
        if args.graph_from in ("smooth", "union"):
            print("[smooth] disabled by configuration; using raw graph")

    print(f"[graph] building Top-{args.k_top} Pearson graphs ...")
    _, A_raw_bin = pearson_topk_graph(X_raw, args.k_top)
    A_smooth_bin = None
    if can_use_smooth:
        _, A_smooth_bin = pearson_topk_graph(X_model, args.k_top)

    if args.graph_from == "raw":
        A_main_bin = A_raw_bin
    elif args.graph_from == "smooth":
        A_main_bin = A_smooth_bin if A_smooth_bin is not None else A_raw_bin
    else:  # union
        A_main_bin = torch.maximum(A_raw_bin, A_smooth_bin) if A_smooth_bin is not None else A_raw_bin

    _, A_lin_bin = lineage_corr_topk(X_raw, K, args.k_top)

    # Candidate edges combine the selected main graph with the mandatory lineage graph.
    A_cand_bin = torch.maximum(A_main_bin, A_lin_bin)
    print(
        f"[graph] edges raw={int(A_raw_bin.sum().item()/2)} "
        f"smooth={int(A_smooth_bin.sum().item()/2) if A_smooth_bin is not None else 0} "
        f"lineage={int(A_lin_bin.sum().item()/2)} "
        f"cand={int(A_cand_bin.sum().item()/2)}"
    )

    # 4) build candidate edges & labels
    edges_cand = build_candidate_edges(A_cand_bin)
    y = make_labels(edges_cand, ref_df, genes)
    pos_covered = int(y.sum())
    pos_total = int(len(ref_df))
    pos_cov_rate = float(pos_covered / (pos_total + 1e-8))
    print(f"[cand] edges={len(edges_cand)} | pos_covered={pos_covered}/{pos_total} (coverage={pos_cov_rate:.4f})")

    excluded_source_tfs = load_name_list(args.exclude_source_tfs, args.exclude_source_tfs_file)
    if excluded_source_tfs:
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        excluded_idx = sorted(gene_to_idx[tf] for tf in excluded_source_tfs if tf in gene_to_idx)
        missing_tfs = [tf for tf in excluded_source_tfs if tf not in gene_to_idx]
        if missing_tfs:
            print(f"[holdout] warning: excluded source TFs not found after align: {missing_tfs}")
        if excluded_idx:
            excluded_mask = np.isin(edges_cand[:, 0], np.array(excluded_idx, dtype=np.int64))
            n_excluded_edges = int(excluded_mask.sum())
            n_excluded_pos = int(y[excluded_mask].sum())
            print(
                f"[holdout] excluding supervision for source TFs={excluded_source_tfs} "
                f"| edges={n_excluded_edges} positives={n_excluded_pos}"
            )
            with open(os.path.join(args.outdir, "excluded_source_tfs.txt"), "w", encoding="utf-8") as f:
                for tf in excluded_source_tfs:
                    f.write(tf + "\n")
            edges_supervised = edges_cand[~excluded_mask]
            y_supervised = y[~excluded_mask]
        else:
            print("[holdout] warning: no excluded source TFs remain after gene alignment; supervision unchanged")
            edges_supervised, y_supervised = edges_cand, y
    else:
        edges_supervised, y_supervised = edges_cand, y

    if args.neg_sampler == "hard":
        keep_idx = sample_negatives_hard_streaming(
            edges_supervised, y_supervised, X_raw, K,
            ratio=args.neg_ratio, seed=args.seed,
            chunk_size=args.hard_chunk_size,
            pool_mult=args.hard_pool_mult,
        )
    else:
        keep_idx = sample_negatives(edges_supervised, y_supervised, ratio=args.neg_ratio, seed=args.seed)
    edges_keep, y_keep = edges_supervised[keep_idx], y_supervised[keep_idx]
    if args.split_mode == "tf_holdout":
        tr_idx, va_idx, te_idx = split_by_tf(edges_keep, seed=args.seed, y_keep=y_keep)
    else:
        tr_idx, va_idx, te_idx = split_indices(len(y_keep), seed=args.seed, y=y_keep)

    for split_name, split_idx in (("train", tr_idx), ("validation", va_idx), ("test", te_idx)):
        if not _has_both_classes(y_keep, split_idx):
            raise ValueError(
                f"The {split_name} partition does not contain both classes. "
                "Provide more reference edges or increase k_top."
            )

    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(edges_keep[tr_idx].astype(np.int64)),
                                           torch.from_numpy(y_keep[tr_idx].astype(np.float32)))
    va_ds = torch.utils.data.TensorDataset(torch.from_numpy(edges_keep[va_idx].astype(np.int64)),
                                           torch.from_numpy(y_keep[va_idx].astype(np.float32)))
    te_ds = torch.utils.data.TensorDataset(torch.from_numpy(edges_keep[te_idx].astype(np.int64)),
                                           torch.from_numpy(y_keep[te_idx].astype(np.float32)))
    tr_ldr = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_ldr = torch.utils.data.DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    te_ldr = torch.utils.data.DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)
    print(
        f"[split] train/val/test = {len(tr_ds)}/{len(va_ds)}/{len(te_ds)} "
        f"(pos ratio train={float(y_keep[tr_idx].mean()):.4f})"
    )

    # 5) build model
    A_raw_mask = (A_main_bin > 0).float().to(device)
    A_lin_mask = (A_lin_bin > 0).float().to(device)
    encoder = scPhyloGRNEncoder(in_dim=X_model.size(1), proj_dim=args.proj_dim,
        out_dim=args.out_dim, n_heads=args.heads, depth=args.depth,
        dropout=args.dropout, beta=args.beta_lineage_att).to(device)
    if args.linear_decoder:
        decoder = scPhyloGRNDecoderLinear(emb_dim=args.out_dim).to(device)
    elif args.no_bilinear:
        decoder = scPhyloGRNDecoderNoBilinear(emb_dim=args.out_dim, hid=max(128, 2*args.proj_dim),
            dropout=args.dropout).to(device)
    else:
        decoder = scPhyloGRNDecoder(emb_dim=args.out_dim, hid=max(128, 2*args.proj_dim),
            dropout=args.dropout).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, verbose=True)
    if args.class_balance:
        y_tr = y_keep[tr_idx]
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        pos_weight = torch.tensor(neg / max(pos, 1.0), device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[loss] BCEWithLogits(pos_weight={float(pos_weight.item()):.4f})")
    else:
        bce = nn.BCEWithLogitsLoss()
        print("[loss] BCEWithLogits(pos_weight=None)")

    def encode_once():
        A_lin_mask_eff = None if args.raw_only_encoder else A_lin_mask
        D_gene_eff = None if args.raw_only_encoder else D_gene
        return encoder(X_model, A_raw_mask, A_lin_mask_eff, D_gene_eff)

    # 6) training loop
    def model_select_score(auroc_val, auprc_val):
        if args.model_select == 'mean':
            return 0.5 * (float(auroc_val) + float(auprc_val))
        if args.model_select == 'min':
            return min(float(auroc_val), float(auprc_val))
        return float(auprc_val)

    best = {'val_auprc':-1,'val_auroc':-1,'test_auprc':-1,'test_auroc':-1,'select_score':-1}
    for ep in range(1, args.epochs + 1):
        encoder.train(); decoder.train()
        total_loss = 0.0
        for edges_b, y_b in tr_ldr:
            edges_b, y_b = edges_b.to(device), y_b.to(device)
            H = encode_once()
            logits = decoder(H, edges_b)
            loss = bce(logits, y_b)

            # (5) lineage consistency loss with gradient
            if args.lambda_lineage > 0.0 and (K is not None):
                probs = torch.sigmoid(logits)
                weights = probs / (probs.sum() + 1e-8)
                corrK = pairwise_corr_for_edges(X_raw, edges_b, K)
                lin_loss = ((1.0 - corrK) * weights).mean()
                loss = loss + args.lambda_lineage * lin_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            total_loss += float(loss.item()) * y_b.numel()

        # eval
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            H = encode_once()
            def eval_loader(loader):
                logits_all, y_all = [], []
                for e_b, y_b in loader:
                    e_b, y_b = e_b.to(device), y_b.to(device)
                    z = decoder(H, e_b)
                    logits_all.append(z.cpu().numpy())
                    y_all.append(y_b.cpu().numpy())
                logits = np.concatenate(logits_all)
                y_true = np.concatenate(y_all)
                probs = sigmoid_np(logits)
                return logits, probs, y_true, roc_auc_score(y_true, probs), average_precision_score(y_true, probs)
            log_va, pr_va, y_va, auroc_va, auprc_va = eval_loader(va_ldr)
            log_te, pr_te, y_te, auroc_te, auprc_te = eval_loader(te_ldr)
        print(f"[ep {ep:03d}] loss={total_loss/len(tr_ds):.4f} | VAL AUC={auroc_va:.4f}/{auprc_va:.4f} | TEST AUC={auroc_te:.4f}/{auprc_te:.4f}")
        print(f"[gate] sigmoid(gate)={torch.sigmoid(encoder.gate).item():.4f}")
        select_score = model_select_score(auroc_va, auprc_va)
        if select_score > best['select_score']:
            best.update({
                'val_auprc': auprc_va,
                'val_auroc': auroc_va,
                'test_auprc': auprc_te,
                'test_auroc': auroc_te,
                'select_score': select_score,
            })
        sched.step(auprc_va)

    # (6) single final calibration
    if args.no_calibration:
        T_temp = 1.0
    else:
        print("[calib] estimating temperature T on validation set once ...")
        T_temp = temperature_calibrate(log_va, y_va)
    pr_va_T = sigmoid_np(log_va, T_temp)
    pr_te_T = sigmoid_np(log_te, T_temp)
    ece_va = ece(pr_va_T, y_va)
    ece_te = ece(pr_te_T, y_te)
    print(f"[calib] Temperature T={T_temp:.3f} | ECE val/test={ece_va:.4f}/{ece_te:.4f}")
    print(f"[select] metric={args.model_select} score={best['select_score']:.4f}")
    print(f"[best] VAL AUROC={best['val_auroc']:.4f} AUPRC={best['val_auprc']:.4f} | TEST AUROC={best['test_auroc']:.4f} AUPRC={best['test_auprc']:.4f}")

    # (7) safe export with timestamp
    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"predicted_edges_{timestamp}.csv")

    df_export = pd.DataFrame({
        'gene_i':[genes[i] for i in edges_cand[:,0]],
        'gene_j':[genes[j] for j in edges_cand[:,1]]
    })
    logits_all = []
    with torch.no_grad():
        H_all = encode_once()
        B = 50000
        for st in range(0, len(edges_cand), B):
            ed = min(len(edges_cand), st + B)
            chunk = torch.from_numpy(edges_cand[st:ed].astype(np.int64)).to(device)
            logits_all.append(decoder(H_all, chunk).cpu().numpy())
    logits_all = np.concatenate(logits_all)
    df_export['logit'] = logits_all
    df_export['prob']   = sigmoid_np(logits_all)
    df_export['prob_T'] = sigmoid_np(logits_all, T_temp)
    df_export.to_csv(out_csv, index=False)
    print(f"[export] wrote {out_csv}")

if __name__ == '__main__':
    main()
