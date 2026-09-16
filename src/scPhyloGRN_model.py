"""Reusable neural-network and lineage utilities for scPhyloGRN."""

from typing import List, Tuple, Optional
import math
import heapq

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== GCN Encoder / Decoder ===============

class GCNLayer(nn.Module):
    """A single GCN layer: H = ReLU( A_norm @ (X @ W) + b ) with Dropout."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, A_norm: torch.Tensor, X: torch.Tensor, activate: bool = True) -> torch.Tensor:
        # X: (G x F)
        Z = A_norm @ (X @ self.lin.weight.T)
        if self.lin.bias is not None:
            Z = Z + self.lin.bias
        if activate:
            Z = F.relu(Z, inplace=False)
            Z = self.dropout(Z)
        return Z


class GCNEncoder(nn.Module):
    """
    Two-layer GCN encoder.
    Args:
        in_dim:  projected feature dim (F)
        hid_dim: hidden dim
        out_dim: output embedding dim (H)
        dropout: dropout rate (applied after ReLU on first layer)
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hid_dim, dropout=dropout, bias=True)
        self.gcn2 = GCNLayer(hid_dim, out_dim, dropout=0.0, bias=True)

    def forward(self, A_norm: torch.Tensor, X_feat: torch.Tensor) -> torch.Tensor:
        H = self.gcn1(A_norm, X_feat, activate=True)
        H = self.gcn2(A_norm, H, activate=False)  # last layer no ReLU
        return H


class DirectedMLPDecoder(nn.Module):
    """
    Directed edge decoder on concatenated features:
      z(i->j) = [h_i, h_j, h_i - h_j, h_i ⊙ h_j]
      logits = MLP(z)
    """
    def __init__(self, in_dim: int, hid_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(Z), inplace=False)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # (E, 1)

# =============== Graph utilities ===============

@torch.no_grad()
def lin_topk_from_scores(S: torch.Tensor, k: int) -> torch.Tensor:
    """
    From a (G x G) similarity matrix S (non-negative), keep row-wise Top-k indices,
    return symmetric 0/1 adjacency (without self-loops).
    """
    G = S.size(0)
    k = min(k, G)
    topk = torch.topk(S, k=k, dim=1, largest=True, sorted=False).indices
    A = torch.zeros_like(S, dtype=torch.float32)
    rows = torch.arange(G, device=S.device).unsqueeze(1).expand_as(topk)
    A[rows, topk] = 1.0
    A.fill_diagonal_(0.0)
    A = torch.maximum(A, A.T)
    return A

def lin_gcn_norm(A: torch.Tensor) -> torch.Tensor:
    """
    Standard GCN normalization: Â = D^{-1/2} (A + I) D^{-1/2}.
    """
    G = A.size(0)
    A_hat = A.clone()
    A_hat.fill_diagonal_(1.0)
    deg = A_hat.sum(dim=1).clamp_min(1e-8)
    D_inv_sqrt = torch.pow(deg, -0.5)
    A_norm = (A_hat * D_inv_sqrt.view(G,1)) * D_inv_sqrt.view(1,G)
    return A_norm

@torch.no_grad()
def topk_pearson_graph(X: torch.Tensor, k_top: int) -> torch.Tensor:
    """
    Build Top-k Pearson graph from X (G x C).
    Returns: GCN-normalized adjacency (dense).
    """
    # Pearson via z-score per gene
    mu = X.mean(dim=1, keepdim=True)
    sd = X.std(dim=1, keepdim=True).clamp_min(1e-8)
    Z = (X - mu) / sd
    S = (Z @ Z.T) / X.size(1)               # cosine on z-scored = Pearson
    S = torch.where(S > 0, S, torch.zeros_like(S))  # keep non-negative
    A_bin = lin_topk_from_scores(S, k_top=k_top)
    return lin_gcn_norm(A_bin)

# =============== Lineage kernel (Newick -> K) ===============

try:
    from Bio import Phylo
    HAVE_BIOPHYLO = True
except Exception:
    HAVE_BIOPHYLO = False

def _build_tree_graph_from_newick(newick_path: str):
    """
    Parse Newick (Bio.Phylo) and build an undirected graph over all clades.
    Returns:
      - adj: dict[node_id] -> list[(nbr_id, weight)]
      - leaf_name_to_node: dict[str] -> node_id
      - node_id_to_name: list[str or None]
    """
    if not HAVE_BIOPHYLO:
        raise RuntimeError("build_lineage_K_from_newick requires biopython. Please `pip install biopython`.")
    tree = Phylo.read(newick_path, "newick")
    # Flatten nodes to list
    nodes = list(tree.find_clades(order="level"))
    node_id = {cl: i for i, cl in enumerate(nodes)}
    node_id_to_name = [ (cl.name if getattr(cl, 'name', None) else None) for cl in nodes ]

    adj = { i: [] for i in range(len(nodes)) }
    leaf_name_to_node = {}

    for cl in nodes:
        i = node_id[cl]
        # record leaves
        if cl.is_terminal() and cl.name:
            leaf_name_to_node[cl.name] = i
        # connect to children
        for ch in cl.clades:
            j = node_id[ch]
            w = ch.branch_length if (getattr(ch, 'branch_length', None) is not None) else 1.0
            w = float(w if w > 0 else 1.0)
            adj[i].append((j, w))
            adj[j].append((i, w))
    return adj, leaf_name_to_node, node_id_to_name

def _dijkstra_topk_leaves(adj, start_id: int, leaf_mask: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """
    From start_id (a leaf), run Dijkstra until we found k nearest leaves (including itself).
    Return list of (leaf_node_id, distance).
    """
    found = []
    visited = set()
    heap = [(0.0, start_id)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited: 
            continue
        visited.add(u)
        if leaf_mask[u]:
            found.append((u, d))
            if len(found) >= k:
                break
        for v, w in adj[u]:
            if v not in visited:
                heapq.heappush(heap, (d + float(w), v))
    return found

def build_lineage_K_from_newick(newick_path: str, cell_names: List[str],
                                tau: float = 4.0, k_top: int = 32) -> np.ndarray:
    """
    Build lineage kernel K (cells x cells) from a Newick tree.
    - If no branch lengths, treat each edge length = 1
    - For each cell a: find top-k nearest leaf neighbors (including itself) by Dijkstra
    - S_ab = exp(-d/τ) for those neighbors; row-normalize => stochastic K

    Returns: K (C x C), np.float32
    """
    adj, leaf_name_to_node, node_id_to_name = _build_tree_graph_from_newick(newick_path)
    C = len(cell_names)
    name_to_idx = {name: i for i, name in enumerate(cell_names)}
    present_leaf_nodes = [leaf_name_to_node.get(n, None) for n in cell_names]
    leaf_mask = np.zeros(len(node_id_to_name), dtype=np.bool_)
    for nid in leaf_name_to_node.values():
        leaf_mask[nid] = True

    K = np.zeros((C, C), dtype=np.float32)

    for cell_idx, nid in enumerate(present_leaf_nodes):
        # cell not in tree -> identity row
        if nid is None:
            K[cell_idx, cell_idx] = 1.0
            continue
        # nearest k leaves (include self)
        topk = _dijkstra_topk_leaves(adj, nid, leaf_mask, k=max(1, min(k_top, C)))
        # translate to cell indices (skip leaves not in provided list)
        row = np.zeros(C, dtype=np.float32)
        for leaf_id, dist in topk:
            leaf_name = node_id_to_name[leaf_id]
            if leaf_name is None:
                continue
            j = name_to_idx.get(leaf_name, None)
            if j is None:
                continue
            row[j] = math.exp(-float(dist) / float(tau))
        # ensure self
        row[cell_idx] = max(row[cell_idx], 1.0)
        s = float(row.sum())
        if s > 0:
            row /= s
        else:
            row[cell_idx] = 1.0
        K[cell_idx] = row

    return K

# =============== Lineage-weighted correlations & mixes ===============

@torch.no_grad()
def lin_lineage_weighted_corr_topk(X: torch.Tensor, K: torch.Tensor, k_top: int) -> torch.Tensor:
    """
    lineage-weighted correlation graph:
      corr_K(i,j) = (x_i^T K x_j) / sqrt( (x_i^T K x_i) (x_j^T K x_j) )
    keep positives, row-wise top-k => bin adjacency => GCN normalize.
    """
    XK = X @ K                 # (G x C)
    N = XK @ X.T               # (G x G)
    denom_i = (XK * X).sum(dim=1).clamp_min(1e-8).sqrt()  # (G,)
    D = denom_i[:, None] * denom_i[None, :]
    corr = (N / D).nan_to_num(0.0).clamp(min=-1.0, max=1.0)
    corr = torch.where(corr > 0, corr, torch.zeros_like(corr))
    A_bin = lin_topk_from_scores(corr, k=k_top)
    return lin_gcn_norm(A_bin)

def lin_mix_adjacency(A_raw_norm: torch.Tensor,
                      A_lin_norm: torch.Tensor,
                      beta: float = 0.5) -> torch.Tensor:
    """Blend two normalized adjacencies."""
    beta = float(beta)
    return beta * A_raw_norm + (1.0 - beta) * A_lin_norm

def lin_multiscale_concat_features(X: torch.Tensor,
                                   K: Optional[torch.Tensor],
                                   scales: List[int]) -> torch.Tensor:
    """
    Concatenate [X, X K^s1, X K^s2, ...] along cell-dimension => (G x C*(S+1)).
    """
    feats = [X]
    if (K is not None) and (scales is not None):
        for s in scales:
            if int(s) < 1: 
                continue
            K_pow = K
            for _ in range(int(s)-1):
                K_pow = K_pow @ K
            feats.append(X @ K_pow)
    return torch.cat(feats, dim=1)

@torch.no_grad()
def lin_pairwise_corr_for_edges(X: torch.Tensor,
                                edges_ij: torch.Tensor,
                                K: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Pairwise correlation for selected edges (E x 2).
      - if K is None: Pearson (per-gene zscore first)
      - else: lineage-weighted corr_K
    Returns: (E,) in [-1,1]
    """
    i = edges_ij[:,0]; j = edges_ij[:,1]
    xi = X[i]; xj = X[j]
    if K is None:
        xi = (xi - xi.mean(dim=1, keepdim=True)) / (xi.std(dim=1, keepdim=True).clamp_min(1e-8))
        xj = (xj - xj.mean(dim=1, keepdim=True)) / (xj.std(dim=1, keepdim=True).clamp_min(1e-8))
        num = (xi * xj).sum(dim=1)
        den = xi.norm(dim=1) * xj.norm(dim=1)
        return (num / den).nan_to_num(0.0).clamp(min=-1.0, max=1.0)
    else:
        xiK = xi @ K
        num = (xiK * xj).sum(dim=1)
        den = ((xiK * xi).sum(dim=1).clamp_min(1e-8).sqrt() *
               (((xj @ K) * xj).sum(dim=1).clamp_min(1e-8).sqrt()))
        return (num / den).nan_to_num(0.0).clamp(min=-1.0, max=1.0)

# =============== Smoothing / regularizers / calibration ===============

def lineage_smooth(X: torch.Tensor, K: torch.Tensor, scales: List[int], alpha: float = 0.3,
                   restandardize: bool = False) -> torch.Tensor:
    """
    Multi-scale lineage smoothing:
      X_sm = mean_s (X @ K^s)
      X' = (1-α) X + α X_sm
    """
    if (K is None) or (alpha <= 0) or (len(scales) == 0):
        return X
    X_sm_list = []
    for s in scales:
        if int(s) < 1: continue
        K_pow = K
        for _ in range(int(s)-1):
            K_pow = K_pow @ K
        X_sm_list.append(X @ K_pow)
    X_sm = torch.stack(X_sm_list, dim=0).mean(dim=0) if len(X_sm_list) > 1 else X_sm_list[0]
    X_new = (1.0 - float(alpha)) * X + float(alpha) * X_sm
    if restandardize:
        mu = X_new.mean(dim=1, keepdim=True)
        sd = X_new.std(dim=1, keepdim=True).clamp_min(1e-8)
        X_new = (X_new - mu) / sd
    return X_new

def lineage_consistency_loss(edge_idx: torch.Tensor,
                             edge_scores: torch.Tensor,
                             X: torch.Tensor,
                             K: torch.Tensor,
                             top_p: float = 0.3) -> torch.Tensor:
    """
    On top-p highest-score edges, penalize dissimilarity under lineage-weighted correlation:
      L = mean( (1 - corr_K(x_i, x_j))^2 )
    """
    E = edge_idx.size(0)
    if E == 0 or K is None:
        return torch.zeros((), device=X.device)
    p = max(1, int(E * float(top_p)))
    sel = torch.topk(edge_scores.detach(), k=p, largest=True).indices
    idx = edge_idx[sel]
    i_idx = idx[:,0]; j_idx = idx[:,1]
    xi = X[i_idx]; xj = X[j_idx]
    xiK = xi @ K
    num = (xiK * xj).sum(dim=1)
    den = ((xiK * xi).sum(dim=1).clamp_min(1e-8).sqrt() *
           (((xj @ K) * xj).sum(dim=1).clamp_min(1e-8).sqrt()))
    corr = (num / den).nan_to_num(0.0).clamp(min=-1.0, max=1.0)
    loss = ((1.0 - corr)**2).mean()
    return loss

def lin_direction_prior_loss(edge_idx: torch.Tensor,
                             edge_scores: torch.Tensor,
                             X: torch.Tensor,
                             t: torch.Tensor,
                             top_p: float = 0.3,
                             margin: float = 0.05) -> torch.Tensor:
    """
    Direction prior: for i->j with high score, encourage corr(x_i, t) >= corr(x_j, t) - margin.
    """
    E = edge_idx.size(0)
    if E == 0:
        return torch.zeros((), device=X.device)
    p = max(1, int(E * float(top_p)))
    sel = torch.topk(edge_scores.detach(), k=p, largest=True).indices
    ij = edge_idx[sel]
    i_idx, j_idx = ij[:,0], ij[:,1]

    # zscore on cells
    t0 = (t - t.mean()) / (t.std().clamp_min(1e-8))
    xi = (X[i_idx] - X[i_idx].mean(dim=1, keepdim=True)) / (X[i_idx].std(dim=1, keepdim=True).clamp_min(1e-8))
    xj = (X[j_idx] - X[j_idx].mean(dim=1, keepdim=True)) / (X[j_idx].std(dim=1, keepdim=True).clamp_min(1e-8))

    rho_i = (xi * t0).sum(dim=1) / (xi.norm(dim=1) * t0.norm())
    rho_j = (xj * t0).sum(dim=1) / (xj.norm(dim=1) * t0.norm())

    loss = F.relu(rho_j - rho_i + float(margin)).mean()
    return loss

# =============== Temperature scaling (calibration) ===============

class _TempScaler(nn.Module):
    """Learn a positive scalar temperature for probability calibration."""
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        # log T parameterization to ensure positivity
        self.logT = nn.Parameter(torch.tensor([math.log(init_T + 1e-8)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.logT).clamp(min=1e-4, max=100.0)
        return logits / T

def temperature_calibrate(logits_val: np.ndarray, y_val: np.ndarray, max_iter: int = 200) -> float:
    """
    Fit a global temperature T by minimizing BCE on validation logits.
    Returns the scalar T (>0).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = _TempScaler(1.0).to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.5, max_iter=max_iter, line_search_fn='strong_wolfe')
    x = torch.from_numpy(logits_val.astype(np.float32)).to(device).view(-1)
    y = torch.from_numpy(y_val.astype(np.float32)).to(device).view(-1)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        z = scaler(x)
        loss = bce(z, y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(scaler.logT).item())
    return T
