"""Unified PageRank solver supporting both row and column stochastic orientations."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class PageRankConfig:
    """Configuration for PageRank algorithm."""

    alpha: float = 0.85
    tol: float = 1e-8
    max_iter: int = 200
    orientation: str = "row"  # "row" or "col" stochastic
    redistribute_dangling: bool = True


def pagerank_sparse(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    n: int,
    teleport: np.ndarray,
    cfg: PageRankConfig,
) -> np.ndarray:
    """
    Generic PageRank on sparse triplets. Supports row/col orientation with the same code path.

    Args:
        rows: Source node indices
        cols: Target node indices
        weights: Edge weights
        n: Number of nodes
        teleport: Teleport probability vector (must sum to 1)
        cfg: PageRank configuration

    Returns:
        PageRank scores vector (sums to 1)
    """
    # Create sparse matrix if scipy is available
    A = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))

    # Compute normalization per orientation
    sums = np.asarray(A.sum(axis=1 if cfg.orientation == "row" else 0)).ravel()

    # Compute inverse for normalization
    inv = np.zeros_like(sums)
    nz = sums > 0
    inv[nz] = 1.0 / sums[nz]

    # Transition multiply implemented via sparse matvec with proper normalization
    def multiply(v: np.ndarray) -> np.ndarray:
        if A is not None:
            if cfg.orientation == "row":
                # P^T @ v, but P row-stochastic => weight rows by inv[row]
                # => equivalent to A^T @ (v * inv)
                return A.T.dot(v * inv)
            else:
                # col-stochastic => v on cols
                # => equivalent to A @ (v * inv)
                return A.dot(v * inv)

        # Fallback: COO-style accumulation
        out = np.zeros(n)
        if cfg.orientation == "row":
            np.add.at(out, cols, v[rows] * (weights * inv[rows]))
        else:
            np.add.at(out, rows, v[cols] * (weights * inv[cols]))
        return out

    # Initialize with teleport vector
    r = teleport / teleport.sum()
    alpha = cfg.alpha

    # Power iteration
    for _ in range(cfg.max_iter):
        Mr = multiply(r)

        # Handle dangling nodes
        dangling = alpha * r[~nz].sum() if cfg.redistribute_dangling else 0.0

        # PageRank update
        r_new = alpha * Mr + (1 - alpha) * teleport + dangling * teleport
        r_new /= r_new.sum()

        # Check convergence
        if np.linalg.norm(r_new - r, 1) < cfg.tol:
            return r_new
        r = r_new

    return r


def pagerank_dense(
    adjacency: np.ndarray,
    teleport: np.ndarray,
    cfg: PageRankConfig,
) -> np.ndarray:
    """
    PageRank on dense adjacency matrix (for backward compatibility).

    Args:
        adjacency: Dense adjacency matrix
        teleport: Teleport probability vector
        cfg: PageRank configuration

    Returns:
        PageRank scores vector
    """
    n = adjacency.shape[0]

    # Convert to stochastic matrix
    if cfg.orientation == "row":
        row_sums = adjacency.sum(axis=1)
        P = np.zeros_like(adjacency)
        nonzero = row_sums > 0
        P[nonzero] = adjacency[nonzero] / row_sums[nonzero, np.newaxis]
    else:
        col_sums = adjacency.sum(axis=0)
        P = np.zeros_like(adjacency)
        nonzero = col_sums > 0
        P[:, nonzero] = adjacency[:, nonzero] / col_sums[nonzero]

    # Initialize
    r = teleport / teleport.sum()
    alpha = cfg.alpha

    # Power iteration
    for _ in range(cfg.max_iter):
        if cfg.orientation == "row":
            r_new = alpha * P.T @ r + (1 - alpha) * teleport
        else:
            r_new = alpha * P @ r + (1 - alpha) * teleport

        # Handle dangling nodes
        if cfg.redistribute_dangling:
            if cfg.orientation == "row":
                dangling_mass = alpha * r[row_sums == 0].sum()
            else:
                dangling_mass = alpha * r[col_sums == 0].sum()
            r_new += dangling_mass * teleport

        r_new /= r_new.sum()

        if np.linalg.norm(r_new - r, 1) < cfg.tol:
            return r_new
        r = r_new

    return r
