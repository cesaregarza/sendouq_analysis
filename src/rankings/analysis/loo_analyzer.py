"""
Leave-One-Match-Out (LOO) Analyzer for ExposureLogOddsEngine.

This module implements exact, efficient leave-one-match-out impact analysis
using low-rank PageRank updates via Sherman-Morrison-Woodbury formula.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)

EPS = 1e-15
DANGLING_EPS = 1e-12


# -------------------------
# Linear Solver Backend
# -------------------------


class LinearSolveBackend:
    """
    Pre-factorized solver for (I - alpha A) X = B with reusable solves.
    method:
      - 'splu'   : sparse LU (fastest, most robust)
      - 'gmres'  : GMRES with ILU preconditioner (less memory if LU is large)
    """

    def __init__(
        self, A_csc: sp.csc_matrix, alpha: float, method: str = "splu"
    ):
        self.n = A_csc.shape[0]
        M = sp.eye(self.n, format="csc") - alpha * A_csc

        if method == "splu":
            # One-time factorization; reuse for all RHS
            self._lu = spla.splu(
                M
            )  # you can tune permc_spec / diag_pivot_thresh if needed
            self._solve = lambda B: self._lu.solve(B)
        elif method == "gmres":
            # ILU preconditioner + GMRES (use if splu memory is an issue)
            ilu = spla.spilu(M, drop_tol=1e-4, fill_factor=10)
            P = spla.LinearOperator(M.shape, matvec=lambda x: ilu.solve(x))

            def _solve(B):
                if B.ndim == 1:
                    B = B[:, None]
                X = np.empty_like(B, dtype=float)
                for j in range(B.shape[1]):
                    x, info = spla.gmres(
                        M, B[:, j], M=P, tol=1e-10, restart=50, maxiter=200
                    )
                    if info != 0:
                        logger.warning(
                            f"GMRES did not fully converge (col {j}, info={info})"
                        )
                    X[:, j] = x
                return X

            self._solve = _solve
        else:
            raise ValueError(f"Unknown method {method}")

    def solve(self, B: np.ndarray) -> np.ndarray:
        """Return X solving (I - alpha A) X = B. Accepts (n,) or (n,k)."""
        return self._solve(B)


# -------------------------
# Matrix Construction Helpers
# -------------------------


def _normalize_and_fill_dangling(A_csr, rho):
    """Efficiently normalize columns and fill dangling with rho."""
    A = A_csr.tocsc()
    col_sums = np.asarray(A.sum(axis=0)).ravel()
    counts = np.diff(A.indptr)
    inv_sums = np.zeros_like(col_sums, dtype=float)
    nz = col_sums > 0
    inv_sums[nz] = 1.0 / col_sums[nz]
    A.data *= inv_sums.repeat(counts)

    # Add columns equal to rho where column-sum was zero:
    if (~nz).any():
        # Construct sparse matrix with those columns set to rho
        cols = np.where(~nz)[0]
        # Repeat rho for each dangling column
        nnz = len(cols) * len(rho)
        data = np.tile(rho, len(cols))
        row_idx = np.tile(np.arange(len(rho)), len(cols))
        col_idx = np.repeat(cols, len(rho))
        D = sp.csc_matrix((data, (row_idx, col_idx)), shape=A.shape)
        A = A + D
    return A, col_sums


def get_sparse_matrices(
    engine,
    matches_df: pl.DataFrame,
    node_to_idx: Dict[int, int],
    rho: np.ndarray,
) -> Tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray, np.ndarray]:
    """
    Build sparse adjacency matrices from engine state with dangling redistribution.

    Returns:
        A_win: Win graph adjacency (column-stochastic with dangling → rho)
        A_loss: Loss graph adjacency (column-stochastic with dangling → rho)
        T_win: Raw column sums for win graph
        T_loss: Raw column sums for loss graph
    """
    from rankings.core import build_exposure_triplets

    # Get triplets from the matches dataframe
    rows, cols, data = build_exposure_triplets(matches_df, node_to_idx)
    n = len(node_to_idx)

    # Win graph: A_win[i,j] = W_ij / sum_k(W_kj)
    A_win_raw = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A_win, col_sums_win = _normalize_and_fill_dangling(A_win_raw, rho)

    # Loss graph: transpose structure
    A_loss_raw = sp.csr_matrix((data, (cols, rows)), shape=(n, n))
    A_loss, col_sums_loss = _normalize_and_fill_dangling(A_loss_raw, rho)

    return A_win, A_loss, col_sums_win, col_sums_loss


def exposures_for_match(
    match_id: int,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    node_to_idx: Dict[int, int],
    graph: str = "win",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns triplets (rows_i, cols_j, raw_weights) for a specific match.

    Args:
        match_id: Match identifier
        matches_df: Processed matches with winners/losers lists
        players_df: Player data (not used in current implementation)
        node_to_idx: Mapping from player IDs to node indices
        graph: "win" or "loss" to specify which graph

    Returns:
        rows_i: Destination node indices
        cols_j: Source node indices
        raw_weights: Raw edge weights for this match
    """
    # Get match data
    match = matches_df.filter(pl.col("match_id") == match_id)
    if match.is_empty():
        return np.array([]), np.array([]), np.array([])

    match_data = match.to_dicts()[0]

    # Extract match weight (includes time decay and tournament influence)
    # Check for different possible column names
    if "weight" in match_data:
        match_weight = float(match_data["weight"])
    elif "w_m" in match_data:
        match_weight = float(match_data["w_m"])
    else:
        # Fallback: try to find any weight-like column
        logger.warning(f"No weight column found for match {match_id}")
        return np.array([]), np.array([]), np.array([])

    # Get winner and loser player IDs
    winners = match_data.get("winners", [])
    losers = match_data.get("losers", [])

    # Ensure they are lists
    if not isinstance(winners, list):
        winners = [winners] if winners is not None else []
    if not isinstance(losers, list):
        losers = [losers] if losers is not None else []

    # Map to node indices
    winner_indices = [node_to_idx[w] for w in winners if w in node_to_idx]
    loser_indices = [node_to_idx[l] for l in losers if l in node_to_idx]

    if graph == "win":
        # Win graph: edges from losers to winners
        rows_i = []
        cols_j = []
        raw_weights = []

        for winner_idx in winner_indices:
            for loser_idx in loser_indices:
                rows_i.append(winner_idx)
                cols_j.append(loser_idx)
                # Raw weight per edge (distributed across all pairs)
                raw_weights.append(match_weight / (len(winners) * len(losers)))

        return (
            np.array(rows_i, dtype=np.int32),
            np.array(cols_j, dtype=np.int32),
            np.array(raw_weights, dtype=np.float64),
        )

    else:  # graph == "loss"
        # Loss graph: edges from winners to losers
        rows_i = []
        cols_j = []
        raw_weights = []

        for loser_idx in loser_indices:
            for winner_idx in winner_indices:
                rows_i.append(loser_idx)
                cols_j.append(winner_idx)
                raw_weights.append(match_weight / (len(winners) * len(losers)))

        return (
            np.array(rows_i, dtype=np.int32),
            np.array(cols_j, dtype=np.int32),
            np.array(raw_weights, dtype=np.float64),
        )


def _compute_total_exposure_mass(matches_df: pl.DataFrame) -> float:
    """
    Compute total exposure mass E = sum over matches of (share * num_participants).
    """
    total = 0.0
    for row in matches_df.iter_rows(named=True):
        share = float(row.get("share", 0.0))
        winners = row.get("winners", []) or []
        losers = row.get("losers", []) or []
        if not isinstance(winners, list):
            winners = [winners] if winners is not None else []
        if not isinstance(losers, list):
            losers = [losers] if losers is not None else []
        total += share * (len(winners) + len(losers))
    return max(total, 1e-12)


def delta_rho_for_match(
    match_id: int,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    node_to_idx: Dict[int, int],
    total_exposure: float,
) -> Optional[np.ndarray]:
    """
    Returns the change in teleport vector if this match is removed.

    Computes change in normalized space: delta = -sigma for each participant,
    where sigma = share / total_exposure.

    Args:
        match_id: Match identifier
        matches_df: Processed matches with share values
        players_df: Player data (not used in current implementation)
        node_to_idx: Mapping from player IDs to node indices
        total_exposure: Total exposure mass E

    Returns:
        Delta rho vector (negative values since we're removing)
    """
    match = matches_df.filter(pl.col("match_id") == match_id)
    if match.is_empty():
        return np.zeros(len(node_to_idx))

    match_data = match.to_dicts()[0]

    # Get share value for this match
    share = float(match_data.get("share", 0.0))
    if share == 0.0:
        return None  # No teleport change

    # Get all players involved
    winners = match_data.get("winners", [])
    losers = match_data.get("losers", [])

    if not isinstance(winners, list):
        winners = [winners] if winners is not None else []
    if not isinstance(losers, list):
        losers = [losers] if losers is not None else []

    # Compute change in normalized space
    sigma = share / total_exposure
    delta_rho = np.zeros(len(node_to_idx))

    # Each participant loses sigma in the normalized space
    for player_id in winners + losers:
        if player_id in node_to_idx:
            idx = node_to_idx[player_id]
            delta_rho[idx] -= sigma

    return delta_rho


# -------------------------
# Block Resolvent Solver
# -------------------------


def block_resolvent_fixed_point(
    A_csc: sp.csc_matrix,
    alpha: float,
    U: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 400,
) -> np.ndarray:
    """
    Solve (I - alpha A) X = U for multiple RHS using fixed-point iteration.

    Uses contractive fixed-point: X_{t+1} = alpha * A @ X_t + U
    Converges linearly since ||alpha*A||_1 <= alpha < 1

    Args:
        A_csc: (n,n) CSC sparse, column-stochastic
        alpha: Damping factor
        U: (n,m) dense RHS matrix
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        X = (I - alpha A)^{-1} U
    """
    n, m = U.shape
    X = np.zeros((n, m), dtype=U.dtype)

    for it in range(max_iter):
        X_new = alpha * (A_csc @ X) + U

        # Relative 1-norm stopping criterion
        num = np.linalg.norm(X_new - X, ord=1)
        den = 1.0 + np.linalg.norm(X_new, ord=1)

        if num < tol * den:
            return X_new

        X = X_new

    logger.warning(
        f"Fixed-point iteration did not converge in {max_iter} iterations"
    )
    return X


# -------------------------
# Column Update Construction
# -------------------------


def build_U_alpha_for_graph(
    A_csc: sp.csc_matrix,
    T_col_raw: np.ndarray,
    rho: np.ndarray,
    rho_new: np.ndarray,  # New teleport after removal
    alpha: float,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    n: int,
) -> Tuple[List[int], np.ndarray]:
    """
    Build per-source-column updates u_j for a match removal.

    Args:
        A_csc: Current adjacency matrix (column-stochastic)
        T_col_raw: Raw column sums before normalization
        rho: Teleport vector
        alpha: Damping factor
        rows, cols, weights: Match triplets
        n: Number of nodes

    Returns:
        j_list: List of affected source columns
        U: (n, k) matrix with columns alpha*u_j
    """
    # Group match contributions by source column j
    by_col = {}
    for i, j, w in zip(rows, cols, weights):
        if T_col_raw[j] <= 0:
            # Raw column sum was zero; skip
            continue
        by_col.setdefault(int(j), []).append((int(i), float(w)))

    j_list = sorted(by_col.keys())
    k = len(j_list)

    if k == 0:
        return j_list, np.zeros((n, 0), dtype=float)

    U = np.zeros((n, k), dtype=float)

    for c, j in enumerate(j_list):
        entries = by_col[j]
        Tj = float(T_col_raw[j])

        # Current normalized column a_j (dense)
        aj = np.zeros(n, dtype=float)
        start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
        aj[A_csc.indices[start:end]] = A_csc.data[start:end]

        sum_w = sum(w for (_i, w) in entries)
        delta = sum_w / Tj  # Fraction of column mass removed

        if delta >= 1.0 - DANGLING_EPS:
            # Column becomes dangling -> replace with rho_new
            u = rho_new - aj
        else:
            # Normal update
            rj = np.zeros(n, dtype=float)
            for i, w in entries:
                rj[i] += w / Tj
            scale = 1.0 / (1.0 - delta)
            u = scale * (delta * aj - rj)

        U[:, c] = alpha * u

    return j_list, U


# -------------------------
# Exact Rank-k Update
# -------------------------


def loo_update_graph_exact(
    A_csc: sp.csc_matrix,
    s: np.ndarray,
    rho: np.ndarray,
    alpha: float,
    T_col_raw: np.ndarray,
    match_rows: np.ndarray,
    match_cols: np.ndarray,
    match_weights: np.ndarray,
    delta_rho_vec: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 400,
    R_solve=None,  # Pre-factorized solver for (I - alpha A)^{-1}
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Exact LOO update for a single match on a single PageRank vector.

    Returns:
        s_new: Updated PageRank vector
        aux: Dictionary with auxiliary information
    """
    n = A_csc.shape[0]

    # Compute new teleport first if needed
    if delta_rho_vec is not None and np.any(delta_rho_vec):
        rho_tmp = rho + delta_rho_vec
        rho_tmp = np.maximum(rho_tmp, 0.0)  # Enforce non-negativity
        total = float(rho_tmp.sum())
        rho_new = rho_tmp / total if total > 0.0 else rho.copy()
    else:
        rho_new = rho

    # Build per-column updates
    j_list, U = build_U_alpha_for_graph(
        A_csc,
        T_col_raw,
        rho,
        rho_new,
        alpha,
        match_rows,
        match_cols,
        match_weights,
        n,
    )
    k = U.shape[1]

    if k == 0 and (delta_rho_vec is None or not np.any(delta_rho_vec)):
        return s.copy(), {"k": 0, "teleport_applied": False}

    # X = R U where R = (I - alpha A)^{-1}
    if k > 0:
        if R_solve is not None:
            X = R_solve(U)
        else:
            X = block_resolvent_fixed_point(
                A_csc, alpha, U, tol=tol, max_iter=max_iter
            )

        # K = I - E^T X
        X_rows = X[j_list, :]  # (k, k)
        K = np.eye(k, dtype=float) - X_rows

        # Check conditioning of K
        try:
            cond = np.linalg.cond(K)
            if not np.isfinite(cond) or cond > 1e8:
                logger.warning(
                    f"K matrix is ill-conditioned: cond={cond:.3e}. Results may be inaccurate."
                )
        except:
            pass  # Conditioning check failed, continue anyway

        # beta solves K beta = g, where g = E^T s = s[j_list]
        beta = np.linalg.solve(K, s[j_list])
        s_star = s + X @ beta
    else:
        X = np.zeros((n, 0), dtype=float)
        K = np.zeros((0, 0), dtype=float)
        s_star = s.copy()

    # Optional teleport update
    if delta_rho_vec is not None and np.any(delta_rho_vec):
        # rho_new was already computed above

        # RHS for teleport change
        v = (1.0 - alpha) * (rho_new - rho)

        # Skip negligible teleport updates
        teleport_norm = (1.0 - alpha) * np.linalg.norm(delta_rho_vec, 1)
        if teleport_norm < 1e-12:
            return s_star, {
                "k": k,
                "teleport_applied": False,
                "rho_new": rho_new,
                "j_list": j_list,
                "K": K if k > 0 else np.zeros((0, 0)),
            }

        if R_solve is not None:
            y = R_solve(v.reshape(-1, 1))[:, 0]
        else:
            y = block_resolvent_fixed_point(
                A_csc, alpha, v.reshape(-1, 1), tol=tol, max_iter=max_iter
            )[:, 0]

        if k > 0:
            gamma = np.linalg.solve(K, y[j_list])
            s_new = s_star + y + X @ gamma
        else:
            s_new = s_star + y

        return s_new, {
            "k": k,
            "teleport_applied": True,
            "rho_new": rho_new,
            "j_list": j_list,
            "K": K,
        }
    else:
        return s_star, {
            "k": k,
            "teleport_applied": False,
            "rho_new": rho,
            "j_list": j_list,
            "K": K,
        }


# -------------------------
# Main LOO Analyzer Class
# -------------------------


class LOOAnalyzer:
    """
    Leave-One-Match-Out analyzer for ExposureLogOddsEngine.

    Computes exact impact of removing a single match on player scores
    using efficient rank-k updates instead of full PageRank recomputation.
    """

    def _check_pagerank_validity(
        self, name: str, s: np.ndarray, rho: np.ndarray
    ) -> None:
        """
        Check PageRank vector validity and log warnings for invariant violations.

        Args:
            name: Name of the PageRank vector (for logging)
            s: PageRank vector to check
            rho: Teleport vector
        """
        # Check for negative entries
        if np.any(s < -1e-12):
            min_val = float(s.min())
            logger.warning(f"{name} has negative entries: min={min_val:.3e}")

        # Check normalization
        ssum = float(s.sum())
        if not (0.9999 <= ssum <= 1.0001):
            logger.warning(f"{name} not normalized: sum={ssum:.6f}")

        # Check against theoretical lower bound
        lower_bound = (1.0 - self.alpha) * rho
        violations = s < (lower_bound - 1e-6)
        if np.any(violations):
            num_violations = int(violations.sum())
            max_violation = float((lower_bound - s)[violations].max())
            logger.debug(
                f"{name} violates (1-α)ρ bound at {num_violations} entries, max violation={max_violation:.3e}"
            )

    def __init__(
        self, engine, matches_df: pl.DataFrame, players_df: pl.DataFrame
    ):
        """
        Initialize LOO analyzer from engine state.

        Args:
            engine: ExposureLogOddsEngine instance after rank_players()
            matches_df: Processed matches DataFrame with winners/losers lists
            players_df: Players DataFrame
        """
        # Core parameters
        self.alpha = float(engine.config.pagerank.alpha)
        self.rho = engine._last_rho.astype(float)
        self.lam = float(engine.last_result.lambda_used)

        # Node mapping
        self.node_ids = engine.last_result.ids
        self.node_to_idx = {pid: i for i, pid in enumerate(self.node_ids)}
        self.n = len(self.node_ids)

        # Current PageRank vectors
        self.s_win = engine.last_result.win_pr.astype(float)
        self.s_loss = engine.last_result.loss_pr.astype(float)

        # Store actual scores from engine
        self.actual_scores = engine.last_result.scores.astype(float)

        # Build sparse matrices
        logger.info("Building sparse adjacency matrices...")
        A_win, A_loss, T_win, T_loss = get_sparse_matrices(
            engine, matches_df, self.node_to_idx, self.rho
        )
        self.A_win = A_win.tocsc()
        self.A_loss = A_loss.tocsc()
        self.T_win = T_win.astype(float)
        self.T_loss = T_loss.astype(float)

        self.matches_df = matches_df
        self.players_df = players_df

        # Compute total exposure mass for teleport updates
        self._total_exposure = _compute_total_exposure_mass(matches_df)

        # Pre-factorize linear solvers for massive speedup
        logger.info("Pre-factorizing linear solvers...")
        self._win_solver = LinearSolveBackend(
            self.A_win, self.alpha, method="splu"
        )
        self._loss_solver = LinearSolveBackend(
            self.A_loss, self.alpha, method="splu"
        )

        # Build match cache for fast triplet/teleport lookup
        logger.info("Building match cache...")
        self._match_cache = self._build_match_cache()

        logger.info(
            f"LOOAnalyzer initialized: {self.n} nodes, "
            f"{self.A_win.nnz} win edges, {self.A_loss.nnz} loss edges"
        )

    def _build_match_cache(self):
        """Pre-compute all match triplets and teleport deltas."""
        cache = {}
        for row in self.matches_df.iter_rows(named=True):
            mid = row["match_id"]
            winners = row.get("winners", []) or []
            losers = row.get("losers", []) or []
            weight = float(row.get("weight", row.get("w_m", 0.0)))
            share = float(row.get("share", 0.0))

            w_idx = [
                self.node_to_idx[p] for p in winners if p in self.node_to_idx
            ]
            l_idx = [
                self.node_to_idx[p] for p in losers if p in self.node_to_idx
            ]
            if not w_idx or not l_idx:
                cache[mid] = None
                continue

            # Triplets (win graph: loser->winner)
            rows_w, cols_w = [], []
            for wi in w_idx:
                for li in l_idx:
                    rows_w.append(wi)
                    cols_w.append(li)
            wts_w = np.full(
                len(rows_w), weight / (len(w_idx) * len(l_idx)), dtype=float
            )

            # Triplets (loss graph: winner->loser)
            rows_l, cols_l = cols_w, rows_w  # same pairs reversed
            wts_l = wts_w.copy()

            # Teleport delta in normalized space
            sigma = (
                share / self._total_exposure
                if self._total_exposure > 0
                else 0.0
            )
            delta_rho = np.zeros(self.n, dtype=float)
            for p in winners + losers:
                j = self.node_to_idx.get(p)
                if j is not None:
                    delta_rho[j] -= sigma

            cache[mid] = (
                np.array(rows_w, np.int32),
                np.array(cols_w, np.int32),
                wts_w,
                np.array(rows_l, np.int32),
                np.array(cols_l, np.int32),
                wts_l,
                delta_rho,
            )
        return cache

    def exposures_for_match(
        self, match_id: int, graph: str = "win"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get match triplets for specified graph."""
        return exposures_for_match(
            match_id, self.matches_df, self.players_df, self.node_to_idx, graph
        )

    def delta_rho_for_match(self, match_id: int) -> Optional[np.ndarray]:
        """Get teleport vector change for match removal."""
        return delta_rho_for_match(
            match_id,
            self.matches_df,
            self.players_df,
            self.node_to_idx,
            self._total_exposure,
        )

    def node_index_for_player(self, player_id: int) -> Optional[int]:
        """Get node index for player ID."""
        return self.node_to_idx.get(player_id, None)

    def _estimate_match_flux(self, match_id: int, player_id: int) -> float:
        """
        Estimate flux for a match without full LOO computation.

        This uses the current PageRank values to estimate how much flux
        flows through the edges created by this match.

        Args:
            match_id: Match to estimate
            player_id: Player of interest

        Returns:
            Estimated flux magnitude
        """
        # Get cached match data
        match_data = self._match_cache.get(match_id)
        if match_data is None:
            return 0.0

        rows_w, cols_w, wts_w, rows_l, cols_l, wts_l, _ = match_data

        player_idx = self.node_index_for_player(player_id)
        if player_idx is None:
            return 0.0

        total_flux = 0.0

        # Estimate win flux (incoming to player if they won)
        for i, j, w in zip(rows_w, cols_w, wts_w):
            if i == player_idx:  # Player is winner (receives flux)
                if self.T_win[j] > 0:
                    flux = self.alpha * self.s_win[j] * (w / self.T_win[j])
                    total_flux += flux
            elif j == player_idx:  # Player is loser (source of flux)
                if self.T_win[player_idx] > 0:
                    flux = (
                        self.alpha
                        * self.s_win[player_idx]
                        * (w / self.T_win[player_idx])
                    )
                    total_flux += flux

        # Estimate loss flux (incoming to player if they lost)
        for i, j, w in zip(rows_l, cols_l, wts_l):
            if i == player_idx:  # Player is loser (receives flux in loss graph)
                if self.T_loss[j] > 0:
                    flux = self.alpha * self.s_loss[j] * (w / self.T_loss[j])
                    total_flux += flux
            elif j == player_idx:  # Player is winner (source in loss graph)
                if self.T_loss[player_idx] > 0:
                    flux = (
                        self.alpha
                        * self.s_loss[player_idx]
                        * (w / self.T_loss[player_idx])
                    )
                    total_flux += flux

        return total_flux

    def impact_of_match_on_player(
        self,
        match_id: int,
        player_id: int,
        include_teleport: bool = True,
        tol: float = 1e-10,
        max_iter: int = 400,
    ) -> Dict[str, Any]:
        """
        Compute exact change in player's log-odds score if match is removed.

        Args:
            match_id: Match to remove
            player_id: Player to analyze
            include_teleport: Whether to account for teleport vector changes
            tol: Convergence tolerance for solvers
            max_iter: Maximum iterations for solvers

        Returns:
            Dictionary with old/new scores and detailed components
        """
        k = self.node_index_for_player(player_id)
        if k is None:
            return {
                "ok": False,
                "reason": f"player_id {player_id} not found in node mapping",
            }

        # Get cached match data
        match_data = self._match_cache.get(match_id)
        if match_data is None:
            return {
                "ok": False,
                "reason": f"match_id {match_id} not found or has no valid edges",
            }

        (
            rows_w,
            cols_w,
            wts_w,
            rows_l,
            cols_l,
            wts_l,
            delta_rho_cached,
        ) = match_data

        # Use cached teleport delta if requested
        delta_rho = delta_rho_cached if include_teleport else None

        # Early exit if flux is negligible
        flux_estimate = self._estimate_match_flux(match_id, player_id)
        if abs(flux_estimate) < 1e-12:
            return {
                "ok": True,
                "old_score": self.actual_scores[k],
                "new_score": self.actual_scores[k],
                "delta": 0.0,
                "flux_estimate": flux_estimate,
                "early_exit": "negligible_flux",
            }

        # Update win graph
        logger.debug(f"Updating win graph for match {match_id}...")
        s_win_new, aux_win = loo_update_graph_exact(
            self.A_win,
            self.s_win,
            self.rho,
            self.alpha,
            self.T_win,
            rows_w,
            cols_w,
            wts_w,
            delta_rho_vec=delta_rho,
            tol=tol,
            max_iter=max_iter,
            R_solve=self._win_solver.solve,
        )

        # Check win PageRank validity
        self._check_pagerank_validity(
            "s_win_new", s_win_new, aux_win.get("rho_new", self.rho)
        )

        # Update loss graph
        logger.debug(f"Updating loss graph for match {match_id}...")
        s_loss_new, aux_loss = loo_update_graph_exact(
            self.A_loss,
            self.s_loss,
            self.rho,
            self.alpha,
            self.T_loss,
            rows_l,
            cols_l,
            wts_l,
            delta_rho_vec=delta_rho,
            tol=tol,
            max_iter=max_iter,
            R_solve=self._loss_solver.solve,
        )

        # Check loss PageRank validity
        self._check_pagerank_validity(
            "s_loss_new", s_loss_new, aux_loss.get("rho_new", self.rho)
        )

        # Use updated rho if teleport was applied
        rho_new = (
            aux_win.get("rho_new", self.rho) if include_teleport else self.rho
        )

        # Compute scores
        def score_from(s_w, s_l, rho_vec, idx):
            # Project PageRank to non-negative (guards against tiny negatives from linear solves)
            sw = max(float(s_w[idx]), 0.0)
            sl = max(float(s_l[idx]), 0.0)

            # Theory-consistent floor: PageRank ensures s_i >= (1-alpha)*rho_i
            # Use half the baseline as a conservative floor
            min_floor = 0.5 * (1.0 - self.alpha) * rho_vec[idx]

            # Apply floor and add smoothing
            w = max(sw, min_floor) + self.lam * rho_vec[idx]
            l = max(sl, min_floor) + self.lam * rho_vec[idx]
            return float(np.log(w / l))

        # Use actual score from engine for old_score
        old_score = float(self.actual_scores[k])
        # Compute what the new score would be
        new_score_computed = score_from(s_win_new, s_loss_new, rho_new, k)
        # The delta is the difference
        score_delta = new_score_computed - score_from(
            self.s_win, self.s_loss, self.rho, k
        )
        new_score = old_score + score_delta

        return {
            "ok": True,
            "player_id": player_id,
            "match_id": match_id,
            "old": {
                "score": old_score,
                "s_win": float(self.s_win[k]),
                "s_loss": float(self.s_loss[k]),
                "rho": float(self.rho[k]),
            },
            "new": {
                "score": new_score,
                "s_win": float(s_win_new[k]),
                "s_loss": float(s_loss_new[k]),
                "rho": float(rho_new[k]),
            },
            "delta": {
                "score": score_delta,
                "s_win": float(s_win_new[k] - self.s_win[k]),
                "s_loss": float(s_loss_new[k] - self.s_loss[k]),
                "rho": float(rho_new[k] - self.rho[k]),
            },
            "internals": {
                "alpha": self.alpha,
                "lambda_smooth": self.lam,
                "k_win_columns": aux_win["k"],
                "k_loss_columns": aux_loss["k"],
                "teleport_applied": aux_win["teleport_applied"]
                or aux_loss["teleport_applied"],
            },
        }

    def analyze_player_matches(
        self,
        player_id: int,
        limit: Optional[int] = None,
        include_teleport: bool = True,
        use_flux_ranking: bool = True,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> pl.DataFrame:
        """
        Analyze impact of matches involving a player.

        Args:
            player_id: Player to analyze
            limit: Maximum number of matches to analyze
            include_teleport: Whether to account for teleport changes
            use_flux_ranking: Whether to prioritize matches by pre-computed flux
            parallel: Whether to analyze matches in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            DataFrame with match impacts sorted by absolute score change
        """
        # Find all matches involving this player
        player_matches = []

        for match in self.matches_df.iter_rows(named=True):
            winners = match.get("winners", [])
            losers = match.get("losers", [])

            if not isinstance(winners, list):
                winners = [winners] if winners is not None else []
            if not isinstance(losers, list):
                losers = [losers] if losers is not None else []

            if player_id in winners or player_id in losers:
                match_info = {
                    "match_id": match["match_id"],
                    "is_win": player_id in winners,
                    "flux_estimate": 0.0,
                }

                # If using flux ranking, compute preliminary flux estimate
                if use_flux_ranking:
                    flux_est = self._estimate_match_flux(
                        match["match_id"], player_id
                    )
                    match_info["flux_estimate"] = flux_est

                player_matches.append(match_info)

        # Sort by flux estimate if using flux ranking
        if use_flux_ranking and player_matches:
            player_matches.sort(
                key=lambda x: abs(x["flux_estimate"]), reverse=True
            )

        if limit:
            player_matches = player_matches[:limit]

        # Analyze each match
        results = []

        if parallel and len(player_matches) > 1:
            # Parallel execution
            def _impact_single(match_info):
                impact = self.impact_of_match_on_player(
                    match_info["match_id"],
                    player_id,
                    include_teleport=include_teleport,
                )
                return match_info, impact

            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(player_matches))
            ) as ex:
                futures = [
                    ex.submit(_impact_single, mi) for mi in player_matches
                ]
                for i, fut in enumerate(as_completed(futures)):
                    match_info, impact = fut.result()
                    logger.info(
                        f"Completed match {i+1}/{len(player_matches)}: {match_info['match_id']}"
                    )

                    if impact["ok"]:
                        results.append(
                            {
                                "match_id": match_info["match_id"],
                                "is_win": match_info["is_win"],
                                "old_score": impact["old"]["score"],
                                "new_score": impact["new"]["score"],
                                "score_delta": impact["delta"]["score"],
                                "abs_delta": abs(impact["delta"]["score"]),
                                "win_pr_delta": impact["delta"]["s_win"],
                                "loss_pr_delta": impact["delta"]["s_loss"],
                            }
                        )
        else:
            # Sequential execution
            for i, match_info in enumerate(player_matches):
                logger.info(
                    f"Analyzing match {i+1}/{len(player_matches)}: {match_info['match_id']}"
                )

                impact = self.impact_of_match_on_player(
                    match_info["match_id"],
                    player_id,
                    include_teleport=include_teleport,
                )

                if impact["ok"]:
                    results.append(
                        {
                            "match_id": match_info["match_id"],
                            "is_win": match_info["is_win"],
                            "old_score": impact["old"]["score"],
                            "new_score": impact["new"]["score"],
                            "score_delta": impact["delta"]["score"],
                            "abs_delta": abs(impact["delta"]["score"]),
                            "win_pr_delta": impact["delta"]["s_win"],
                            "loss_pr_delta": impact["delta"]["s_loss"],
                        }
                    )

        if not results:
            return pl.DataFrame()

        return pl.DataFrame(results).sort("abs_delta", descending=True)
