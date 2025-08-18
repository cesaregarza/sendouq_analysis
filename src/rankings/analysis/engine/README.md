# Unified Ranking System: Mathematical and Implementation Guide

This document explains the unified ranking system in `rankings.analysis.engine` which provides two ranking modes within a single framework:

- **Exposure Log-Odds Mode** (`exposure_logodds.py`) — recommended, volume-bias resistant
- **Tick-Tock Mode** (`core.py`) — PageRank with tournament-strength feedback

Both modes share the same core pipeline: graphs from match outcomes, time decay, tournament-strength weighting, and custom teleport vectors. They differ primarily in their final scoring calculation and bias correction approaches.

---

## Notation

- Entities (nodes): teams or players. Denote the set of nodes by \(V\), \(|V|=N\).
- Matches: edges directed from loser to winner. A single match between a losing entity \(i\) and a winning entity \(j\) contributes weight \(w_{ij} > 0\).
- Tournament influence: \(S_t\) for tournament \(t\) (normalized, mean \(=1\)).
- **Time decay:** For an event at timestamp \(\tau\) with reference time \(T_{\text{ref}}\),
  $$
  \Delta = \frac{T_{\text{ref}}-\tau}{86400}\ \text{(days)},\qquad
  \lambda^{\mathrm d} = \frac{\ln 2}{\text{half\_life\_days}},\qquad
  \text{match\_weight} = \exp(-\lambda^{\mathrm d}\,\Delta)\, S_t^{\beta}.
  $$
- Damping factor: \(\alpha \in (0,1)\) for PageRank.
- Teleport vector: probability vector \(\rho \in \mathbb{R}^N\) with \(\sum_i \rho_i = 1\).

---

## Core building blocks

### Time and tournament-strength weighting

For each match in tournament \(t\) at time \(\tau\):
$$
\text{match\_weight} = \exp(-\lambda^{\mathrm d}\,\Delta)\, S_t^{\beta}
$$
where \(\beta\) controls how much tournament strength influences the graph.

### Loser→Winner adjacency and normalization

Let \(W_{ij}\) be the sum of match weights from loser \(i\) to winner \(j\). Per-source totals:
$$
W^{\text{loss}}_i=\sum_j W_{ij},\qquad
W^{\text{win}}_i=\sum_k W_{ki}.
$$

Denominator with smoothing:
$$
\operatorname{denom}_i =
\begin{cases}
W^{\text{loss}}_i + \gamma\, W^{\text{win}}_i & \text{(wins‑proportional)}\\[2pt]
W^{\text{loss}}_i + c & \text{(constant)}\\[2pt]
W^{\text{loss}}_i & \text{(none)}
\end{cases}
$$

Let the *added smoothing* be \(\sigma_i \coloneqq \operatorname{denom}_i - W^{\text{loss}}_i\) and (optionally) cap it by \(\sigma_i \le \kappa\, W^{\text{loss}}_i\). Then
$$
P_{ij}=\frac{W_{ij}}{\operatorname{denom}_i},\qquad
\sum_j P_{ij}=\frac{W^{\text{loss}}_i}{\operatorname{denom}_i}\le 1.
$$

### Teleport vectors

- Uniform: \(\rho_i = 1/N\).
- Volume-inverse: \(\rho_i \propto 1/\sqrt{\max(\text{loss\_count}_i,1)}\).
- Volume-mix: convex combination of uniform and a volume component \((\epsilon + \text{count}_i)^{\gamma}\) with mixing weight \(\eta\).
- Custom dict: directly specify \(\rho\) over nodes.

After construction, add \(\varepsilon>0\) and renormalize \(\rho\) to avoid zeros.

---

## Tick-Tock Mode — `core.py`

### High-level algorithm

1. Initialize tournament influences \(S_t \leftarrow 1\) for all tournaments present in the data.
2. Repeat until convergence or max iterations:
   - Tick: build edges using current \(S_t\) and compute PageRank scores \(r\).
   - Tock: recompute \(S_t\) by aggregating current \(r\) over participants per tournament; normalize to mean 1.
3. Return final rankings and retrospective tournament strength metrics.

This produces mutually consistent ratings and tournament strengths.

### PageRank with row deficits (iteration and fixed point)

Let \(P\) be the row-normalized matrix above and define \(q_i=1-\sum_j P_{ij}\). With damping \(\alpha\) and teleport \(\rho\),
$$
\boxed{
\mathbf{r}^{(k+1)}=(1-\alpha)\rho+\alpha P^\top \mathbf{r}^{(k)}+\alpha\,\delta^{(k)}\,\rho,\quad
\delta^{(k)}=\sum_i q_i\, r^{(k)}_i
}
$$
This conserves mass since \(\sum \mathbf{r}^{(k+1)}=1\).

At the fixed point \(\mathbf{r}\), the exact linear system is
$$
\boxed{
\bigl(I-\alpha P^\top-\alpha\,\rho\,\mathbf{q}^\top\bigr)\,\mathbf{r}=(1-\alpha)\,\rho
}
$$
which makes clear that the solution is linear in \(\rho\).

> *Edge‑flux diagnostic:* under this row‑stochastic convention, the per‑edge flux from \(i\to j\) is \(\alpha\, r_i P_{ij}\).

Convergence is tested via \(\ell_1\) change \(\|\mathbf{r}^{(k+1)} - \mathbf{r}^{(k)}\|_1\) below a tolerance.

### Tournament influence update

For tournament \(t\), gather all participating nodes \(\mathcal{P}_t\) and their ratings \(r_i\) (use a small global prior for unseen IDs). Aggregate via one of:
- mean, sum, median
- top-k sum or mean (e.g., top 20)
- log-compressed or sqrt-compressed variants

Then normalize so \(\mathbb{E}[S_t]=1\). Optionally shift so \(\min_t S_t\) equals a configured floor.

### Retrospective tournament strength

Separately compute a strength score for each tournament by aggregating \(r_i\) with a chosen method (mean, median, trimmed mean, or topN sum). This table is returned alongside \(S_t\).

### Parameters of interest

- Decay: half-life days, derived \(\lambda^{\mathrm d}\)
- Damping \(\alpha\)
- Strength exponent \(\beta\)
- Teleport spec: uniform, volume-inverse, volume-mix, or custom
- Smoothing: mode (wins/constant/none), \(\gamma\), \(c\), and cap ratio
- Influence aggregation and retrospective strength aggregation (and \(k\) for trimmed/topN)

### Complexity

- Edge build: \(\mathcal{O}(E)\) with vectorized grouping
- PageRank: \(\mathcal{O}(K(E+N))\) for \(K\) iterations
- Tick–tock: multiply by the number of tick–tock iterations (typically small)

---

## Exposure Log-Odds Mode — `exposure_logodds.py`

### Goal

Remove volume bias by computing separate PageRanks over wins and losses using the same exposure-based teleport. The final score is a smoothed log-odds ratio.

### Match conversion

For players: expand each team match into all loser–winner player pairs. For teams: use team IDs directly. Each converted match has:
- winners list, losers list
- weight \(\text{match\_weight} = \exp(-\lambda^{\mathrm d}\,\Delta)\, S_t^{\beta}\)
- tournament and match IDs, and a timestamp

For a single match with winners \(\mathcal{W}\) and losers \(\mathcal{L}\), split weight equally across all pairs to preserve total mass:
$$
\text{pair\_share}=\frac{w}{|\mathcal{W}|\,|\mathcal{L}|}.
$$

### Exposure-based teleport

Exposure for node \(i\) is the total sum of all match weights involving \(i\):
$$
E_i=\sum_{m\ni i} w_m,\qquad
\rho_i=\frac{E_i+\varepsilon}{\sum_k(E_k+\varepsilon)}.
$$

### Win and loss graphs

- Build the win adjacency \(A^{\text{win}}\) with edges loser\(\to\)winner and set \(A^{\text{loss}}=(A^{\text{win}})^\top\).
- Form **column-stochastic** transitions for each graph separately (handling dangling columns by redistributing to \(\rho\)):
  $$
  P_{\text{win}}=\operatorname{colnorm}(A^{\text{win}}),\qquad
  P_{\text{loss}}=\operatorname{colnorm}(A^{\text{loss}}).
  $$
> **Note.** With this (standard) convention, in general \(P_{\text{loss}}\neq P_{\text{win}}^\top\). The adjacency is an exact transpose, but the per-graph column normalizations differ.

### PageRank updates (dangling redistributed to \(\rho\))

For \(\mathbf{s}\) (wins) and \(\boldsymbol{\ell}\) (losses),
$$
\mathbf{s}^{(k+1)}=(1-\alpha)\rho+\alpha\bigl(P_{\text{win}}+\rho\,\mathbf{d}_{\text{win}}^\top\bigr)\mathbf{s}^{(k)},\qquad
\boldsymbol{\ell}^{(k+1)}=(1-\alpha)\rho+\alpha\bigl(P_{\text{loss}}+\rho\,\mathbf{d}_{\text{loss}}^\top\bigr)\boldsymbol{\ell}^{(k)},
$$
where \(d_j=1\) if column \(j\) is dangling, else 0.

*Idealized linear form (no dangling):*
$$
\mathbf{s}=(1-\alpha)(I-\alpha P_{\text{win}})^{-1}\rho,\qquad
\boldsymbol{\ell}=(1-\alpha)(I-\alpha P_{\text{loss}})^{-1}\rho.
$$

> **Exact‑transpose variant (optional).** If you want \(P_{\text{loss}}=P_{\text{win}}^\top\) *exactly*, build the win transition \(P_{\text{win}}\) as **row-stochastic**, set \(P_{\text{loss}}:=P_{\text{win}}^\top\), and use the row-deficit update (the same form as in the core engine) for both \(\mathbf{s}\) and \(\boldsymbol{\ell}\).

### Smoothed log-odds score

Let \(\mathbf{s}\) be the win PageRank and \(\boldsymbol{\ell}\) be the loss PageRank (both sum to 1). Choose a smoothing \(\mu\) (auto-tuned if unset) and compute the score for each node \(i\):
$$
\boxed{
\text{score}_i = \log \frac{s_i + \mu\,\rho_i}{\ell_i + \mu\,\rho_i}
}
$$
Choose \(\mu\) so that \(\mu\,\operatorname{median}(\rho)\) is a small fraction of a typical PageRank mass. A concrete, scale-free choice is
$$
\mu \;\text{s.t.}\; \mu\,\operatorname{median}(\rho)=\tau_0\cdot \frac{1}{N}\quad\text{with }\tau_0\in[0.02,0.03],
$$
since \(\tfrac{1}{N}\) is a robust “typical mass” for a probability vector.

### Why volume cancels (intuition and derivation)

- The invariance intuition does **not** require \(P_{\text{loss}}=P_{\text{win}}^\top\); it only uses that both PageRanks are linear in \(\rho\) (given fixed \(P\)'s) and that the same exposure-proportional \(\rho\) is used in both. If node \(i\)'s volume is multiplied by \(k\) while its opponent-mix fractions stay the same, edges touching \(i\) scale and \(\rho_i\) increases accordingly; both \(\mathbf{s}\) and \(\boldsymbol{\ell}\) shift in tandem, so the shared \(\mu\,\rho_i\) stabilizes the ratio.
- **Toy model with exact cancellation:** Assume homogeneous mixing so that for node \(i\):
  - Total exposure (incidence mass) is \(e_i\)
  - Conversion quality is a fixed \(q_i \in (0,1)\), so win-related mass is \(e_i q_i\) and loss-related mass is \(e_i(1-q_i)\)
  - Teleport is exposure-proportional: \(\rho_i = e_i / E\) with \(E = \sum_k e_k\)

  Under this simplification, PageRanks are proportional to those masses:
  $$
  s_i \propto e_i q_i, \qquad \ell_i \propto e_i(1-q_i).
  $$
  With smoothing \(\mu\), the log-odds used by the engine is
  $$
  \log\frac{e_i q_i + \mu\, e_i/E}{e_i(1-q_i) + \mu\, e_i/E}
  = \log\frac{q_i + \mu/E}{1-q_i + \mu/E},
  $$
  which is independent of exposure \(e_i\). Thus in the idealized case the engine measures only conversion quality (with mild smoothing \(\mu/E\)).

### Optional surprisal weighting (temperature symbol)

During edge construction, an upset-aware weight multiplies \(w\):
$$
U = -\ln\!\big(\max(p_{\text{win}},10^{-10})\big),\quad
p_{\text{win}} = \frac{1}{1+\exp\!\big(-(r_{\mathcal{W}}-r_{\mathcal{L}})/\tau\big)}.
$$
where \(r_{\mathcal{W}}\) and \(r_{\mathcal{L}}\) are average provisional ratings for winners and losers, and \(\tau\) is the logistic temperature. Surprisal is iterated a few times to refine ratings.

### Inactivity decay (post)

After computing log-odds, apply an inactivity decay to scores after a grace period \(D\) days:
$$
\text{decay\_factor}(d)=
\begin{cases}
1,& d\le D\\
(1-\rho_d)^{\,d-D},& d>D
\end{cases}
$$
with daily decay rate \(\rho_d\). This preserves recent performance while gently decaying stale scores.

### Output columns

- Players: `id`, `player_rank` (log-odds), `win_pr`, `loss_pr`, `exposure`
- Teams: `id`, `team_rank` (log-odds), `win_pr`, `loss_pr`, `exposure`

A convenience `post_process_rankings` can filter by minimum tournaments and assign rank labels.

### Complexity

Two PageRanks of similar size plus preprocessing; overall similar to the core engine’s single PageRank per tick.

---

## Implementation notes

- **DataFrames:** Polars is used for efficient joins, group-bys, and vectorized transforms. The code avoids double-decay and only computes timestamps once per path.
- **Timestamps:** engines prefer `last_game_finished_at` if present, else fallback to `match_created_at`, else now.
- **Missing participants:** byes/forfeits are flagged via `is_bye` and ignored when building edges.
- **Numerical hygiene:** teleports are strictly normalized with small \(\varepsilon\); all PR vectors are renormalized per iteration; assertions check graph mirroring and probability sums.
- **Edge flux:** the core PageRank computes a per-edge flux \(\alpha\, r_i P_{ij}\) for diagnostics and visualizations.
- **Teleport safety:** Always add \(\varepsilon>0\) to \(\rho\) and renormalize to avoid zeros.

---

## Mode Selection Guide

- **Exposure Log-Odds Mode**: Recommended for public-facing rankings where volume bias is a concern. Set \(\beta \in [0.5, 1]\), keep default \(\alpha\approx 0.85\), and use auto \(\mu\).
- **Tick-Tock Mode**: Useful when you need explicit tournament strength modeling and want to see the iterative refinement process. Prefer wins-proportional smoothing with a small \(\gamma\) and a reasonable cap ratio to stabilize low-activity nodes.
- Use surprisal weighting (available in both modes) when you want to highlight upsets; keep iterations small (e.g., 2).
- Aggregation `top_20_sum` is a good default for tournament influence; normalize minimum influence if you observe extreme lows.

---

## Quick Start: Mode Selection

The unified ranking system provides a single entry point with mode selection:

```python
from rankings.analysis.engine import build_ranking_engine

# Recommended: Exposure Log-Odds mode (volume-bias resistant)
engine = build_ranking_engine(mode="exposure_logodds", beta=1.0)
rankings = engine.rank_players(matches_df, players_df)

# Alternative: Tick-Tock mode (explicit tournament strength modeling)
engine = build_ranking_engine(mode="tick_tock", beta=1.0, max_tick_tock=5)
rankings = engine.rank_players(matches_df, players_df)
```

## Detailed Examples

### Exposure Log-Odds Mode (Players)

```python
from rankings.analysis.engine.exposure_logodds import ExposureLogOddsEngine

engine = ExposureLogOddsEngine(
    beta=1.0,                    # Tournament strength exponent
    lambda_smooth=None,          # Auto-tune smoothing
    use_surprisal=False,         # No upset weighting
    min_exposure=None            # No minimum exposure filter
)

player_rankings = engine.rank_players(matches_df, players_df)
# Returns: id, player_rank (log-odds), win_pr, loss_pr, exposure
```

### Tick-Tock Mode (Teams)

```python
from rankings.analysis.engine.core import RatingEngine

engine = RatingEngine(
    beta=1.0,                           # Tournament strength exponent
    influence_agg_method="top_20_sum",  # Tournament influence calculation
    max_tick_tock=10,                   # Maximum iterations
    teleport_spec="volume_inverse"      # Teleport vector type
)

team_rankings = engine.rank_teams(matches_df)
# Returns: id, rating, tournament_influence_, tournament_strength_
```
