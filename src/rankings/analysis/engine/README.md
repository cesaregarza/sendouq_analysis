# Rankings Engines: Mathematical and Implementation Guide

This document explains the two engines in `rankings.analysis.engine` in both conceptual and implementation detail:

- Exposure Log-Odds Engine (`exposure_logodds.py`) — recommended, volume-bias resistant
- Core Tick–Tock Engine (`core.py`) — PageRank with tournament-strength feedback

The engines share common ideas: graphs from match outcomes, time decay, tournament-strength weighting, and custom teleport vectors.

---

## Notation

- Entities (nodes): teams or players. Denote the set of nodes by \(V\), \(|V|=N\).
- Matches: edges directed from loser to winner. A single match between a losing entity \(i\) and a winning entity \(j\) contributes weight \(w_{ij} > 0\).
- Tournament influence: \(S_t\) for tournament \(t\) (normalized, mean \(=1\)).
- Time decay: for an event at timestamp \(\tau\) with reference time \(T\), the decay factor is \(\exp(-\lambda\,\Delta)\), where \(\Delta = (T-\tau)/86400\) days and \(\lambda = \ln 2 / \text{half\_life\_days}\).
- Damping factor: \(\alpha \in (0,1)\) for PageRank.
- Teleport vector: probability vector \(\rho \in \mathbb{R}^N\) with \(\sum_i \rho_i = 1\).

---

## Core building blocks

### Time and tournament-strength weighting
For each match in tournament \(t\) at time \(\tau\):
\[
\text{match\_weight} = \exp(-\lambda\,\Delta)\, S_t^{\beta}
\]
where \(\beta\) controls how much tournament strength influences the graph.

### Loser→Winner adjacency and normalization
Let \(W_{ij}\) be the sum of match weights from loser \(i\) to winner \(j\). Define per-source totals:
\[
W^{\text{loss}}_i = \sum_j W_{ij},\qquad W^{\text{win}}_i = \sum_k W_{ki}
\]
We form a per-source denominator \(\operatorname{denom}_i\) with smoothing to avoid overconfident sources with few losses:
\[
\operatorname{denom}_i =
\begin{cases}
W^{\text{loss}}_i + \gamma\, W^{\text{win}}_i & \text{(wins-proportional smoothing)}\\
W^{\text{loss}}_i + c & \text{(constant smoothing)}\\
W^{\text{loss}}_i & \text{(no smoothing)}
\end{cases}
\]
Optionally cap the smoothing term so \(\lambda_i = \operatorname{denom}_i - W^{\text{loss}}_i \leq \kappa\, W^{\text{loss}}_i\).

Normalized edge probabilities from \(i\) to \(j\):
\[
P_{ij} = \frac{W_{ij}}{\operatorname{denom}_i}\quad \Rightarrow \quad \sum_j P_{ij} = \frac{W^{\text{loss}}_i}{\operatorname{denom}_i} \le 1.
\]
Per-row deficits \(1 - \sum_j P_{ij}\) are routed to teleport during PageRank.

### Teleport vectors
- Uniform: \(\rho_i = 1/N\).
- Volume-inverse: \(\rho_i \propto 1/\sqrt{\text{loss\_count}_i}\).
- Volume-mix: convex combination of uniform and a volume component \((\epsilon + \text{count}_i)^{\gamma}\) with mixing weight \(\eta\).
- Custom dict: directly specify \(\rho\) over nodes.

After construction, \(\rho\) is strictly normalized and lightly smoothed to avoid zeros.

---

## Core Engine (Tick–Tock) — `core.py`

### High-level algorithm
1. Initialize tournament influences \(S_t \leftarrow 1\) for all tournaments present in the data.
2. Repeat until convergence or max iterations:
   - Tick: build edges using current \(S_t\) and compute PageRank scores \(r\).
   - Tock: recompute \(S_t\) by aggregating current \(r\) over participants per tournament; normalize to mean 1.
3. Return final rankings and retrospective tournament strength metrics.

This produces mutually consistent ratings and tournament strengths.

### PageRank with row deficits
Let \(M\) be the row-stochastic matrix of normalized edges \(P\) (implementation stores row sums separately). With damping \(\alpha\) and teleport \(\rho\), one iteration is
\[
\mathbf{r}^{(k+1)} = (1-\alpha)\,\rho + \alpha\, M^\top \mathbf{r}^{(k)} + \alpha\, \delta\, \rho,
\]
where \(\delta = \sum_i (1 - \sum_j P_{ij}) r^{(k)}_i\) is the total mass deficit from rows that sum to \(\le 1\). This matches per-source smoothing and ensures \(\sum_i r_i = 1\) at convergence.

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
- Decay: half-life days, derived \(\lambda\)
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

## Exposure Log-Odds Engine — `exposure_logodds.py`

### Goal
Remove volume bias by computing separate PageRanks over wins and losses using the same exposure-based teleport. The final score is a smoothed log-odds ratio.

### Match conversion
For players: expand each team match into all loser–winner player pairs. For teams: use team IDs directly. Each converted match has:
- winners list, losers list
- weight \(w = \exp(-\lambda\,\Delta)\, S_t^{\beta}\)
- tournament and match IDs, and a timestamp

For a single match with winners \(\mathcal{W}\) and losers \(\mathcal{L}\), split weight equally across all pairs to preserve total mass:
\[
\text{pair\_share} = \frac{w}{|\mathcal{W}|\,|\mathcal{L}|}.
\]

### Exposure-based teleport
Exposure for node \(i\) is the total sum of all match weights involving \(i\):
\[
E_i = \sum_{m \ni i} w_m,\qquad \rho_i = \frac{E_i + \varepsilon}{\sum_k (E_k + \varepsilon)}.
\]
Using the same \(\rho\) for both win and loss graphs ensures volume effects cancel in the final ratio.

### Win and loss graphs
- Build \(A^{\text{win}}\) with edges loser→winner (as above)
- Define the exact mirrored loss graph by transpose: \(A^{\text{loss}} = (A^{\text{win}})^\top\)

Both graphs are column-normalized to produce transition matrices for PageRank with identical teleport \(\rho\). The engine uses a standard column-stochastic PageRank update with dangling-mass redistribution to \(\rho\).

### Smoothed log-odds score
Let \(\mathbf{s}\) be the win PageRank and \(\mathbf{\ell}\) be the loss PageRank (both sum to 1). Choose a smoothing \(\lambda\) (auto-tuned if unset) and compute the score for each node \(i\):
\[
\text{score}_i = \log \frac{s_i + \lambda\,\rho_i}{\ell_i + \lambda\,\rho_i}.
\]
Auto-tuning sets \(\lambda\) so that \(\lambda\,\operatorname{median}(\rho)\) is a small fraction (e.g., 2.5%) of a typical PageRank mass, stabilizing ratios for low-signal nodes.

### Why volume cancels (intuition and derivation)

- Intuition: Exposure \(E_i\) controls how often node \(i\) appears. We use the same exposure-proportional teleport \(\rho\) for both win and loss PageRanks, and the loss graph is the exact transpose of the win graph. If \(i\)'s volume increases by a factor \(k\) while its conversion quality (win:loss mix against the same opponents) stays the same, then:
  - Edge weights touching \(i\) scale by \(k\)
  - Exposure \(E_i\) and thus \(\rho_i\) scale by \(k\)
  - Both \(s_i\) and \(\ell_i\) scale approximately by \(k\)
  - The smoothed ratio \(\frac{s_i + \lambda\rho_i}{\ell_i + \lambda\rho_i}\) remains invariant to \(k\)

- Sketch derivation: Write PageRank in linear form
  \[
  \mathbf{s} = (1-\alpha)\,(I-\alpha P_{\text{win}}^\top)^{-1}\,\rho,\quad
  \mathbf{\ell} = (1-\alpha)\,(I-\alpha P_{\text{loss}}^\top)^{-1}\,\rho
  \]
  with \(P_{\text{loss}} = P_{\text{win}}^\top\). If all matches involving \(i\) are replicated \(k\) times while maintaining per-opponent fractions, both the column/row of \(P\) associated with \(i\) stay the same (fractions unchanged) but \(\rho_i \propto E_i\) scales by \(k\). Since the solution is linear in \(\rho\), we get \(s_i \mapsto k s_i\) and \(\ell_i \mapsto k \ell_i\).

  Therefore the smoothed log-odds becomes
  \[
  \log \frac{k s_i + \lambda\,k\rho_i}{k \ell_i + \lambda\,k\rho_i}
  = \log \frac{s_i + \lambda\,\rho_i}{\ell_i + \lambda\,\rho_i},
  \]
  cancelling the multiplicative volume factor \(k\). When quality changes (win/loss mix shifts), \(P\) changes and the ratio moves accordingly, which is precisely what we want to measure.

  Caveats and exact cases:
  - The PageRank solutions are probability vectors (sum to 1) and \(\rho\) is normalized, so literal scaling by \(k\) does not occur in practice. The derivation above treats the linear system before re-normalization to explain the cancellation mechanism.
  - If you scale the entire dataset uniformly (duplicate all matches by \(k\)), then \(A\), \(P\), \(\rho\), \(\mathbf{s}\), and \(\mathbf{\ell}\) are unchanged after normalization; the ratio is exactly invariant.
  - If you scale only node \(i\)'s matches and keep per-opponent fractions fixed, \(P_{\text{win}}\) and \(P_{\text{loss}}\) are unchanged; only \(\rho\) shifts toward \(i\). Since both PRs are linear in \(\rho\) but with different propagators, the ratio is not strictly invariant; however the shared \(\rho\) in numerator and denominator substantially attenuates pure exposure effects.
  - In low-signal regimes (or large \(\lambda\)), both numerator and denominator are dominated by the same \(\lambda\,\rho_i\) term, so the ratio tends to 1 (log-odds tends to 0), providing stable behavior for sparse or grinder-only exposure.

#### Toy model with exact cancellation

Assume homogeneous mixing so that for node \(i\):
- Total exposure (incidence mass) is \(e_i\)
- Conversion quality is a fixed \(q_i \in (0,1)\), so win-related mass is \(e_i q_i\) and loss-related mass is \(e_i(1-q_i)\)
- Teleport is exposure-proportional: \(\rho_i = e_i / E\) with \(E = \sum_k e_k\)

Under this simplification, PageRanks are proportional to those masses:
\[ s_i \propto e_i q_i, \qquad \ell_i \propto e_i(1-q_i). \]
With smoothing \(\lambda\), the log-odds used by the engine is
\[
\log \frac{e_i q_i + \lambda\, e_i/E}{e_i (1-q_i) + \lambda\, e_i/E}
\;=\; \log \frac{q_i + c}{1-q_i + c}, \quad c = \lambda/E,
\]
which is independent of exposure \(e_i\). Thus in the idealized case the engine measures only conversion quality (with mild smoothing \(c\)).

### Optional surprisal weighting
During edge construction, an upset-aware weight multiplies \(w\):
\[
U = -\log\big(\max(p_{\text{win}}, 10^{-10})\big),\quad p_{\text{win}} = \frac{1}{1+\exp(-(r_{\mathcal{W}}-r_{\mathcal{L}})/T)},
\]
where \(r_{\mathcal{W}}\) and \(r_{\mathcal{L}}\) are average provisional ratings for winners and losers. Surprisal is iterated a few times to refine ratings.

### Inactivity decay (post)
After computing log-odds, apply an inactivity decay to scores after a grace period \(D\) days:
\[
\text{decay\_factor}(d) = \begin{cases}
1 & d \le D \\
(1-\rho_d)^{(d-D)} & d > D
\end{cases}
\]
with daily decay rate \(\rho_d\). This preserves recent performance while gently decaying stale scores.

### Output columns
- Players: `id`, `player_rank` (log-odds), `win_pr`, `loss_pr`, `exposure`
- Teams: `id`, `team_rank` (log-odds), `win_pr`, `loss_pr`, `exposure`

A convenience `post_process_rankings` can filter by minimum tournaments and assign rank labels.

### Complexity
Two PageRanks of similar size plus preprocessing; overall similar to the core engine’s single PageRank per tick.

---

## Implementation notes

- DataFrames: Polars is used for efficient joins, group-bys, and vectorized transforms. The code avoids double-decay and only computes timestamps once per path.
- Timestamps: engines prefer `last_game_finished_at` if present, else fallback to `match_created_at`, else now.
- Missing participants: byes/forfeits are flagged via `is_bye` and ignored when building edges.
- Numerical hygiene: teleports are strictly normalized with small \(\varepsilon\); all PR vectors are renormalized per iteration; assertions check graph mirroring and probability sums.
- Edge flux: the core PageRank computes a per-edge flux \(\alpha\, r_i P_{ij}\) for diagnostics and visualizations.

---

## Practical guidance

- Start with Exposure Log-Odds for public-facing rankings; set \(\beta \in [0.5, 1]\), keep default \(\alpha\approx 0.85\), and use auto \(\lambda\).
- Use surprisal when you want to highlight upsets; keep iterations small (e.g., 2).
- For the core engine, prefer wins-proportional smoothing with a small \(\gamma\) and a reasonable cap ratio to stabilize low-activity nodes.
- Aggregation `top_20_sum` is a good default for tournament influence; normalize minimum influence if you observe extreme lows.

---

## Minimal examples

Exposure Log-Odds (players):
```python
from rankings import parse_tournaments_data
from rankings.analysis.engine.exposure_logodds import ExposureLogOddsEngine

tables = parse_tournaments_data(tournaments)
engine = ExposureLogOddsEngine(beta=1.0)
rankings = engine.rank_players(tables["matches"], tables["players"])
```

Core Tick–Tock (players):
```python
from rankings import parse_tournaments_data, RatingEngine

tables = parse_tournaments_data(tournaments)
engine = RatingEngine(beta=1.0, influence_agg_method="top_20_sum")
rankings = engine.rank_players(tables["matches"], tables["players"])
```


