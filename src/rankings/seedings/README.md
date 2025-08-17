# Entropy-Controlled Tournament Seeding System

A **variance-aware, parameter-free** seeding and division assignment system built on top of player **log-odds** ratings. It adjusts only the **ace surplus** in a team's log-sum-exp (LSE) strength using the **Shannon entropy** of contribution weights. No gating, no thresholds, and fully exposure-aware.

* **Input**: player log-odds $r_i$ (from your ranking engine), optional exposure weights $w_i$
* **Output**: entropy-controlled ratings, seeding order, division/bracket assignments
* **Use**: flexible tournament structures, division-based competitions, bracket seeding

---

## Core Concepts

### Notation and Inputs

* Top-N players likely to play (default N=4), sorted $r_1 \ge r_2 \ge r_3 \ge r_4$
* Exposure weights $w_i>0$ (minutes/maps share). If unavailable, set $w_i=1$
* We normalize internally when needed; only **relative** weights matter

Define **weighted LSE** terms (natural logs):

$$
\begin{aligned}
R      &= \log\!\sum_{i=1}^N \exp(r_i + \log w_i) \\
O      &= \log\!\sum_{i=2}^N \exp(r_i + \log w_i) \\
\rho   &= \exp\!\big((r_1+\log w_1) - O\big)  \qquad\text{(ace-to-others ratio)} \\
\end{aligned}
$$

Note: $R = O + \log(1+\rho)$.

**Contribution weights** (exposure-aware softmax):

$$
p_i=\frac{\exp(r_i+\log w_i)}{\sum_{j=1}^N \exp(r_j+\log w_j)},\qquad \sum_i p_i = 1.
$$

**Entropy** of the top-N distribution:

$$
H = -\sum_{i=1}^N p_i \log p_i,\quad H_{\max}=\log N.
$$

---

## Entropy-Controlled Aggregator

Shrink **only the ace surplus** by a retention factor $\lambda\in[0,1]$ derived from entropy.

**Default (linear normalization):**

$$
\lambda \;=\; \frac{H}{\log N}.
$$

**Variant (effective contributors):**

$$
N_{\text{eff}}=\exp(H),\quad
\lambda \;=\; \frac{N_{\text{eff}}-1}{N-1}.
$$

**Entropy-controlled log strength:**

$$
\boxed{R_{\text{EC}} \;=\; O + \log\!\big(1+\lambda\,\rho\big)}
$$

Bounds: $O \le R_{\text{EC}} \le R$. Equalities at $\lambda=0$ (fully concentrated) and $\lambda=1$ (fully balanced).

---

## Tournament Score and Assignment

Let $\Delta$ be the team's **recency** term (e.g., 60–90d delta, clipped to $\pm 0.5$).
Optional depth-scaled recency:

$$
\Delta_{\text{eff}}=\lambda\,\Delta.
$$

A single, small **gap** penalty handles extreme dispersion:

$$
p_{\text{gap}}=\min\!\Big(0.60,\; 0.30\cdot \max\!\big(0,\ (r_1-r_3)-1.30\big)^2\Big).
$$

**Score:**

$$
\boxed{S \;=\; R_{\text{EC}} \;+\; 0.25\,\Delta_{\text{eff}} \;-\; p_{\text{gap}}}
$$

**Safety cap** (prevents catastrophic collapses):

$$
S \;\ge\; \big(R + 0.25\,\Delta\big) - 1.60.
$$

**Assignment Options:**

1. **Division-based**: Sort teams by $S$; fill divisions by specified capacities
2. **Bracket seeding**: Create tournament brackets of various sizes
3. **Custom structures**: Define your own tournament format

**Confidence Metrics:**
* High $\ge 0.15$, Medium $0.07$–$0.15$, Low $<0.07$ (margin to boundary)
* **Provisional flag**: if recent exposure is low, set $\Delta=0$ and mark `provisional=True`

---

## Properties

* **Continuity:** $S$ is smooth in $r_i$ and $w_i$
* **Monotonicity:** increasing any $r_i$ increases $S$
* **Scale invariance:** adding a constant $c$ to all $r_i$ adds $c$ to $R, O, R_{\text{EC}}, S$
* **Exposure awareness:** all quantities use $r_i+\log w_i$; bench stars don't masquerade as full-timers
* **No cliffs / no gates:** separation comes entirely from concentration via $H$

---

## Usage Examples

### Basic Seeding
```python
from rankings.seedings.entropy_seeding import EntropySeedingSystem

# Initialize the system
seeder = EntropySeedingSystem(top_n=4, entropy_variant="linear")

# Compute ratings from team data
teams_df = seeder.compute_team_ratings(teams_data, skill_field="rating")

# Get seeding order
seeded_teams = seeder.get_seeding_order()
```

### Division Assignment
```python
# Define custom division structure
division_config = [
    ("Premier", 8),
    ("Division 1", 16),
    ("Division 2", 32),
    ("Division 3", 64),
]

# Assign teams to divisions
divisions = seeder.assign_divisions(
    division_config=division_config,
    overflow_division="Open"
)

# Get division statistics
stats = seeder.get_division_statistics()
```

### Tournament Brackets
```python
# Create bracket structure
bracket_sizes = [8, 8, 16, 16, 32, 32]
brackets = seeder.create_brackets(
    bracket_sizes=bracket_sizes,
    naming_pattern="Bracket {}"
)
```

### Working with DataFrames
```python
# From existing polars DataFrames
teams_df = pl.DataFrame({"team_id": [...], ...})
players_df = pl.DataFrame({"team_id": [...], "score": [...], ...})

# Compute ratings
team_ratings = seeder.compute_from_dataframe(
    teams_df=teams_df,
    players_df=players_df,
    skill_col="score",
    exposure_col="exposure_weight"  # optional
)
```

### Evaluation and Adjustments
```python
# Evaluate against actual divisions
accuracy = seeder.evaluate_accuracy(
    actual_divisions=actual_df,
    actual_div_col="true_division"
)

# Apply manual adjustments if needed
adjustments = {"team_123": 0.5, "team_456": -0.3}
adjusted = seeder.apply_manual_adjustments(adjustments)
```

### Low-level API
```python
from rankings.seedings.entropy_seeding import (
    compute_entropy_controlled_rating,
    compute_team_entropy
)

# Direct computation for a single team
skills = [2.5, 2.1, 1.8, 1.5]  # Player log-odds
exposures = [1.0, 0.9, 0.8, 0.7]  # Optional weights

rating, debug_info = compute_entropy_controlled_rating(
    skills, exposures, top_n=4
)

# Just entropy calculation
H, lambda_val, p_dist = compute_team_entropy(skills, exposures)
```

---

## API Reference

### Main Class: `EntropySeedingSystem`

**Constructor:**
- `top_n` (int): Number of top players to consider (default: 4)
- `entropy_variant` (str): 'linear' or 'effective' (default: 'linear')

**Methods:**
- `compute_team_ratings(teams, skill_field, exposure_field)`: Compute ratings for all teams
- `compute_from_dataframe(teams_df, players_df, ...)`: Compute from polars DataFrames
- `assign_divisions(division_config, overflow_division)`: Assign to divisions
- `get_seeding_order()`: Get teams in seeding order with seeds
- `create_brackets(bracket_sizes, naming_pattern)`: Create tournament brackets
- `get_division_statistics()`: Get per-division statistics
- `evaluate_accuracy(actual_divisions, ...)`: Compare against true divisions
- `apply_manual_adjustments(adjustments, ...)`: Apply manual rating adjustments

### Utility Functions

- `compute_entropy_controlled_rating(skills, exposures, top_n)`: Compute single team rating
- `compute_team_entropy(skills, exposures)`: Calculate entropy metrics
- `assign_divisions(teams, rating_column, division_config, overflow_division)`: Standalone division assignment

### Backward Compatibility

The old `EntropyDivisionAssigner` class name is aliased to `EntropySeedingSystem` for backward compatibility.

---

## Performance and Flexibility

* **Tested accuracy**: High correlation with human seeding decisions across various tournament formats
* **Adaptable**: Works with any division structure, bracket sizes, or custom tournament formats
* **Parameter-free**: No manual tuning required for different competition types
* **Entropy variants**: Choose between 'linear' (default) or 'effective' (stronger separation at boundaries)

---

## Design Philosophy

* **Aggregator-level fix:** Star dominance is handled inside LSE via entropy, not by gates or large additive penalties
* **Exposure-aware everywhere:** The same weights $w_i$ drive both LSE and entropy; bench-inflated artifacts are suppressed
* **Bounded and interpretable:** All adjustments are visible; shrinkage only affects the ace surplus

---

## Output Format

Include the following in team cards / CSV:

```
division, S, margin, confidence,
R, R_ec, O, rho, H, lambda, p1, delta, delta_eff,
gap, p_gap, provisional(bool)
```

* `margin`: distance to nearest division boundary in score units
* `confidence`: High (≥0.15), Medium (0.07–0.15), Low (<0.07)
* `p1`: ace contribution share (= p[0])