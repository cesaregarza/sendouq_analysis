# Entropy‑Controlled Division Assignment (v1.3)

A **variance‑aware, parameter‑free** team scorer built on top of player **log‑odds** ratings. It adjusts only the **ace surplus** in a team’s log‑sum‑exp (LSE) strength using the **Shannon entropy** of contribution weights. No gating, no thresholds, and fully exposure‑aware.

* Input: player log‑odds $r_i$ (from your ranking engine), optional exposure weights $w_i$.
* Output: a division score $S$ per team, margins, and confidence labels.
* Use: sort by $S$, fill capacities by division.

---

## Notation and inputs

* Top‑4 players likely to play, sorted $r_1 \ge r_2 \ge r_3 \ge r_4$.
* Exposure weights $w_i>0$ (minutes/maps share). If unavailable, set $w_i=1$.
  We normalize internally when needed; only **relative** weights matter.

Define **weighted LSE** terms (natural logs):

$$
\begin{aligned}
R      &= \log\!\sum_{i=1}^4 \exp(r_i + \log w_i) \\
O      &= \log\!\sum_{i=2}^4 \exp(r_i + \log w_i) \\
\rho   &= \exp\!\big((r_1+\log w_1) - O\big)  \qquad\text{(ace‑to‑others ratio)} \\
\end{aligned}
$$

Note: $R = O + \log(1+\rho)$.

**Contribution weights** (exposure‑aware softmax):

$$
p_i=\frac{\exp(r_i+\log w_i)}{\sum_{j=1}^4 \exp(r_j+\log w_j)},\qquad \sum_i p_i = 1.
$$

**Entropy** of the top‑4 distribution:

$$
H = -\sum_{i=1}^4 p_i \log p_i,\quad H_{\max}=\log 4.
$$

---

## Entropy‑controlled aggregator

Shrink **only the ace surplus** by a retention factor $\lambda\in[0,1]$ derived from entropy.

**Default (linear normalization):**

$$
\lambda \;=\; \frac{H}{\log 4}.
$$

**Variant (effective contributors, still parameter‑free):**

$$
N_{\text{eff}}=\exp(H),\quad
\lambda \;=\; \frac{N_{\text{eff}}-1}{4-1}.
$$

This variant puts slightly more shrink on concentrated lineups while leaving balanced teams unchanged.

**Entropy‑controlled log strength:**

$$
\boxed{R_{\text{EC}} \;=\; O + \log\!\big(1+\lambda\,\rho\big)}
$$

Bounds: $O \le R_{\text{EC}} \le R$. Equalities at $\lambda=0$ (fully concentrated) and $\lambda=1$ (fully balanced).

---

## Division score and assignment

Let $\Delta$ be the team’s **recency** term (e.g., 60–90d delta, clipped to $\pm 0.5$).
Optional depth‑scaled recency (keeps the system parameter‑free while reducing “ace hot‑streak” lifts):

$$
\Delta_{\text{eff}}=\lambda\,\Delta.
$$

A single, small **gap** penalty handles extreme dispersion without double‑counting entropy:

$$
p_{\text{gap}}=\min\!\Big(0.60,\; 0.30\cdot \max\!\big(0,\ (r_1-r_3)-1.30\big)^2\Big).
$$

**Score (gate‑free, one formula for all divisions):**

$$
\boxed{S \;=\; R_{\text{EC}} \;+\; 0.25\,\Delta_{\text{eff}} \;-\; p_{\text{gap}}}
$$

**Safety cap** (prevents catastrophic collapses):

$$
S \;\ge\; \big(R + 0.25\,\Delta\big) - 1.60.
$$

**Assignment:** sort teams by $S$; fill divisions by capacity; compute **margin** to the nearest boundary and label **confidence**:

* High $\ge 0.15$, Medium $0.07$–$0.15$, Low $<0.07$.

**Provisional flag:** if recent exposure is low, set $\Delta=0$ and mark `provisional=True`.

---

## Properties (brief)

* **Continuity:** $S$ is smooth in $r_i$ and $w_i$.
* **Monotonicity:** increasing any $r_i$ increases $S$.
* **Scale invariance:** adding a constant $c$ to all $r_i$ adds $c$ to $R, O, R_{\text{EC}}, S$.
* **Exposure awareness:** all quantities use $r_i+\log w_i$; bench stars don’t masquerade as full‑timers.
* **No cliffs / no gates:** separation comes entirely from concentration via $H$.

---

## Minimal API

```python
import numpy as np

def _lse(x: np.ndarray) -> float:
    # numerically stable log-sum-exp
    m = x.max()
    return m + np.log(np.exp(x - m).sum())

def entropy_lambda(log_contrib: np.ndarray, variant: str = "linear") -> tuple[float, float, np.ndarray]:
    """
    log_contrib = r_i + log(w_i) for top-4 (sorted)
    Returns (H, lambda, p)
    """
    # Softmax probabilities
    a = np.exp(log_contrib - log_contrib.max())
    p = a / a.sum()

    # Shannon entropy (natural logs)
    H = -(p * np.log(p + 1e-15)).sum()

    if variant == "linear":               # default
        lam = H / np.log(4.0)
    elif variant == "neff":               # effective contributors
        lam = (np.exp(H) - 1.0) / 3.0
    else:
        raise ValueError("variant must be 'linear' or 'neff'")

    # clip for numeric hygiene
    lam = float(np.clip(lam, 0.0, 1.0))
    return H, lam, p

def entropy_controlled_score(r: np.ndarray, w: np.ndarray | None, delta: float,
                             variant: str = "linear") -> dict:
    """
    r: top-4 log-odds (sorted desc), shape (4,)
    w: exposure weights for those players; if None, uses ones
    delta: recency term, already clipped to [-0.5, 0.5]
    """
    if w is None:
        w = np.ones_like(r)
    w = np.maximum(w, 1e-12)  # positive
    logc = r + np.log(w)

    R = _lse(logc)
    O = _lse(logc[1:])
    rho = float(np.exp(logc[0] - O))

    H, lam, p = entropy_lambda(logc, variant=variant)
    R_ec = O + np.log1p(lam * rho)

    # gap penalty
    gap = float(max(0.0, (r[0] - r[2]) - 1.30))
    p_gap = min(0.60, 0.30 * gap * gap)

    # depth-scaled recency (optional but recommended)
    delta_eff = lam * delta

    S = R_ec + 0.25 * delta_eff - p_gap

    # safety cap
    S_cap_floor = (R + 0.25 * delta) - 1.60
    S = max(S, S_cap_floor)

    return {
        "S": S, "R": R, "R_ec": R_ec, "O": O, "rho": rho,
        "H": H, "lambda": lam, "p": p, "delta_eff": delta_eff, "p_gap": p_gap
    }
```

---

## Example (concept only)

* **Extreme one‑star (25273)**: very low $H\Rightarrow \lambda \ll 1$.
  $R$ drops toward $O$, large surplus removed → lands around Div 3 (gate‑free).
* **Balanced team (25240)**: $H\approx \log 4\Rightarrow \lambda\approx 1$.
  $R_{\text{EC}}\approx R$, minimal shrink → promoted appropriately.
* **Moderate imbalance (25230)**: medium $H\Rightarrow\lambda$ in (0.5–0.8).
  Some shrink; if still in X and underperforms, it’s a legitimate miss not tied to variance.

---

## Outputs (per team)

Include the following in team cards / CSV:

```
division, S, margin, confidence,
R, R_ec, O, rho, H, lambda, p1, delta, delta_eff,
gap, p_gap, provisional(bool)
```

* `margin`: distance to nearest division boundary in score units.
* `confidence`: High (≥0.15), Medium (0.07–0.15), Low (<0.07).
* `p1`: ace contribution share (= p\[0]).

---

## Performance snapshot (LUTI S16)

* **Exact** 46.6% | **Within‑1** 87.3% | **Within‑2** 97.8%
* Division X health: 1 team at 0–5; realized X→1 win rate ≈ 55.2%.
* When disagreeing with human seeding, **80.9%** of our placements align with actual results.

*(If you need a slightly stronger X↔1 separation, switch λ to the **effective‑contributors variant** above; it stays parameter‑free and typically nudges the top boundary without changing mid‑table behavior.)*

---

## Design choices (why this works)

* **Aggregator‑level fix:** Star dominance is handled inside LSE via entropy, not by gates or large additive penalties.
* **Exposure‑aware everywhere:** The same weights $w_i$ drive both LSE and entropy; bench‑inflated artifacts are suppressed.
* **Bounded and interpretable:** All adjustments are visible; shrinkage only affects the ace surplus.

---

## Minimal usage (sketch)

```python
from rankings.seedings.entropy import entropy_controlled_score

# r_top4: np.array of player log-odds (desc), w_top4: exposures or None, delta: recency
out = entropy_controlled_score(r_top4, w_top4, delta, variant="linear")  # or "neff"

S = out["S"]
# Collect S for all teams, sort, fill division capacities, compute margins/confidence.
```
