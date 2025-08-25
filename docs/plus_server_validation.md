# Plus Server Validation — Outcome-Only Rankings vs Peer Voting

This document consolidates the validation of an outcomes‑only ranking engine against Plus server voting. It provides context, methodology, results with confidence intervals, calibration and thresholds, stratified evidence, and limitations — without naming specific players.

## What Are the Plus Servers?
- Community‑run, invite‑only “Plus” servers where players are evaluated by peers for admission at three tiers:
  - +1: highly skilled; often selective or semi‑retired
  - +2: very strong and typically active
  - +3: strong with broader, more heterogeneous membership
- Voting is independent from any ranking model and reflects human judgements of latent skill. It can be influenced by recency, reputation, and participation.

## Objective
- Test how well a bottom‑up, outcomes‑only ranking signal (Exposure Log‑Odds + Tick‑Tock) aligns with Plus voting.
- Use ROC AUC (with 95% CIs) to measure how often a random pass is scored above a random fail, overall and per tier.

## Data & Filters
- Source: local scraped tournaments (`data/tournaments`).
- Ranked events only: filtered via metadata and minimum size.
- Temporal freeze: include only tournaments with `start_time < 2025‑08‑17 00:00:00 UTC` (up to and including 2025‑08‑16 UTC) to avoid leakage.
- Active‑only: exclude players with `last_active > 90 days` before the cutoff to reduce identifiable noise from long inactivity.
- Plus voting CSV: `sendou_plus_voting_results.csv` (columns: `id`, `username`, `pass_or_fail`, `tier` ∈ {1,2,3} corresponding to +1/+2/+3). Names are not surfaced in this doc.

## Ranking Engine (fixed)
- Half‑life: 180 days; Damping (alpha): 0.85
- Inner score: log‑odds of win/loss PageRanks with exposure baseline (auto‑lambda)
- Tick‑Tock: convergence_tol=0.01; max_ticks=5; influence_method: log_top_20_sum
- Inactivity score decay: delay=180 days; rate=0.01 (mild)
- Post‑processing: no inactivity drop; `min_tournaments=0`; `display_score = 25 × score`

## Method Summary
1) Parse tournaments → filter ranked → freeze to cutoff.
2) Rank players (fixed config) and compute `display_score`.
3) Join rankings to Plus voting by player id.
4) Active‑only filter (≤ 90 days since last activity).
5) Compute ROC AUC overall (micro) and per tier; report macro AUCs (mean across tiers), all with 95% bootstrap CIs.

## Main Results (frozen, active‑only ≤ 90d)
- Matched: 424/511 voting rows
- Overall AUC (micro): 0.7424 (95% CI 0.693–0.789)
- Macro AUC (unweighted): 0.8045 (95% CI 0.752–0.851)
- Macro AUC (weighted by samples): 0.8075 (95% CI 0.762–0.849)
- Per tier:
  - +1 (tier=1): 0.7690 (95% CI 0.634–0.885) [n=72]
  - +2 (tier=2): 0.8469 (95% CI 0.782–0.909) [n=127]
  - +3 (tier=3): 0.7977 (95% CI 0.734–0.854) [n=225]

Interpretation: Agreement is strong across tiers, especially +2. Filtering long‑inactive players improves stability and alignment.

## Behavioral Interpretation by Tier (Plausible Hypothesis)
- +1 (lower AUC): Cohort includes selective/semi‑retired talent; human votes can reflect reputation and sporadic activity, creating some mismatch with outcomes‑only signals.
- +2 (highest AUC): Cohort is both selective and engaged; outcomes capture current form well, so human votes and rankings align strongly.
- +3 (good AUC, more noise): Larger, heterogeneous pool; more variability in recency and consistency, but alignment remains strong.

These are plausible and consistent with the observed stratified evidence below; further validation recommended.

## Stratified Evidence (Active‑Only ≤ 90d)
- By recency (days since last active):
  - +1: 0–30d AUC ≈ 0.79; older slices sparse or lower (small N).
  - +2: strong across buckets; best at 0–30d ≈ 0.86.
  - +3: solid at 0–30d ≈ 0.81; small‑N volatility at 61–90d.
- By participation (unique tournaments in last 90d):
  - +1: AUC rises with participation (3+ ≈ 0.82; 1–2 tiny N).
  - +2: high AUC in both 1–2 and 3+ (≈ 0.84–0.89).
  - +3: 3+ ≈ 0.81; 1–2 ≈ 0.70 (noisier at lower volume).

## Calibration & Thresholds (Youden’s J)
- Platt scaling per tier (logistic mapping from `display_score` → P(pass)).
- Expected Calibration Error (ECE; 10 bins) and Youden’s J optimal thresholds:
  - +1: ECE ≈ 0.017; J‑prob threshold ≈ 0.732 (95% CI 0.590–0.916); score threshold ≈ 38.6 (95% CI 29.4–53.2)
  - +2: ECE ≈ 0.075; J‑prob ≈ 0.658 (0.294–0.764); score ≈ 26.1 (16.1–29.4)
  - +3: ECE ≈ 0.097; J‑prob ≈ 0.646 (0.420–0.731); score ≈ 7.0 (−4.0–10.2)

These thresholds summarize where the ranking best separates pass/fail within a tier. Use with care: calibration quality varies by tier and sample size.

## Error Analysis Framework (No Names)
- Definitions:
  - Type I (False Positive): vote = fail, but score above the tier’s pass threshold.
  - Type II (False Negative): vote = pass, but score below the tier’s pass threshold.
- We maintain anonymized candidate lists internally; this doc does not disclose names. For each candidate, the top matches by influence are identified via leave‑one‑match‑out (LOO) analysis to explain mismatches (e.g., a few high‑impact wins/losses near the cutoff date).

## Limitations
- Voting is a noisy proxy of latent skill (reputation effects, variable engagement). Active‑only filtering reduces — but doesn’t eliminate — these effects.
- Cross‑tier thresholds differ; micro AUC is less comparable than macro. Small‑N slices inflate uncertainty.
- This is a cross‑section at a fixed cutoff. Temporal snapshots before the voting window can further validate causality.
