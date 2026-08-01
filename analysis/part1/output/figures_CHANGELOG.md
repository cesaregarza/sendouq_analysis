# Part I figure changelog

All figures now use a fixed entity palette, 1520-pixel PNG exports, matching SVGs, and identical dark/light layouts. The top-level figure files mirror the dark variants for backward-compatible essay paths.

Export QA passed for all 13 figures on both surfaces. The closest adjacent entity pair remains separated by 0.241 in the Machado severity-100 deuteranopia diagnostic, and every system also has a redundant marker or dashed-line encoding. Both complete surface contact sheets and dense panels were inspected at final resolution. All 52 dark/light PNG/SVG exports were byte-identical across two consecutive report rerenders after fixing SVG metadata and hash salts. Machine-readable checks are in `results/figure_validation.json`.

| Figure | Change | Why |
|---|---|---|
| 01_future_prediction | Added tournament-cluster 95% intervals, fixed entity colors, and the 0.027 LOOPR–ordinal bracket. | Shows uncertainty without recoloring systems by performance. |
| 02_loopr_calibration | Added Wilson intervals and moved bin counts to the subtitle. Annotated the low-probability bin. | The actual upper extreme is close and very slightly overconfident, so the requested plural “underconfident at the extremes” would be inaccurate. |
| 03_plus_and_shuffle | Replaced truncated AUC bars with suggested-only tier dots; incumbents are a separate open-marker group. Split the two nulls into small multiples. | Keeps the institutional decisions distinct and prevents overlapping null histograms. The tier+status statistic exceeds 199 of 200 shuffles—not all 200—so the annotation reports the observed count. |
| 04_monthly_stability | Replaced retention bars with a zero-based line/dot panel; added movement and retained-player Spearman. | Membership and ordering are different operational properties. The 60% dip is annotated from a dedicated audit: 39 of 40 departures aged out of the 90-day activity window. |
| 05_leave_one_tournament_out | Preserved the count and disruption forms, labeled every retention count, and named Triton-Cup 131 at the curve head. | Makes the small 96–99 counts and the worst p90 event readable. |
| A1_half_life_tradeoff | Used actual day spacing, separated no-decay, and marked the shipped 180-day setting on both panels. | Shows a tradeoff neighborhood instead of implying equally spaced categories or a unique optimum. |
| A2_unfamiliar_lineups | Replaced grouped bars with system slopes, included OpenSkill μ, and added novel-lineup cluster intervals. | The slope is the result: LOOPR approaches chance on novel combinations while ordinal does not. |
| N1_rank_bootstrap_intervals | Added a 50-player anonymized caterpillar from full tournament-cluster LOOPR recomputations. | Makes rank uncertainty visible where the published list suppresses it. Two hundred repetitions are used because each draw recomputes the complete pipeline; all samples and draws are cached. |
| N2_incumbent_autopsy | Used the pre-registered binary-outcome fallback: score versus pass/fail, faceted by status and sized by exposure. | Every historical Plus CSV in the repository lacks candidate-level vote percentages or vote counts; no continuous value was imputed. “Never stood for vote” is also unavailable. |
| N3_auc_by_exposure_band | Added within-tier macro AUC across outcome-independent pre-vote exposure tertiles. | Separates limited-evidence behavior from the pooled institutional mix. |
| N4_core0_diagnostic | Added core-0 log loss for LOOPR, OpenSkill μ, and ordinal with cluster intervals. | Isolates ordinal’s μ−3σ uncertainty penalty from additional match information. |
| N5_cross_system_event_deletion | Added typical and maximum rank movement for all five systems over 25 predeclared important events. | Makes bounded influence comparable in rank units. The maximum panel uses a labeled log scale so the 14 versus 33–47 comparison remains readable beside win rate’s 676-place shift. |
| N6_ablation_walk | Added the cumulative component walk for prediction loss and top-100 Spearman with source intervals and requirement labels. | Shows which requirements the historical backtest can and cannot see. |

## Deviations from the requested wording or form

- **02_loopr_calibration:** the annotation says “underconfident at the low extreme,” not “at the extremes,” because the upper bin is slightly overconfident.
- **03_plus_and_shuffle:** the tier+status statistic is labeled as exceeding 199 of 200 shuffles, because one shuffled value was higher. The within-tier statistic does exceed all 200.
- **N2_incumbent_autopsy:** vote percentage and “never stood for vote” are absent from every historical export, so the pre-registered binary pass/fail fallback is shown and no values are imputed.
- **N5_cross_system_event_deletion:** the all-system comparison covers the 25 predeclared dense-center events available in the intended-use sensitivity table, not all 665 events. The original 665-event LOOPR-only analysis remains in figure 05. The maximum panel uses a log rank axis to keep the 14 versus 33–47 comparison legible beside win rate’s 676-place shift.

No other figure required a substantive deviation.

No computed point estimate or existing evaluation inclusion rule was changed. Publication-only intervals and new analyses are recorded in `evaluation_manifest.json`.
