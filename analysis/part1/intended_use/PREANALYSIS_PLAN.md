# Intended-use evaluation: frozen analysis plan

**Frozen from prior audit:** commit `f0656220fcd8c3db61c9f7cad408b0d5a1be92a3`  
**Random seed:** `20260731`  
**Status:** definitions below are fixed before inspecting any new model-by-support,
seeding-agreement, or adversarial result.

## Question

The prior audit asked whether LOOPR contains out-of-sample information about
future match outcomes. That remains a valid generalization test. This addendum
asks a different question:

> Does LOOPR have useful properties for ordering established competitive
> players in a public ranking, even if another system produces better average
> match probabilities?

No single statistic is defined as “public-ranking quality.” The intended-use
claim will be evaluated as a requirements profile: ordinal evidence in the
dense graph, convergence with pre-existing institutional judgments, stability,
leave-one-event robustness, schedule response, and degradation under deliberate
leaderboard attacks. A favorable result on one requirement cannot compensate
silently for failure on another.

## Systems

Every comparative intended-use analysis will include the same five systems:

1. LOOPR, production configuration from the frozen manifest.
2. OpenSkill posterior mean (`mu`).
3. OpenSkill conservative ordinal (`mu - 3 sigma`).
4. L2-regularized batched team Bradley-Terry, using the frozen audit's
   regularization and player-identity design.
5. Trailing 90-day player win-rate logit, with the frozen audit's smoothing.

Chance remains in the all-match generalization table but is not a meaningful
public-ranking comparator. Scores will always be computed from information
strictly before the evaluated event or intervention. Team score is the mean of
the four participating player scores for every system. Alternate sum and
top-four roster aggregates are sensitivity checks only.

## Support is defined without ratings or outcomes

At each tournament cutoff, construct an undirected **opponent graph** from all
eligible earlier matches. Each player is a node. A cross-team player pair gets
an edge when the players have opposed one another. Match direction and model
scores are not used. Repeated meetings supply edge weights only for exposure
and diversity measures.

For each player, calculate:

- `prior_matches`: number of earlier match appearances.
- `unique_opponents`: distinct earlier opposing players.
- `effective_opponents`: inverse-Simpson effective opponent count,
  `(sum_j w_j)^2 / sum_j(w_j^2)`, where `w_j` is the number of meetings with
  opponent `j`.
- `core_number`: the player's k-core number in the simple opponent graph.
- `component_size` and `in_largest_component`.
- `nonbridge_edge_share`: share of incident opponent edges that are not graph
  bridges; these edges have at least one alternate graph route.
- `two_hop_redundancy`: `1 - distinct_two_hop_nodes / two_hop_paths`, set to
  zero when there are fewer than two two-hop paths. Larger values mean that
  multiple paths reach the same surrounding players.
- `unique_teammates` and `effective_lineups`, as participation-only supporting
  measures.

`core_number` is the primary support axis because it has a direct topological
meaning: a player in the k-core belongs to a recursively defined subgraph in
which every retained player has at least `k` distinct opponent connections.
The other measures test whether any conclusion is peculiar to k-core.

### Predeclared support cuts

Use nested k-core thresholds `k >= 1, 2, 4, 8, 16, 32`, stopping when coverage
falls below five held-out tournaments or 200 matches. Powers of two span graph
density without choosing a cutoff from model performance.

- A held-out match meets threshold `k` only when **all participating players**
  have pre-event `core_number >= k`. The minimum participant core is therefore
  the primary match-support measure; median participant core is a sensitivity
  analysis.
- A player pair meets threshold `k` only when both players meet it.
- A seeded team comparison meets threshold `k` only when every rostered player
  used in both team scores meets it; a sensitivity analysis uses the minimum
  of the two team medians.

Coverage is reported at every threshold. Thresholds will not be moved because
a result becomes favorable or unfavorable.

## Generalization estimands

The frozen all-match prediction results remain unchanged and primary for the
question “does this score generalize to future competition?”

For graph-support analysis, report at every supported k-core threshold:

- match-micro and tournament-macro log loss;
- uncalibrated ordering accuracy (`raw_difference > 0`);
- Brier score as a secondary probability metric;
- each baseline's paired tournament-level log-loss and ordering-accuracy
  difference versus LOOPR;
- tournament-cluster bootstrap intervals; and
- match and tournament coverage.

To test whether relative performance changes continuously with support, regress
the per-match paired loss difference `(baseline loss - LOOPR loss)` on
`log2(1 + minimum participant core)`. Estimate uncertainty by resampling
tournaments, not individual matches. A positive slope means LOOPR improves
relative to that baseline as support rises; it does not by itself establish
that LOOPR is best in the dense core.

## Historical tournament-seed audit

The raw `team.seed` field is not assumed to be expert judgment. Before it can
support a model comparison, audit:

1. non-null coverage, uniqueness, contiguity, and team/roster coverage;
2. stability of team seeds across duplicate historical snapshots;
3. tournament format and whether `seed` plausibly represents a one-dimensional
   strength order (rather than group assignment or a later-stage slot);
4. agreement with registration order (`team.createdAt`);
5. agreement with the stored `avgSeedingSkillOrdinal` field;
6. higher-seed win rate and final-performance association, overall and by seed
   band; and
7. evidence that only top seeds, rather than the entire list, carry information.

The repository does not contain a direct “manual/random/copied” provenance
field. The report will distinguish observed patterns from provenance:

- `registration-order-equivalent`: exact seed order equals registration order.
- `automatic-ordinal-equivalent`: exact seed order equals the ordering implied
  by complete `avgSeedingSkillOrdinal` values.
- `non-default-like`: neither exact ordering applies.
- `unstable`: at least one common team changes seed across available copies.
- `provenance unknown`: the data cannot distinguish manual seeding from a
  randomization, copied order, private tool, or undocumented organizer process.

The phrase **manual seed** will not be used unless repository evidence directly
establishes manual assignment.

### Structurally eligible seed comparison

Model-to-seed comparison is restricted before examining model agreement to
tournaments that:

- are eligible under the frozen ranked-tournament protocol;
- have at least eight teams with four scoreable roster members;
- have a complete, unique, contiguous `1..N` seed permutation;
- have stable seeds across every available copy containing the same teams; and
- begin with one single- or double-elimination bracket, or one single-group
  Swiss bracket. Multi-group and progression-derived seed numbers are excluded
  from the primary comparison because one numeric seed may encode bracket
  placement rather than a global strength order.

Results are shown separately for registration-order-equivalent,
automatic-ordinal-equivalent, and non-default-like events. They are never pooled
silently.

For each model, report tournament-macro Kendall tau-b, all-pair concordance,
top-eight pair concordance, and tournament-cluster intervals. For pairs on which
LOOPR and another model disagree, report the share for which the historical seed
supports each ordering, ties/exclusions, and a tournament-cluster interval for
the paired difference. Seeding agreement measures concordance with the stored
organizer-controlled artifact, not true skill or seeding quality.

If the structural audit leaves fewer than ten events or if seed stability is
poor, seeding will remain a data-quality finding and will not become primary
validation evidence.

## Intended-use robustness on a common player universe

Monthly and final-cutoff comparisons use the frozen activity definition:
rostered for an eligible event with a valid match in the preceding 90 days.
Players must have at least one earlier resolved match. Missing scores receive
the system's documented population prior; ties are broken by player id only for
deterministic membership calculations and are reported as ties for pairwise
agreement.

For every system, report:

- adjacent-month top-100 retention;
- Spearman correlation and median/p90 movement among retained players;
- leave-one-tournament-out top-100 retention and rank movement;
- worst event and largest shift; and
- the same summaries restricted to players with `core_number >= 8` and
  `core_number >= 16`.

Stability and robustness are operational properties, not correctness metrics.
A system can be stably wrong or robustly uninformative.

## Adversarial protocols

Interventions use the same underlying synthetic matches for all five systems.
Targets and opponents are selected without LOOPR or OpenSkill scores.

### Common target pool

At the final frozen cutoff, eligible targets must be active, have at least 20
earlier matches, be in the 8-core, and belong to an observed four-player lineup
with at least three prior appearances. When more than 20 lineups qualify, choose
20 by a stable hash of the sorted player ids. Report the full selected list.

### Weak-win farming

Define weak opponent lineups using only pre-intervention outcomes: at least ten
matches in the trailing 90 days, win rate at or below 25%, and a complete
four-player observed lineup. Define strong controls analogously with win rate at
or above 75%. Add `1, 5, 20, 50` wins for each target lineup against one weak
lineup in a new synthetic event at the cutoff; repeat with a strong opponent.
Recompute every full rating pipeline after each intervention.

Report median and p90 target rank/percentile gain, top-100 entries, score change
in within-system standard-deviation units, and the weak-win gain divided by the
strong-win gain. Smaller weak-win movement is evidence of farming resistance;
failure to reward strong wins is not counted as success.

### Volume without better results

For each target lineup, duplicate a deterministic balanced block of matched
opponent results so wins and losses, opponent identities, and timestamps retain
their observed proportions at `2x` and `5x` volume. Report rank and standardized
score movement. OpenSkill ordinal is expected to react to reduced uncertainty;
that response is described, not automatically labeled a defect.

### Evidence thinning

Remove 10%, 25%, and 50% of pre-cutoff matches incident to target players using
ten deterministic hash seeds. Recompute all systems and report top-100
retention, median/p90 rank movement, and pairwise ordering retention in the
8-core. This is a missing-evidence stress test, not a confidence interval.

### Schedule and transitivity

Construct equal-record interventions from the observed graph: one target lineup
receives the same win/loss count against strong-control lineups and another
receives it against weak-control lineups. Report the ordering and standardized
score gap for every system. A separate three-lineup chain adds A-over-B and
B-over-C evidence without A-C matches and tests whether the induced A-C order is
in the intended direction. These are requirement tests, not estimates of real
player skill.

## Plus decomposition

Retain the frozen institutional pooled/tier-macro AUC and the frozen within-tier,
within-status pair-weighted AUC. Do not describe the latter as cleaner. It asks a
narrower question and deliberately discards cross-status and cross-tier parts of
the actual institution.

Add:

1. exact passer-failer pair decomposition by same/different tier and
   same/different candidate status;
2. tier-only, status-only, tier-plus-status, activity-only, graph-support-only,
   and combined baseline scores evaluated out of sample;
3. LOOPR incremental performance over those baselines;
4. LOOPR AUC at the predeclared k-core thresholds with coverage; and
5. the existing 200-repetition outcome-shuffle null, stratified by graph support
   when cell sizes permit.

The exact pair decomposition is the primary explanation of pooled AUC because
pooled AUC is a weighted average over passer-failer pair types. Cross-validated
log loss is used for incremental baseline models because AUC differences do not
form an additive variance decomposition.

## Interpretation rules

- Average log loss measures probabilistic forecasting, not public-ranking
  legitimacy and not the whole intended-use construct.
- High-support match ordering still measures future team outcomes; it is closer
  to intended use than whole-graph log loss but is not direct player-level truth.
- Seed agreement measures concordance with a historical tournament artifact.
- Plus agreement measures concordance with a consequential community process.
- Stability, LOTO, and attacks measure operational behavior, not correctness.
- No result licenses “true skill,” exact-pair validity, or leaderboard
  legitimacy.
- Claims will be weakened when support trends are flat, seeds are untrustworthy,
  or simpler systems match LOOPR on the intended-use requirements.

## Planned outputs

- `seed_audit.csv` and a seed provenance/quality table.
- `graph_support_player_tournament.parquet`.
- `prediction_by_graph_support.csv` and paired trend estimates.
- `seeding_model_agreement.csv` and disagreement-pair results, if eligible.
- all-system monthly stability and leave-one-event tables.
- weak-win, volume, thinning, schedule, and transitivity attack curves.
- Plus pair decomposition and baseline table.
- one additive technical report; no prior artifact is deleted or rewritten.
