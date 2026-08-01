# Implementation decisions fixed before results

This file resolves computational details that were not fully specified in
`PREANALYSIS_PLAN.md`. It is committed before the new result stages run.

## Time windows

- Graph-stratified prediction uses the exact frozen walk-forward events and the
  existing report window beginning 2025-08-17.
- Model-to-seed comparison uses the same 2025-08-17 start. Events still need to
  pass the structural seed audit.
- Plus decomposition uses the frozen 2025-11-16 peer vote.
- Stability uses the same month-end snapshots as the prior audit from
  2025-01-31 through the data cutoff.

## Historical seed rosters

Primary model-to-seed comparison requires exactly four ids in the stored
`activeRosterUserIds` field for every compared team. If that field is absent,
an exactly-four-member roster is accepted. Tournaments with larger or
ambiguous rosters remain in the seed-quality audit but not the primary model
comparison. This prevents model-specific “best four” selection.

## Comparative tournament removal

The prior audit retains its full 665-event LOOPR leave-one-tournament-out run.
Exact all-system recomputation is added for 25 structurally important events,
selected without scores or outcome direction:

1. calculate the final-cutoff opponent graph and active player set;
2. count player-match appearances by active players in the 8-core for each
   tournament;
3. take the 25 events with the largest count, breaking ties by total valid
   matches and tournament id.

The active comparison universe is held fixed at its full-data definition when
an event is removed. This isolates rating sensitivity from the separate fact
that removing an event can make a player fail a 90-day activity rule. The
original LOOPR analysis, which reports both effects, remains unchanged.

## Ranking ties and missing scores

- LOOPR, Bradley-Terry, and 90-day win rate use a neutral score of zero for a
  player absent from a fitted score map.
- OpenSkill uses its population prior: 25 for `mu` and 0 for the conservative
  ordinal.
- Top-k membership uses player id as a deterministic final tie breaker.
- Pairwise concordance awards half credit to exact score ties and reports their
  frequency.

## Uncertainty

- New paired predictive and seeding comparisons use 2,000
  tournament-cluster bootstrap draws.
- Adversarial interventions are deterministic sensitivity analyses. Their
  across-target distributions are not labeled confidence intervals.
- Support thresholds stop at the first predeclared level with fewer than five
  tournaments or 200 matches; lower supported thresholds remain reported.

## Computational fallback

If fewer than the preregistered number of attack lineups satisfy an attack's
criteria, use every qualifying lineup and report the shortfall. Do not relax a
win-rate, exposure, lineup-frequency, or graph-core threshold after seeing
system behavior.
