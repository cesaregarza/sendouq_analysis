# Data and methodology audit

## Bottom line

The historical 0.526 LOOPR prediction figure, the 0.497/0.489 ablation figures, and the 0.772/0.779 Plus figures are reproducible from saved outputs—but they are not estimates from one common protocol. The draft’s 0.86 Plus AUC has no recoverable saved protocol and should be removed. The revised analysis uses deterministic tournament canonicalization, explicit timestamp fallback, strict event-by-event temporal cutoffs, prior-only calibration, and paired tournament-cluster uncertainty.

## Available data

- 2,251 canonical tournament records selected from 2,686 JSON records across 181 files.
- 665 eligible ranked tournaments and 28,647 valid decided non-bye matches from 2024-05-13 through 2026-01-28.
- 5,518 players resolved into valid matches; 5,622 unique roster players.
- Match timestamps, team IDs, winners and losers, tournament metadata, rosters, and partial appearance/substitution payloads.
- Historical Plus peer-vote snapshots with tier, suggested/incumbent status, and pass/fail outcome.
- LOOPR, OpenSkill, win-rate, evaluation scripts, saved ablations, and the two essay drafts.

The complete file-level inventory of relevant prior scripts, notebooks, saved tables/models, reports, figures, vote files, implementation files, and environment files is in [`audit/project_file_inventory.csv`](audit/project_file_inventory.csv), with byte counts and SHA-256 hashes. Raw tournament JSON is represented by its aggregate plus the canonical record-source table.

## Missing or incomplete data

- Vote percentages are absent from the available historical CSVs; continuous vote-percentage analysis is therefore omitted.
- Exact appearances are available on both sides for only 2,762 matches (9.6%). The other 25,885 matches use tournament-team rosters, so unrecorded substitutions cannot be recovered.
- 2,385 matches require tournament-start timestamp fallback after checking finish and creation times.
- Reliable exogenous labels for “weak,” skill-capped, local, or major events are unavailable; the tournament-weighting property test is consequently endogenous and limited.
- There is no objective player-skill vector. That is a construct boundary, not a missing field to fabricate.

## Duplicate and timestamp decisions

The raw tree contains 365 duplicated tournament IDs and 435 extra copies. Of those IDs, 194 have differing contents. The revised loader selects the most-complete record deterministically and merges supplemental appearance rows. The legacy recursive loader is retained only for reproduction.

The repository converter assigned missing match times a global/latest timestamp. That is invalid for historical cutoffs. The revised priority is exact finish time, then match creation time, then tournament start. Counts are recorded in the machine-readable manifest.

## Reproduction ledger

| result | displayed | recomputed | reproduced | protocol | explanation |
| --- | --- | --- | --- | --- | --- |
| LOOPR prediction log loss (January harness) | 0.526394 | 0.526394 | True | 3 sampled temporal blocks x 10 tournaments; ratings frozen within each block; match-iid bootstrap | Weighted aggregation of the three saved split results reproduces 0.526394. Raw rebuild is version-sensitive. |
| LOOPR + exposure teleport: prediction log loss | 0.496653 | 0.496653 | True | ablation script's hand-built variant; same 3x10 block structure | 0.497/0.497/0.489 are different model rows, not reruns of the production harness. |
| LOOPR + exposure teleport: Plus macro AUC | 0.771593 | 0.771593 | True | active <=90d; November 2025; unweighted mean of tier AUCs | 0.772 is the exposure/smoothing row; 0.779 is the later tick-tock row. |
| LOOPR + smoothing: prediction log loss | 0.496540 | 0.496540 | True | ablation script's hand-built variant; same 3x10 block structure | 0.497/0.497/0.489 are different model rows, not reruns of the production harness. |
| LOOPR + smoothing: Plus macro AUC | 0.771650 | 0.771650 | True | active <=90d; November 2025; unweighted mean of tier AUCs | 0.772 is the exposure/smoothing row; 0.779 is the later tick-tock row. |
| LOOPR + tick-tock: prediction log loss | 0.488770 | 0.488770 | True | ablation script's hand-built variant; same 3x10 block structure | 0.497/0.497/0.489 are different model rows, not reruns of the production harness. |
| LOOPR + tick-tock: Plus macro AUC | 0.779160 | 0.779160 | True | active <=90d; November 2025; unweighted mean of tier AUCs | 0.772 is the exposure/smoothing row; 0.779 is the later tick-tock row. |
| Draft Plus AUC | 0.860000 | — | False | not specified in draft | No saved table or script output in the project identifies a pooled/tiered protocol that yields the draft's 0.86. It must not be retained as an auditable result. |

### Discrepancy reconciliation

- **0.526 versus about 0.499:** 0.526394 is the saved January production harness, using three sampled temporal blocks of ten tournaments with ratings frozen inside each block. The roughly 0.497 and 0.489 numbers are separate hand-built ablation rows. They are not reruns of one model under one protocol.
- **0.772 versus 0.779:** 0.772 is the exposure/smoothing variant; 0.779 is the later tick-tock variant.
- **0.86 Plus AUC:** no saved table or script output identifies a protocol that produces it. It remains unreproduced and is excluded from the revised claims.
- **Score-field semantics:** before the March 2026 implementation migration, `player_rank` held a score; afterward it holds ordinal rank and `score` holds the log-odds value. The revised harness reads `score` explicitly and pins `loopr-ranking==0.0.1`.

## Revised temporal and uncertainty protocol

- Training data for event *t* contain only matches timestamped strictly before that event’s cutoff; event *t* itself is excluded.
- Ratings are recomputed before every held-out event and frozen only within that event.
- One positive calibration temperature is fitted only from earlier out-of-sample raw differences, after at least 200 matches and five tournaments.
- The report window is 2025-08-17 through 2026-01-28; the warmup begins 2025-07-15.
- All prediction systems use the same 4,397 matches in 110 tournaments. Unseen and low-history players remain in the test set under declared population priors.
- Model comparisons use paired differences. The primary resampling cluster is tournament (5,000 bootstrap repetitions), because matches within an event are dependent.
- Plus candidate intervals use 2,000 bootstraps stratified by tier, status, and outcome. Outcome shuffling uses 200 full recomputations with seed 20260731.

## Historical Plus-rule provenance

The November 16, 2025 analysis uses peer-vote results under the historical 60% rule. The [current Sendou FAQ](https://sendou.ink/faq) documents the vote and SendouQ admission paths; the version-pinned [August/November-era database migration](https://github.com/sendou-ink/sendou.ink/blob/c13ff32b406647fe66da7d971fd238951a832e54/migrations/041-plus-sixty.js) encodes the 60% decision as a mean ±1 vote score of at least 0.2. Automatic SendouQ admissions are absent from the exported `PlusVotingResult` rows. Later 2026 rule changes are not substituted into this vote.

## Leakage and dependence audit

- No random match split is used in the revised analysis.
- The held-out tournament is removed from every fitted rating, including the global LOOPR graph.
- Calibration never sees the event it predicts.
- The half-life sweep uses the same 53 warmup and 110 report events; the 180-day row exactly reproduces the primary LOOPR result.
- Plus AUC is secondary in the half-life sweep and is not used to select a winning parameter.
- Match dependence is handled by paired tournament-cluster resampling for prediction comparisons.

## Limitations that remain

1. Sparse appearance payloads make player-level lineup attribution imperfect.
2. Persistent teammate/lineup clusters weaken within-cluster identifiability even though the graph is connected.
3. Evaluation covers one observed organized-play network and ends on 2026-01-28.
4. Plus results cover one historical vote and 310 active candidates; vote percentage is unavailable.
5. The unfamiliar-lineup group is small (314 matches) and especially difficult.
6. Property tests for smoothing and tournament weighting are underpowered or endogenous and do not establish their intended benefits.
7. Measurement evidence cannot decide whether publishing named exact ranks is socially legitimate.

## Automated consistency checks

- Passed: manifest has 665 eligible tournaments.
- Passed: manifest has 28,647 matches.
- Passed: prediction methods use identical match counts.
- Passed: prediction methods use identical tournament counts.
- Passed: Plus methods use identical candidate counts.
- Passed: LOTO removed every eligible tournament.
- Passed: 180-day sweep reproduces primary LOOPR micro log loss.
- Passed: 180-day sweep reproduces primary LOOPR macro log loss.
