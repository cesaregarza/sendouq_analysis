# Proposed Part I evaluation reframing

## What I Was Actually Trying to Build

There is a version of this story where I evaluate LOOPR like a prediction model, discover that OpenSkill and Bradley–Terry assign slightly better probabilities to future matches, and stop there.

That result is real. It is also not the whole design problem.

LOOPR was meant to become a public player ranking. A good ranking still needs to generalize to future competition, but it also needs to resist grinding weak wins, account for schedule, remain stable enough to use, and avoid letting one event rewrite the visible top. Those properties do not collapse into average log loss.

So I kept the prediction test and separated two questions:

- **Generalization:** does the score contain information about any future match?
- **Intended use:** does it produce a defensible ordering among established competitors, and does that ordering fail gracefully when people or data put pressure on it?

There is no objective true-skill vector behind either question. The construct I can defend is narrower:

> LOOPR estimates a player's demonstrated ability to convert organized-play match opportunities into tournament victories, given the available network of teammates, opponents, events, and results.

## Where the Graph Can Support a Player Ranking

I defined graph support without looking at any rating or winner direction. A player's k-core number asks whether they sit inside a recursively dense opponent network. For a held-out match to enter a support stratum, every participant had to meet the threshold.

The distinction mattered. LOOPR was poor on core-0 matches: log loss 0.681, versus 0.480 for OpenSkill ordinal. In the core-32+ graph, which covered 2,827 held-out matches, LOOPR's log loss was 0.486, OpenSkill μ's was 0.481, and Bradley–Terry's was 0.482. Tournament-cluster uncertainty did not clearly separate those systems there. LOOPR's raw winner ordering was slightly highest at 76.5%.

That does not make the sparse failure disappear. It defines the scope of the instrument. LOOPR works best where there are enough transitive paths for its design to have something to work with.

## An Existing Artifact Closer to Ranking Use

Tournament pages preserve team seeds. I did not assume those were expert labels. I audited them first.

The stored order is not registration order, is strongly correlated with Sendou's automatic seeding ordinal without usually copying it exactly, changes across repeated snapshots in only three tournaments, and predicts match direction far above chance. Information remains below the top four seeds. The repository cannot tell me whether an organizer manually edited a list, copied part of an earlier event, or used a private process.

In the 23 events that passed every roster and format restriction after the prediction start date, all four schedule-aware systems matched seeds strongly. OpenSkill ordinal had the highest concordance point estimate. When LOOPR and OpenSkill or Bradley–Terry disagreed, the seed data did not clearly favor either side. LOOPR clearly beat 90-day win rate.

That makes seeds useful convergent evidence for schedule-aware ranking. It does not make them proof that LOOPR is best.

## What the Extra Computation Bought

The clearest differences appeared when I stressed the rankings directly.

Adding fifty wins against weak lineups moved LOOPR by a median 0.040 within-system standard deviations. The same attack moved OpenSkill μ by 0.090, OpenSkill ordinal by 0.069, Bradley–Terry by 0.092, and 90-day win rate by 1.533. LOOPR still rewarded strong wins and gave a 0.805-standard-deviation advantage to the same 5–5 record against strong rather than weak opposition.

Duplicating each selected lineup's block of up to ten recent exact-lineup matches fivefold moved LOOPR least in the tail. Removing half the selected players' historical matches over ten deterministic perturbations also moved LOOPR least: 95% median top-100 retention and 0.238 standard deviations at the median replicate's 90th percentile.

Monthly stability was a tradeoff. Bradley–Terry kept more of the same top-100 names. LOOPR best preserved the order among names that remained. When I removed each of the 25 events with the most active dense-core participation and rebuilt every system, LOOPR had 99% worst-case top-100 retention and a maximum observed shift of 14 ranks—smaller than the other schedule-aware systems.

These are leaderboard properties. They do not prove accuracy. A stable ranking can be stably wrong. But they answer why a more complicated ranking might be worthwhile even when its probabilities are not best.

## Re-reading Plus

I kept the pooled Plus AUC because it describes agreement with the institution that actually operated. Its literal value was 0.651; the equal-tier institutional AUC was 0.777. The within-tier/status AUC was 0.615, but that is not a cleaner number. It is a narrower question that discards cross-tier and cross-status decisions.

Candidate status explains a large part of the institution. Tier plus status alone reached 0.749 out of fold. A model with tier, status, activity, and graph support reached 0.790; adding LOOPR raised that to 0.812 and improved log loss by 0.014. All evaluated candidates were already in the 32-core, so this vote says nothing about sparse newcomers.

I also reran the full outcome-shuffle control for literal pooled AUC. Preserving participation and event structure while destroying winner direction produced a mean AUC of 0.503; none of 200 draws reached 0.651. Match outcomes add information beyond showing up. Status and opportunity still matter.

## What I Can Claim

Several different tests now answer different parts of the design problem. LOOPR predicted future competition, became competitive with the strongest probability baselines in the dense graph, matched consequential historical orderings, strongly rewarded schedule quality, and degraded unusually gracefully under farming, volume, missing-history, and event-removal stress.

It also failed important tests. It was poor at the sparse edge. It did not beat OpenSkill or Bradley–Terry at general-purpose probabilities. It did not uniquely match organizer seeds. Bradley–Terry retained top-100 membership better month to month. The transitive-chain test did not distinguish LOOPR from simpler systems.

The defensible conclusion is not that LOOPR recovered true skill or won every metric. It is that LOOPR is a strong outcome-derived skill-o-meter—and a credible ranking-oriented design—for established organized-play competitors inside a well-supported network.

I assumed that meant I had built a strong public leaderboard.

The Splatoon community rejected it anyway.

Part II is about why measurement quality was never the only question.
