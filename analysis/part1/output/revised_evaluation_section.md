### Validation Without a Ground Truth

Building a complicated ranking system is easy. Showing that it measures anything useful is the hard part.

There is no objective player-skill number hiding in the data for LOOPR to recover. Skill is latent, and team results are produced by more than mechanics: communication, leadership, adaptation, teammate selection, access to events, and opportunity all matter. So I stopped asking whether LOOPR had found *the* true ranking. I asked a narrower set of questions that the available evidence could actually answer.

Throughout this section, the construct is:

> **LOOPR estimates a player’s demonstrated ability to convert organized-play match opportunities into tournament victories, given the available network of teammates, opponents, events, and results.**

That is deliberately narrower than “true skill,” and broader than an individual-stat line. A player who makes teams win through coordination or leadership belongs inside the construct. Network access and teammate opportunity may also leak into it. Part II deals with what happens when that conditional, outcome-derived estimate is published as an exact social ordering.

#### What the Dataset Can See

The rating assumes that players mix enough for individual effects to emerge from team results. The graph passes the first test: all 5,518 players observed in valid matches belong to one weakly connected component, as do all of the final top 100. The typical player has six unique teammates, 48 unique opponents, and three distinct lineups.

But connection is not the same as individual identifiability. Fifteen percent of players with sufficient matches use one lineup at least 80% of the time, and 32% have one teammate beside them in at least 90% of matches. For those players, the data may support a claim about the pair or lineup more strongly than a claim about the exact ordering within it.

[Insert “What the Dataset Can See” table.]

#### Can It Predict What Happens Next?

The cleanest test is temporal. Before each of 110 held-out tournaments, I rebuilt every rating using only earlier eligible matches. I calibrated probabilities using earlier held-out events only, predicted the next tournament, then advanced the cutoff. Ratings were frozen within an event and recomputed before the next one.

The primary metric was **log loss**. In plain language: did the model assign high probability to the team that actually won, without becoming recklessly confident? Lower is better.

LOOPR’s match-weighted log loss was **0.513** across 4,397 matches. That was substantially better than a 90-day win-rate model (**0.578**) and a 50/50 prediction (**0.693**). The paired LOOPR advantage over recent win rate was 0.065 log-loss points, with a tournament-cluster 95% interval from 0.047 to 0.082. Tournament-weighted results told the same basic story, so one giant event did not create the finding.

That supports a real but bounded claim: LOOPR contains substantial information about future tournament outcomes beyond recent win rate alone.

It was not the best predictor in the comparison. OpenSkill ordinal reached **0.486**, OpenSkill μ **0.490**, and a regularized team Bradley–Terry model **0.489**. Their paired intervals versus LOOPR all excluded zero in their favor. These systems are baselines, not ground truth, but the result rules out saying that LOOPR extracts the available outcome signal better than the alternatives.

[Insert future-prediction table and calibration figure.]

The calibration curve was directionally sensible: higher predicted probabilities corresponded to higher observed win rates. It was not perfect. LOOPR was somewhat underconfident in the lower and middle bins and slightly overconfident in the highest bin.

There was also an important transfer limit. LOOPR log loss was 0.452 on familiar lineups, 0.530 on mixed lineups, and 0.688—nearly chance—on the 314 mostly novel-lineup matches. The model is strongest inside the teammate/opponent network it has already seen. That is evidence for conditional organized-play measurement, not a context-free player essence.

#### Does It Converge With Plus Voting?

Plus voting is not ground truth. It is a different, fallible, socially consequential measurement process. That makes agreement useful as **convergent evidence**.

I used the November 16, 2025 peer-vote snapshot and the historical 60% pass rule. Automatic SendouQ admissions were not present in these vote-result rows. Of 468 candidates, 310 met the predeclared 90-day roster-activity rule; every method was evaluated on the same 310.

First I asked the institutional question: within each tier, did the score tend to put passers above failers? LOOPR’s macro AUC was **0.777** (95% CI 0.719–0.831). This is the equal-weight mean of the three tier-specific passer-above-failer probabilities: 77.7% here. It does not mean the ranking is 77.7% accurate.

Then I made the comparison stricter. Incumbents and newly suggested candidates face different visibility, activity, and pass-rate conditions, so I compared passers and failers only within the same tier *and* candidate status, then combined the six cells by their available passer–failer pairs. That like-for-like AUC fell to **0.615** (95% CI 0.523–0.707). The evidence still points in the intended direction, but much less dramatically.

Activity alone did not reproduce the institutional result: recent-match AUC was 0.460, and recent-tournament AUC was 0.445. Under the like-for-like comparison, however, their AUCs were 0.577 and 0.581. LOOPR’s estimated advantage was small relative to uncertainty. So activity does not explain the headline institutional agreement, but this sample does not clearly show that LOOPR beats activity within every comparable decision context.

#### Does the Outcome Direction Matter, or Just Who Participated?

To separate results from access and visibility, I preserved every player, lineup, tournament, timestamp, and match count, replaced each winner with a fair coin flip, and recomputed the full LOOPR pipeline 200 times—including tournament-strength iteration.

The shuffled institutional AUC averaged **0.501**, with a 95% range of 0.403–0.591. None of the 200 draws reached the observed 0.777 (corrected tail probability 0.005). The shuffled like-for-like AUC averaged 0.506; the observed 0.615 also exceeded that null (corrected tail probability 0.010).

This is strong evidence that win/loss direction adds information beyond participation structure. It does not erase the role of access or visibility; it says showing up alone did not produce the observed agreement.

[Insert Plus-and-shuffle figure.]

#### Is It Stable Without Being Frozen?

Stability is an operational requirement, not proof of accuracy. An incorrect ranking can be perfectly stable.

Across 13 adjacent monthly transitions, average top-100 retention was **83.8%**. Among players retained in both lists, average Spearman ordering correlation was **0.965**, and the median transition’s typical retained player moved four positions. But retention ranged from 60% to 98%. Between October 31 and November 30, only 60 names remained; among those retained, the median player moved 11 places.

The old Spearman-only summary hid that churn. The accurate conclusion is: ordering among retained players is usually stable, while membership can change substantially when activity and new results alter who belongs in the top 100.

[Insert monthly-stability figure.]

#### Does One Tournament Control the Ranking?

For each of 665 eligible tournaments, I removed the event and recomputed the entire pipeline. Median top-100 retention was **100%**; the worst case was **96%** after removing Crobattle Royale: Mel0's Birthday Edition (28th). The median event produced a median shared-player movement of zero places, and 90% of event removals had a 90th-percentile shared-player movement of no more than one place.

There is a tail. Removing Baja's Blast #31 moved one still-eligible top-100 player 63 places. That player had 152 historical matches, so the largest shared shift was not limited to a low-match newcomer. Eighteen removals also changed current rank eligibility for at least one original top-100 player; those cases are counted separately rather than pretending the player fell to a numeric bottom rank.

The ranking is broadly supported by the network at this cutoff, but no individual should be told that one event can never matter sharply.

[Insert leave-one-tournament-out figure.]

#### Did the Components Behave as Intended?

The targeted tests were mixed.

- Among 2,793 real-player comparisons matched approximately on win rate and volume, the player with the stronger measured opponent schedule had the higher LOOPR score 72.8% of the time.
- After controlling for win rate and opponent strength with simple flexible terms, the residual standardized coefficient on log match volume was −0.009, with partial R² 0.00025. That supports volume resistance, not perfect cancellation.
- In 24 one-match perturbations, smoothing and no-smoothing movement were nearly identical. This test did not demonstrate the intended low-evidence protection.
- The observed tournament-weighting test did not cleanly show that weighting suppresses leverage from weak events. Complete Goodhart resistance remains unestablished.

The half-life sweep was also a tradeoff, not an optimization victory. Ninety days predicted slightly better than 180 days (0.511 versus 0.513 match-level log loss), while monthly retention differed by about one percentage point. Several settings were similar. Choosing 180 days can be defended as a continuity preference, but not as the uniquely correct value.

### What the Evidence Supports

Several distinct tests pointed in the same direction. LOOPR predicted later tournament outcomes, beat recent win rate and chance, produced coherent probability ordering, converged with a consequential community judgment, exceeded a participation-only shuffled null, and was usually robust to removing one event. Its schedule and volume behavior also moved in the intended direction.

The same audit set boundaries. LOOPR did not beat the stronger predictive baselines. Like-for-like Plus agreement was moderate. Transfer to mostly novel lineups was weak. Monthly membership could churn. Smoothing and tournament-strength weighting did not earn strong empirical claims from these tests.

Taken together, that supports treating LOOPR as a **meaningful outcome-derived skill-o-meter** for demonstrated organized-play success. Calling it “strong” is defensible if that means substantial information supported by several different tests—not best-in-class prediction, objective skill recovery, or validation of every exact pairwise rank.

I assumed that building a meaningful measurement instrument meant I had built a meaningful public leaderboard.

The Splatoon community rejected it anyway.

That is where Part II begins.
