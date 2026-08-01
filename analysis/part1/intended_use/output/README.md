
# Intended-use evaluation addendum

This directory is additive to the frozen Part I audit. Start with:

- `technical_report.md` — answer-first analysis and recommendations.
- `proposed_reframed_evaluation_section.md` — replacement narrative for Part I.
- `intended_use_results.ipynb` — executed, reviewable companion.
- `tables/requirements_profile.csv` — no-composite all-system comparison.
- `PREANALYSIS_PLAN.md` and `IMPLEMENTATION_DECISIONS.md` one directory up — definitions frozen before results.

Rebuild result tables with `python -m analysis.part1.intended_use.run ...` and render this package with `python -m analysis.part1.intended_use.report --output analysis/part1/intended_use/output`.
