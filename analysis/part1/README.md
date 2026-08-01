# Building Public Rankings, Part I audit

This package contains the strict temporal evaluation, robustness analyses, and
reader-facing deliverables for Part I.

- Start with [`output/technical_report.md`](output/technical_report.md).
- The exact protocol is in [`output/evaluation_manifest.json`](output/evaluation_manifest.json).
- Rebuild instructions are in [`output/README.md`](output/README.md).
- The proposed essay replacement is in [`output/revised_evaluation_section.md`](output/revised_evaluation_section.md).
- The claim-by-claim audit is in [`output/claim_ledger.md`](output/claim_ledger.md).
- Essay-grade figure changes and documented deviations are in
  [`output/figures_CHANGELOG.md`](output/figures_CHANGELOG.md).
- Dark and light publication exports are under `output/figures/dark/` and
  `output/figures/light/`; machine-readable render checks are in
  [`output/results/figure_validation.json`](output/results/figure_validation.json).

The source entry points are `run.py`, `run_half_life_one.py`, `publication.py`,
`publication_figures.py`, and `report.py`. Run the `publication` stage after the
core audit stages, then run `report.py`; exact commands are in `output/README.md`.
Analysis-only package pins are in `requirements-analysis.txt`.
