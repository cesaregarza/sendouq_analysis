# Part I evaluation package

This directory is the reproducible audit for *Building Public Rankings*, Part I.

## Environment

From the repository root, install the project and analysis-only packages into an isolated environment. The versions used for the frozen results are recorded in `evaluation_manifest.json`; the extra requirements are in `../requirements-analysis.txt`.

```bash
python -m pip install -e .
python -m pip install -r analysis/part1/requirements-analysis.txt
```

## Rebuild

Every stage reads raw data from `--source-root`, sets seed 20260731, and saves intermediates under this output directory.

```bash
python -m analysis.part1.run audit --source-root /path/to/sendouq_analysis
python -m analysis.part1.run connectivity --source-root /path/to/sendouq_analysis
python -m analysis.part1.run prediction --source-root /path/to/sendouq_analysis
python -m analysis.part1.run plus --source-root /path/to/sendouq_analysis
python -m analysis.part1.run shuffle --source-root /path/to/sendouq_analysis --shuffle-repetitions 200
python -m analysis.part1.run stability --source-root /path/to/sendouq_analysis
python -m analysis.part1.run loto --source-root /path/to/sendouq_analysis
python -m analysis.part1.run properties --source-root /path/to/sendouq_analysis
```

The half-life rows can be run independently (and safely in parallel):

```bash
for setting in 90 120 180 270 365 no_decay; do
  python -m analysis.part1.run_half_life_one \
    --source-root /path/to/sendouq_analysis \
    --output-dir analysis/part1/output \
    --setting "$setting"
done
```

Freeze tables, figures, reports, notebook, hashes, and the aggregate results file:

```bash
python -m analysis.part1.report --output-dir analysis/part1/output
python -m analysis.part1.run collect \
  --source-root /path/to/sendouq_analysis \
  --output-dir analysis/part1/output
```

The `collect` stage does not read raw data; `--source-root` remains required by the common CLI parser.

## Stage semantics

- `audit`: inventory, duplicate/timestamp audit, reproduction ledger, evaluation manifest.
- `connectivity`: graph component, teammate/opponent, lineup concentration diagnostics.
- `prediction`: strict event-by-event walk-forward ratings, prior-only calibration, paired cluster bootstrap, reliability, unfamiliar-lineup analysis.
- `plus`: institutional and tier/status-stratified AUCs with activity baselines.
- `shuffle`: full LOOPR recomputation after outcome-direction randomization.
- `stability`: adjacent monthly top-100 membership, ordering, and movement.
- `loto`: full leave-one-tournament-out sensitivity analysis.
- `properties`: schedule, residual volume, smoothing perturbation, and event-weighting checks.
- `run_half_life_one`: one predeclared decay setting under the unified event sequence.
- `report`: consistency assertions and all reader-facing deliverables.

## Primary artifacts

- `technical_report.md`
- `data_methodology_audit.md`
- `evaluation_manifest.json`
- `claim_ledger.csv` and `claim_ledger.md`
- `revised_evaluation_section.md`
- `part1_results.ipynb`
- `results/results.json`
- `artifact_index.csv`
- `audit/project_file_inventory.csv`

Large checkpoint files live in `cache/`; auditable analysis intermediates live in `intermediate/` and `tables/`.
