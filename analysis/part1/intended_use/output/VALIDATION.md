
# Validation record

- PASS: all six intended-use invariance and artifact-integrity tests.
- PASS: Ruff checks for the complete `analysis/part1/intended_use` package.
- PASS: executed notebook; all six code cells completed with no error outputs.
- PASS: report consistency checks for five-system coverage, seed-event count, attack-pool scope, thinning replicates, Plus graph support, and 200 pooled shuffle recomputations.
- ENVIRONMENT LIMIT: the broader repository suite reaches completion but has two failures in `tests/test_loss_improvements.py`. Both arise in untouched geometric team aggregation because the available runtime has Polars 1.39.0 while `pyproject.toml` declares Polars `^0.20.0` (`map_elements` now supplies scalars where that legacy code expects a Series). The new intended-use tests do not fail.

This environment mismatch was not repaired because it is outside the additive evaluation scope and changing production aggregation code would risk altering the frozen audit.
