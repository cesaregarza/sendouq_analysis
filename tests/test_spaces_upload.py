import os
from pathlib import Path

from rankings.utils.spaces_upload import upload_outputs


def test_upload_outputs_dry_run_prints_without_error(
    tmp_path, monkeypatch, capsys
):
    out_run = tmp_path / "20240101_000000"
    out_run.mkdir(parents=True)
    # Create some files; leave others missing to exercise partial uploads
    for name in ["engine_state.json", "manifest.json", "players.parquet"]:
        (out_run / name).write_text("{}")

    monkeypatch.setenv("RANKINGS_UPLOAD_DRY_RUN", "true")
    # Use S3 URI with HTTPS form (Spaces-compatible)
    monkeypatch.setenv(
        "RANKINGS_S3_URI",
        "https://rankings.nyc3.digitaloceanspaces.com/runs",
    )
    # No credentials required for dry run
    upload_outputs(out_run)

    out = capsys.readouterr().out
    # Should print that it would upload existing files
    assert "Would upload" in out
    assert "engine_state.json" in out
    assert "manifest.json" in out
    # Should derive endpoint from HTTPS host
    assert "nyc3.digitaloceanspaces.com" in out
