from pathlib import Path

from rankings.utils.spaces_upload import upload_outputs


def test_upload_outputs_dry_run_s3_uri_with_explicit_endpoint(
    tmp_path, monkeypatch, capsys
):
    out_run = tmp_path / "20240101_000000"
    out_run.mkdir(parents=True)
    # Create a couple of files
    for name in ["engine_state.json", "manifest.json"]:
        (out_run / name).write_text("{}")

    monkeypatch.setenv("RANKINGS_UPLOAD_DRY_RUN", "true")
    monkeypatch.setenv("RANKINGS_S3_URI", "s3://mybucket/some/prefix")
    monkeypatch.setenv(
        "RANKINGS_S3_ENDPOINT", "https://nyc3.digitaloceanspaces.com"
    )

    upload_outputs(out_run)

    out = capsys.readouterr().out
    assert "Would upload" in out
    # Bucket, prefix, and run subdir should appear in the destination path
    assert "mybucket/some/prefix/20240101_000000/engine_state.json" in out
    # Explicit endpoint should be reflected in output
    assert "endpoint=https://nyc3.digitaloceanspaces.com" in out
