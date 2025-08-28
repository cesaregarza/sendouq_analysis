from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def upload_outputs(out_run: Path, s3_prefix: Optional[str] = None) -> None:
    """Upload outputs in out_run to S3/Spaces destination defined by env.

    Supported envs:
      - RANKINGS_S3_URI: s3://bucket/prefix OR https://<bucket>.<endpoint>/<prefix>
      - RANKINGS_S3_BUCKET and RANKINGS_S3_PREFIX
      - Optional: RANKINGS_S3_ENDPOINT (e.g., https://nyc3.digitaloceanspaces.com)
      - Optional: RANKINGS_UPLOAD_DRY_RUN=true to print actions without uploading

    For DigitalOcean Spaces, for example:
      RANKINGS_S3_URI=https://rankings.nyc3.digitaloceanspaces.com
      RANKINGS_S3_ENDPOINT=https://nyc3.digitaloceanspaces.com
    Credentials should be provided via AWS_* env vars.
    """
    dry_run = os.getenv("RANKINGS_UPLOAD_DRY_RUN", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    s3_uri = os.getenv("RANKINGS_S3_URI")
    endpoint = os.getenv("RANKINGS_S3_ENDPOINT")
    bucket: Optional[str] = None
    prefix_env: Optional[str] = None

    if s3_uri:
        from urllib.parse import urlparse

        u = urlparse(s3_uri)
        if u.scheme in {"s3"}:
            if not u.netloc:
                raise ValueError(f"Invalid RANKINGS_S3_URI: {s3_uri}")
            bucket = u.netloc
            prefix_env = u.path.lstrip("/")
        elif u.scheme in {"http", "https"}:
            # Expecting https://<bucket>.<endpoint>/<optional-prefix>
            host_parts = (u.netloc or "").split(".")
            if (
                len(host_parts) < 4
            ):  # e.g., rankings.nyc3.digitaloceanspaces.com
                raise ValueError(
                    f"Invalid Spaces URL in RANKINGS_S3_URI: {s3_uri}"
                )
            bucket = host_parts[0]
            # Derive endpoint if not provided explicitly
            if not endpoint:
                endpoint = f"{u.scheme}://{'.'.join(host_parts[1:])}"
            prefix_env = u.path.lstrip("/")
        else:
            raise ValueError(
                f"Unsupported RANKINGS_S3_URI scheme '{u.scheme}'. Use s3:// or https://"
            )
    else:
        bucket = os.getenv("RANKINGS_S3_BUCKET")
        prefix_env = os.getenv("RANKINGS_S3_PREFIX", "") or ""

    if not bucket:
        print(
            "upload_s3 enabled but no destination provided; set RANKINGS_S3_URI or RANKINGS_S3_BUCKET/PREFIX"
        )
        return

    # Default key prefix: 'rankings/<run_ts>/' if none supplied
    run_ts = out_run.name
    if s3_prefix:
        base_dir = s3_prefix.strip("/")
    else:
        base_dir = (prefix_env or "rankings").strip("/")
    base_prefix = f"{base_dir}/{run_ts}" if base_dir else run_ts

    files = [
        out_run / "engine_state.json",
        out_run / "manifest.json",
        out_run / "matches.parquet",
        out_run / "players.parquet",
        out_run / "rankings.parquet",
    ]

    if dry_run:
        for fp in files:
            if fp.exists():
                key = f"{base_prefix}/{fp.name}" if base_prefix else fp.name
                dest = f"{bucket}/{key}"
                print(
                    f"[DRY RUN] Would upload {fp} to {dest} (endpoint={endpoint or 'aws default'})"
                )
        return

    try:
        import boto3  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"boto3 not available; skipping upload: {e}")
        return

    # Create client; allow custom endpoint for Spaces
    if endpoint:
        s3 = boto3.client("s3", endpoint_url=endpoint)
    else:
        s3 = boto3.client("s3")

    for fp in files:
        if fp.exists():
            key = f"{base_prefix}/{fp.name}" if base_prefix else fp.name
            try:
                s3.upload_file(str(fp), bucket, key)
                print(f"Uploaded {fp.name} to {bucket} at {key}")
            except Exception as e:
                print(f"Failed to upload {fp} to {bucket}/{key}: {e}")
