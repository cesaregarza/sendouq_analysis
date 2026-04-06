from pathlib import Path


def test_run_weekly_loo_workflow_contains_schedule_and_command():
    workflow = Path(
        "/root/dev/sendouq_analysis/.github/workflows/run_weekly_loo.yml"
    ).read_text()

    assert "workflow_dispatch:" in workflow
    assert "cron: '0 2 * * 1'" in workflow
    assert "run rankings_weekly_loo" in workflow
    assert 'export PATH="/opt/poetry/bin:$PATH"' in workflow
    assert "-lc '" not in workflow
    assert "RANKINGS_DATABASE_URL" in workflow
    assert "RANKINGS_DB_SCHEMA" in workflow
    assert "SENTRY_DSN" in workflow
    assert "RANKINGS_SENTRY_DSN" in workflow
    assert ".github/scraper-image-version.txt" in workflow
    assert "registry.digitalocean.com/sendouq/scraper:v$VERSION" in workflow
    assert "registry.digitalocean.com/sendouq/scraper:latest" not in workflow
    assert workflow.count("command_timeout: 330m") == 2
