from pathlib import Path


ROOT = Path("/root/dev/sendouq_analysis")
VERSION_FILE = ROOT / ".github" / "scraper-image-version.txt"
WORKFLOWS = [
    "build_scraper.yml",
    "fix_tournaments.yml",
    "repull_tournament.yml",
    "run_aggregator.yml",
    "run_ranked.yml",
    "run_scraper.yml",
    "run_weekly_loo.yml",
]


def test_scraper_image_version_file_is_numeric():
    version = VERSION_FILE.read_text().strip()
    assert version.isdigit()
    assert int(version) > 0


def test_scraper_workflows_use_versioned_image_tags():
    for name in WORKFLOWS:
        workflow = (ROOT / ".github" / "workflows" / name).read_text()
        assert ".github/scraper-image-version.txt" in workflow
        assert "registry.digitalocean.com/sendouq/scraper:latest" not in workflow


def test_build_scraper_workflow_builds_on_main_push_and_pushes_versioned_tag():
    workflow = (
        ROOT / ".github" / "workflows" / "build_scraper.yml"
    ).read_text()

    assert "push:" in workflow
    assert "- main" in workflow
    assert "docker build -t \"scraper:v${{ steps.scraper_image.outputs.version }}\"" in workflow
    assert "docker push \"${{ steps.scraper_image.outputs.image }}\"" in workflow
