import pytest
from sqlalchemy import create_engine

from rankings.sql import views as rankings_views
from sendouq_analysis.sql import views as analysis_views


@pytest.mark.parametrize(
    "schema", ["", "public; DROP TABLE x; --", "a.b", "a-b"]
)
def test_rankings_views_rejects_invalid_schema(schema: str):
    engine = create_engine("sqlite://")
    with pytest.raises(ValueError):
        rankings_views.ensure_tournament_event_times_view(engine, schema=schema)


@pytest.mark.parametrize(
    "schema", ["", "public; DROP TABLE x; --", "a.b", "a-b"]
)
def test_analysis_views_rejects_invalid_schema(schema: str):
    engine = create_engine("sqlite://")
    with pytest.raises(ValueError):
        analysis_views.ensure_tournament_event_times_view(engine, schema=schema)
