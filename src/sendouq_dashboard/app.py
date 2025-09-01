import logging
import os

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback, dash_table, dcc, html

from rankings.core.sentry import init_sentry
from sendouq_dashboard.load.load_data import load_latest_player_stats


def _configure_logging() -> None:
    level = os.getenv("DASH_LOG_LEVEL", "INFO").upper()
    try:
        lvl = getattr(logging, level)
    except Exception:
        lvl = logging.INFO
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _init_sentry() -> None:
    """Initialize Sentry for the Dash/Flask app (best-effort)."""
    try:
        from sentry_sdk.integrations.flask import (
            FlaskIntegration,  # type: ignore
        )
    except Exception:
        FlaskIntegration = None  # type: ignore
    extra = [FlaskIntegration()] if FlaskIntegration is not None else None
    init_sentry(
        context="dash_app",
        dsn_envs=["DASH_SENTRY_DSN", "SENTRY_DSN"],
        extra_integrations=extra,
    )


_configure_logging()
_init_sentry()
df = load_latest_player_stats()

app = Dash(__name__)

theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
}

app.layout = html.Div(
    [
        html.H1(children="SendouQ Dashboard", style={"textAlign": "center"}),
        dash_table.DataTable(
            data=df.to_dict("records"),
            page_size=10,
            sort_action="native",
            sort_mode="multi",
        ),
        dcc.Graph(
            figure=px.histogram(
                df,
                x="sp",
                marginal="rug",
                hover_data=df.columns,
                title="Distribution of SP",
                labels={"sp": "SP"},
                template="plotly_dark",
                color_discrete_sequence=["#636EFA"],
            ).update_layout(
                xaxis=dict(title="SP"),
                yaxis=dict(title="Count"),
                bargap=0.2,
                bargroupgap=0.1,
                showlegend=False,
            )
        ),
    ]
)


def debug_run():
    app.run(host="127.0.0.1")


if __name__ == "__main__":
    app.run(host="127.0.0.1")
