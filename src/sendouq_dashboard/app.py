import logging
import os

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback, dash_table, dcc, html

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
    dsn = os.getenv("SENTRY_DSN") or os.getenv("DASH_SENTRY_DSN")
    if not dsn:
        return
    try:
        import sentry_sdk  # type: ignore
        from sentry_sdk.integrations.flask import (
            FlaskIntegration,  # type: ignore
        )
        from sentry_sdk.integrations.logging import (
            LoggingIntegration,  # type: ignore
        )
    except Exception:
        return

    env = os.getenv("SENTRY_ENV") or os.getenv("ENV") or "development"

    def _fenv(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    traces = _fenv("SENTRY_TRACES_SAMPLE_RATE", 0.0)
    profiles = _fenv("SENTRY_PROFILES_SAMPLE_RATE", 0.0)
    debug = os.getenv("SENTRY_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    logging_integration = LoggingIntegration(
        level=logging.INFO, event_level=logging.ERROR
    )
    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=env,
            integrations=[FlaskIntegration(), logging_integration],
            traces_sample_rate=traces,
            profiles_sample_rate=profiles,
            debug=debug,
        )
        sentry_sdk.set_tag("service", "dash_app")
    except Exception:
        pass


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
