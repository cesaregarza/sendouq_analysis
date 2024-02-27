import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback, dash_table, dcc, html

from sendouq_dashboard.load.load_data import load_latest_player_stats

df = load_latest_player_stats()

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(children="SendouQ Dashboard", style={"textAlign": "center"}),
        dash_table.DataTable(data=df.to_dict("records"), page_size=50),
        dcc.Graph(
            figure=px.histogram(
                df, x="sp", marginal="rug", hover_data=df.columns
            )
        ),
    ]
)


@callback(
    Output("graph-content", "figure"), Input("dropdown-selection", "value")
)
def update_graph(value):
    dff = df[df.country == value]
    return px.line(dff, x="year", y="pop")


def debug_run():
    app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True)
