"""Small reusable Dash UI building blocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import html


def kpi(label: str, value: str, sub: str | None = None, tone: str = "indigo"):
    return html.Div(
        className=f"kpi tone-{tone}",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className="kpi-sub") if sub else None,
        ],
    )


def empty_state(title: str, text: str = ""):
    return html.Div(
        className="empty-state",
        children=[html.Div("◌", className="empty-icon"), html.Div(title, className="empty-title"),
                  html.Div(text, className="empty-text")],
    )


def alert(kind: str, text: str):
    return html.Div(text, className=f"alert alert-{kind}")


def insight_card(level: str, title: str, text: str):
    tag = {"warn": "Attention", "info": "Note", "good": "Good"}[level]
    return html.Div(
        className=f"insight lvl-{level}",
        children=[
            html.Div(tag, className="insight-tag"),
            html.Div([html.Div(title, className="insight-title"), html.Div(text, className="insight-text")],
                     className="insight-body"),
        ],
    )


def stat_chip(label: str, value: str):
    return html.Div([html.Span(label, className="chip-label"), html.Span(value, className="chip-value")],
                    className="stat-chip")


# ----------------------------------------------------------------------- #
# Tables
# ----------------------------------------------------------------------- #
def _is_missing(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _format(v, thousands: bool):
    if _is_missing(v):
        return html.Span("—", className="na")
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}" if thousands else str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v.is_integer() and abs(v) < 1e15:
            return f"{int(v):,}" if thousands else str(int(v))
        text = f"{v:,.4f}".rstrip("0").rstrip(".") if thousands else f"{v:.4f}".rstrip("0").rstrip(".")
        return text
    if isinstance(v, pd.Timestamp):
        return f"{v:%Y-%m-%d}" if (v.hour, v.minute, v.second) == (0, 0, 0) else f"{v:%Y-%m-%d %H:%M}"
    text = str(v)
    return text if len(text) <= 60 else text[:57] + "…"


def type_badge(v):
    cls = {"Numeric": "b-num", "Categorical": "b-cat", "Datetime": "b-dt", "Boolean": "b-bool"}.get(v, "b-cat")
    return html.Span(v, className=f"badge {cls}")


def pct_bar(v):
    if _is_missing(v):
        return html.Span("—", className="na")
    width = max(0.0, min(100.0, float(v)))
    tone = "ok" if width < 5 else "mid" if width < 20 else "bad"
    return html.Div(
        className="pct-cell",
        children=[html.Div(html.Div(className=f"pct-fill {tone}", style={"width": f"{width}%"}), className="pct-track"),
                  html.Span(f"{v:.1f}%", className="pct-text")],
    )


def html_table(df: pd.DataFrame, max_rows: int | None = None, thousands: bool = False,
               renderers: dict | None = None, max_height: int = 420):
    renderers = renderers or {}
    data = df if max_rows is None else df.head(max_rows)
    numeric = {c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and c != "Row #"}
    head = html.Thead(html.Tr([html.Th(str(c), className="num" if c in numeric else None) for c in data.columns]))
    body_rows = []
    for row in data.itertuples(index=False, name=None):
        cells = []
        for col, v in zip(data.columns, row):
            content = renderers[col](v) if col in renderers else _format(v, thousands)
            cells.append(html.Td(content, className="num" if col in numeric else None))
        body_rows.append(html.Tr(cells))
    return html.Div(
        className="table-wrap",
        style={"maxHeight": f"{max_height}px"},
        children=html.Table([head, html.Tbody(body_rows)], className="data-table"),
    )
