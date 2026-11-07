"""Interactive EDA Dashboard - entry point.

Run:  python app.py        then open http://127.0.0.1:8050

Workflow (matches the project paper):
Dataset Upload -> Profiling -> Quality Inspection -> Chart Selection ->
Visualization -> Interactive Interpretation.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from eda import charts, loader
from eda import profiling as prof
from eda.components import (alert, empty_state, html_table, insight_card, kpi, pct_bar, stat_chip,
                            type_badge)

APP_TITLE = "Interactive EDA Dashboard"

app = Dash(
    __name__,
    title=APP_TITLE,
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # for gunicorn / deployment

PAGES = [
    ("overview", "Overview", "Dataset structure, column types and a live preview"),
    ("quality", "Data Quality", "Missing values, duplicate rows and text consistency"),
    ("visualize", "Visualize", "Build bar, histogram, box, scatter and heatmap views"),
    ("stats", "Statistics", "Descriptive statistics for numeric and categorical columns"),
    ("insights", "Insights", "Auto-generated findings to guide deeper analysis"),
]
PAGE_META = {k: (t, s) for k, t, s in PAGES}

CHART_TYPES = [
    {"label": "Bar chart", "value": "bar"},
    {"label": "Histogram", "value": "hist"},
    {"label": "Box plot", "value": "box"},
    {"label": "Scatter plot", "value": "scatter"},
    {"label": "Correlation heatmap", "value": "heatmap"},
]
CHART_TITLES = {"bar": "Category comparison", "hist": "Distribution", "box": "Spread & outliers",
                "scatter": "Relationship", "heatmap": "Correlation overview"}
GUIDES = {
    "bar": "Compares categories by row count or an aggregated value. Taller bars mean larger totals; use "
           "“Split by” to see how a second category contributes. Date columns are grouped by month.",
    "hist": "Shows how the values of one numeric column are distributed. Look for symmetry, skew, gaps and "
            "several peaks. The dashed lines mark the mean and median - a wide gap between them hints at skew or outliers.",
    "box": "The box spans the middle 50% of values (Q1 to Q3) with the median line; the dashed line is the mean. "
           "Whiskers reach 1.5×IQR and points beyond are flagged as possible outliers - these can be valid events, "
           "measurement issues or a smaller subgroup, so check context before removing them.",
    "scatter": "Each point is one row. Look for trends, clusters and unusual combinations. A strong correlation "
               "describes association only and does not by itself establish causation.",
    "heatmap": "Colour encodes pairwise correlation from −1 (blue: move in opposite directions) to +1 "
               "(red: move together). Pearson captures linear association; Spearman captures monotonic "
               "relationships and is less sensitive to outliers.",
}

# Which control wrappers are visible for each chart type, and their labels.
VISIBLE = {
    "bar": {"x", "y", "color", "agg", "topn"},
    "hist": {"x", "color", "bins"},
    "box": {"x", "y", "points"},
    "scatter": {"x", "y", "color", "trend"},
    "heatmap": {"heat", "method", "annot"},
}
LABELS = {
    "bar": ("Category (X axis)", "Value column (optional)", "Split by (optional)"),
    "hist": ("Numeric column", "", "Group by (optional)"),
    "box": ("Group by (optional)", "Numeric column", ""),
    "scatter": ("X axis (numeric)", "Y axis (numeric)", "Colour by (optional)"),
    "heatmap": ("", "", ""),
}
WRAPPERS = ["x", "y", "color", "bins", "agg", "topn", "points", "trend", "heat", "method", "annot"]


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
def card(title, children, subtitle=None, actions=None, class_name=""):
    head = html.Div(
        className="card-head",
        children=[html.Div([html.Div(title, className="card-title"),
                            html.Div(subtitle, className="card-sub") if subtitle else None]),
                  actions],
    )
    return html.Section(className=f"card {class_name}".strip(), children=[head, html.Div(children, className="card-body")])


def control(wrapper_id, label_id, label, component):
    return html.Div(id=f"wrap-{wrapper_id}", className="ctl-group",
                    children=[html.Label(label, id=label_id, className="ctl-label") if label_id else None, component])


def sidebar():
    return html.Aside(
        className="sidebar",
        children=[
            html.Div(className="brand", children=[
                html.Div(className="brand-mark", children=[html.Span(), html.Span(), html.Span()]),
                html.Div([html.Div("EDA Dashboard", className="brand-name"),
                          html.Div("Interactive · Automated", className="brand-tag")]),
            ]),
            html.Div("WORKFLOW", className="side-heading"),
            html.Nav(className="nav", children=[
                html.Button([html.Span(f"0{i + 1}", className="nav-num"), html.Span(title, className="nav-text")],
                            id=f"nav-{key}", n_clicks=0, className="nav-btn")
                for i, (key, title, _) in enumerate(PAGES)
            ]),
            html.Div(className="side-note", children=[
                html.Div("Evidence, not verdicts", className="side-note-title"),
                html.Div("The dashboard reports what is in the data. Interpretation stays with the analyst.",
                         className="side-note-text"),
            ]),
            html.Div(className="side-foot", children=[
                html.Div("Data Science · 24AD3107", className="foot-course"),
                html.Div("Y. Revanth · G. Rithwik · A. Nikitha", className="foot-team"),
                html.Div("Guide: Dr. N. Sunanda", className="foot-team"),
                html.Div("KL University, Hyderabad", className="foot-uni"),
            ]),
        ],
    )


def data_panel():
    return html.Section(className="data-panel", children=[
        dcc.Upload(
            id="upload", multiple=False, max_size=loader.MAX_UPLOAD_BYTES,
            className="upload-zone", className_active="upload-zone active", className_reject="upload-zone reject",
            children=html.Div(className="upload-inner", children=[
                html.Div("↑", className="upload-icon"),
                html.Div([html.Div(["Drag & drop a dataset, or ", html.Span("browse", className="link")],
                                   className="upload-title"),
                          html.Div("CSV · TSV · Excel (.xlsx) · JSON  —  up to 25 MB", className="upload-sub")]),
            ]),
        ),
        html.Div(className="sample-box", children=[
            html.Label("…or try a sample dataset", className="ctl-label"),
            dcc.Dropdown(id="sample-select", clearable=False, searchable=False,
                         options=[{"label": v[0], "value": k} for k, v in loader.SAMPLES.items()],
                         value=next(iter(loader.SAMPLES))),
        ]),
        html.Div(id="load-msg", className="load-msg"),
    ])


def page_overview():
    return [
        html.Div(id="ov-kpis", className="kpi-grid"),
        html.Div(className="grid-2", children=[
            card("Column types", dcc.Graph(id="ov-donut", config={"displayModeBar": False}),
                 "How the columns are distributed by data type"),
            card("Column profile", html.Div(id="ov-columns"), "Type, completeness and range of every column"),
        ]),
        card("Data preview", html.Div(id="ov-preview"), "First rows of the dataset, exactly as loaded",
             actions=html.Div(className="inline-ctl", children=[
                 html.Span("Rows", className="ctl-label"),
                 dcc.Dropdown(id="preview-rows", options=[10, 25, 50, 100], value=10, clearable=False,
                              searchable=False, style={"width": "92px"})])),
    ]


def page_quality():
    return [
        html.Div(id="q-kpis", className="kpi-grid"),
        html.Div(className="grid-2 even", children=[
            card("Missing values by column", dcc.Graph(id="q-missing-bar", config={"displayModeBar": False}),
                 "Share of rows with an empty value"),
            card("Missing-value pattern", dcc.Graph(id="q-missing-matrix", config={"displayModeBar": False}),
                 "Each dark cell is a missing value (rows evenly sampled)"),
        ]),
        card("Duplicate rows", html.Div(id="q-dups"), "Flagged for review - never deleted automatically"),
        card("Text consistency", html.Div(id="q-text"), "Whitespace and spelling/case variants in text columns"),
    ]


def page_visualize():
    controls = html.Div(className="controls", children=[
        html.Div("Chart type", className="ctl-label"),
        dcc.RadioItems(id="chart-type", options=CHART_TYPES, value="bar", className="chart-radio",
                       labelClassName="chart-radio-item"),
        html.Hr(),
        control("x", "lbl-x", "Category", dcc.Dropdown(id="ctl-x", placeholder="Select a column")),
        control("y", "lbl-y", "Value", dcc.Dropdown(id="ctl-y", placeholder="Select a column")),
        control("color", "lbl-color", "Colour", dcc.Dropdown(id="ctl-color", placeholder="None")),
        control("heat", None, "", html.Div([
            html.Label("Numeric columns", className="ctl-label"),
            dcc.Dropdown(id="ctl-heat", multi=True, placeholder="Select columns")])),
        control("agg", None, "", html.Div([
            html.Label("Aggregation", className="ctl-label"),
            dcc.Dropdown(id="ctl-agg", clearable=False, searchable=False, value="count",
                         options=["count", "mean", "sum", "median", "min", "max"])])),
        control("method", None, "", html.Div([
            html.Label("Method", className="ctl-label"),
            dcc.RadioItems(id="ctl-method", value="pearson", className="inline-radio",
                           options=[{"label": "Pearson", "value": "pearson"},
                                    {"label": "Spearman", "value": "spearman"}])])),
        control("bins", None, "", html.Div([
            html.Label("Bins", className="ctl-label"),
            dcc.Slider(id="ctl-bins", min=5, max=100, step=1, value=30, marks={5: "5", 50: "50", 100: "100"},
                       tooltip={"placement": "bottom", "always_visible": False})])),
        control("topn", None, "", html.Div([
            html.Label("Max categories", className="ctl-label"),
            dcc.Slider(id="ctl-topn", min=5, max=30, step=1, value=15, marks={5: "5", 15: "15", 30: "30"},
                       tooltip={"placement": "bottom", "always_visible": False})])),
        control("points", None, "", dcc.Checklist(
            id="ctl-points", className="check", value=[],
            options=[{"label": "Overlay individual points (≤ 2,000 rows)", "value": "on"}])),
        control("trend", None, "", dcc.Checklist(
            id="ctl-trend", className="check", value=[],
            options=[{"label": "Show linear trend line", "value": "on"}])),
        control("annot", None, "", dcc.Checklist(
            id="ctl-annot", className="check", value=["on"],
            options=[{"label": "Show values in cells (≤ 15 columns)", "value": "on"}])),
    ])
    chart_card = html.Section(className="card chart-card", children=[
        html.Div(className="card-head", children=[
            html.Div([html.Div(id="chart-title", className="card-title"),
                      html.Div("Hover for details · drag to zoom · use the toolbar to export a PNG",
                               className="card-sub")])]),
        html.Div(className="card-body", children=[
            html.Div(id="chart-stats", className="chips"),
            dcc.Loading(type="dot", color="#4F46E5", children=dcc.Graph(
                id="main-chart", config={"displaylogo": False, "responsive": True,
                                         "toImageButtonOptions": {"format": "png", "scale": 2,
                                                                  "filename": "eda_chart"}})),
            html.Div(id="chart-guide", className="guide"),
        ]),
    ])
    return [html.Div(className="viz-layout", children=[html.Section(className="card", children=[
        html.Div(className="card-body", children=controls)]), chart_card])]


def page_stats():
    return [
        card("Numeric summary", html.Div(id="st-num"),
             "Count, centre, spread, quartiles, skew and IQR-flagged outliers",
             actions=html.Button("Download column profile (CSV)", id="btn-profile", className="btn btn-ghost")),
        card("Categorical summary", html.Div(id="st-cat"), "Cardinality and the most frequent value per column"),
    ]


def page_insights():
    return [
        html.Div(id="in-summary", className="kpi-grid"),
        card("Findings", html.Div(id="in-list", className="insight-list"),
             "Rule-based observations - evidence to investigate, not conclusions",
             actions=html.Button("Download insights (.md)", id="btn-insights", className="btn btn-ghost")),
    ]


PAGE_BUILDERS = {"overview": page_overview, "quality": page_quality, "visualize": page_visualize,
                 "stats": page_stats, "insights": page_insights}

app.layout = html.Div(className="app-shell", children=[
    dcc.Store(id="dataset-store", storage_type="memory"),
    dcc.Download(id="download-profile"),
    dcc.Download(id="download-insights"),
    sidebar(),
    html.Main(className="main", children=[
        html.Header(className="topbar", children=[
            html.Div([html.H1(id="page-title", className="page-title"), html.P(id="page-subtitle", className="page-sub")]),
            html.Div(id="dataset-chip", className="dataset-chip"),
        ]),
        data_panel(),
        *[html.Div(id=f"page-{key}", className="page", style={"display": "flex" if key == "overview" else "none"},
                   children=PAGE_BUILDERS[key]()) for key, _, _ in PAGES],
    ]),
])


# --------------------------------------------------------------------------- #
# Data loading & navigation
# --------------------------------------------------------------------------- #
@callback(
    Output("dataset-store", "data"), Output("load-msg", "children"),
    Input("upload", "contents"), Input("sample-select", "value"), State("upload", "filename"),
)
def load_dataset(contents, sample, filename):
    try:
        if ctx.triggered_id == "upload" and contents:
            df, name = loader.from_upload(contents, filename), filename
        else:
            df, name = loader.load_sample(sample)
        ds_id = loader.register(df, name)
    except loader.LoadError as exc:
        return no_update, alert("error", str(exc))
    except Exception as exc:  # noqa: BLE001
        return no_update, alert("error", f"Unexpected error while reading the dataset: {exc}")
    ov = prof.overview_stats(df)
    return ({"id": ds_id, "name": name},
            alert("ok", f"Loaded {name} — {ov['rows']:,} rows × {ov['cols']} columns."))


@callback(Output("dataset-chip", "children"), Input("dataset-store", "data"))
def dataset_chip(store):
    entry = loader.get_entry(store)
    if entry is None:
        return html.Span("No dataset loaded", className="chip-muted")
    rows, cols = entry["df"].shape
    return [html.Span(className="dot"), html.Span(entry["name"], className="chip-name"),
            html.Span(f"{rows:,} rows · {cols} cols", className="chip-meta")]


@callback(
    *[Output(f"page-{k}", "style") for k, _, _ in PAGES],
    *[Output(f"nav-{k}", "className") for k, _, _ in PAGES],
    Output("page-title", "children"), Output("page-subtitle", "children"),
    *[Input(f"nav-{k}", "n_clicks") for k, _, _ in PAGES],
)
def navigate(*_):
    active = ctx.triggered_id.replace("nav-", "") if ctx.triggered_id else "overview"
    styles = [{"display": "flex" if k == active else "none"} for k, _, _ in PAGES]
    classes = ["nav-btn active" if k == active else "nav-btn" for k, _, _ in PAGES]
    title, sub = PAGE_META[active]
    return (*styles, *classes, title, sub)


def _df(store):
    return loader.get_df(store)


NO_DATA = empty_state("No dataset loaded", "Upload a file or choose a sample dataset above.")


# --------------------------------------------------------------------------- #
# 01 Overview
# --------------------------------------------------------------------------- #
@callback(
    Output("ov-kpis", "children"), Output("ov-donut", "figure"), Output("ov-columns", "children"),
    Output("ov-preview", "children"),
    Input("dataset-store", "data"), Input("preview-rows", "value"),
)
def render_overview(store, n_rows):
    df = _df(store)
    if df is None:
        return [], charts.empty_figure("No dataset"), NO_DATA, NO_DATA
    ov = prof.overview_stats(df)
    kpis = [
        kpi("Rows", f"{ov['rows']:,}", "records", "indigo"),
        kpi("Columns", f"{ov['cols']}", "variables", "teal"),
        kpi("Numeric", f"{ov['numeric']}", "columns", "indigo"),
        kpi("Categorical", f"{ov['categorical']}", f"+ {ov['datetime']} datetime" if ov["datetime"] else "columns", "teal"),
        kpi("Completeness", f"{ov['completeness']:.1f}%", f"{ov['missing_cells']:,} empty cells",
            "green" if ov["missing_cells"] == 0 else "amber"),
        kpi("Memory", prof.fmt_bytes(ov["memory"]), "in memory", "slate"),
    ]
    profile = prof.column_profile(df)
    table = html_table(profile, thousands=True, renderers={"Type": type_badge, "Missing %": pct_bar}, max_height=300)
    return kpis, charts.type_donut(df), table, html_table(df, max_rows=int(n_rows or 10), max_height=440)


# --------------------------------------------------------------------------- #
# 02 Data quality
# --------------------------------------------------------------------------- #
@callback(
    Output("q-kpis", "children"), Output("q-missing-bar", "figure"), Output("q-missing-matrix", "figure"),
    Output("q-dups", "children"), Output("q-text", "children"),
    Input("dataset-store", "data"),
)
def render_quality(store):
    df = _df(store)
    if df is None:
        e = charts.empty_figure("No dataset")
        return [], e, e, NO_DATA, NO_DATA
    ov = prof.overview_stats(df)
    extra, involved, preview = prof.duplicate_rows(df)
    ti = prof.text_issues(df)
    kpis = [
        kpi("Completeness", f"{ov['completeness']:.1f}%", "of all cells filled", "green" if ov["missing_cells"] == 0 else "amber"),
        kpi("Missing cells", f"{ov['missing_cells']:,}", f"{ov['missing_pct']:.2f}% of cells", "amber" if ov["missing_cells"] else "green"),
        kpi("Columns affected", f"{ov['cols_with_missing']}", f"of {ov['cols']} columns", "indigo"),
        kpi("Duplicate rows", f"{extra:,}", f"{extra / ov['rows'] * 100:.2f}% of rows", "red" if extra else "green"),
        kpi("Text issues", f"{len(ti)}", "columns to review", "amber" if len(ti) else "green"),
    ]
    if extra == 0:
        dups = alert("ok", "No duplicate rows were found.")
    else:
        dups = html.Div([
            html.P(f"{extra:,} row(s) repeat an earlier record exactly. The {involved:,} rows involved are listed below "
                   f"(first {len(preview)} shown). Decide whether they are an input problem or a legitimate repeat.",
                   className="muted-p"),
            html_table(preview, max_height=320)])
    text = alert("ok", "No whitespace or case/spelling inconsistencies were found.") if ti.empty else html.Div([
        html.P("The same category may be counted as several values in charts until these are reviewed.", className="muted-p"),
        html_table(ti, thousands=True, max_height=300)])
    return kpis, charts.missing_bar(df), charts.missing_matrix(df), dups, text


# --------------------------------------------------------------------------- #
# 03 Visualize
# --------------------------------------------------------------------------- #
def _opts(cols):
    return [{"label": c, "value": c} for c in cols]


def _low_card(df, cols, limit):
    return [c for c in cols if df[c].nunique(dropna=True) <= limit]


def _default_scatter_pair(df, cols):
    """Start from the most strongly (linearly) associated pair - an informative first view."""
    cols = cols[:40]
    if len(cols) < 2:
        return (cols[0] if cols else None), None
    corr = df[cols].corr().abs()
    best, pair = -1.0, (cols[0], cols[1])
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r > best:
                best, pair = r, (a, b)
    return pair


@callback(
    Output("ctl-x", "options"), Output("ctl-x", "value"),
    Output("ctl-y", "options"), Output("ctl-y", "value"),
    Output("ctl-color", "options"), Output("ctl-color", "value"),
    Output("ctl-heat", "options"), Output("ctl-heat", "value"),
    Output("lbl-x", "children"), Output("lbl-y", "children"), Output("lbl-color", "children"),
    *[Output(f"wrap-{w}", "style") for w in WRAPPERS],
    Input("dataset-store", "data"), Input("chart-type", "value"),
)
def configure_controls(store, chart):
    shown = VISIBLE[chart]
    styles = [{} if w in shown else {"display": "none"} for w in WRAPPERS]
    lx, ly, lc = LABELS[chart]
    df = _df(store)
    if df is None:
        return ([], None, [], None, [], None, [], None, lx, ly, lc, *styles)

    types = prof.split_columns(df)
    n = len(df)
    all_num = types.numeric
    good_num = prof.analysis_numeric(df) or all_num
    continuous = [c for c in good_num if df[c].nunique(dropna=True) > 12] or good_num
    cat_cols = [c for c in types.categorical_like if df[c].nunique(dropna=True) < n or n <= 20]
    cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= 500]
    small_cat = _low_card(df, types.categorical_like + [c for c in all_num if df[c].nunique() <= 12], 30)

    x_opts = y_opts = c_opts = []
    x_val = y_val = c_val = None
    if chart == "bar":
        x_cols = cat_cols + types.datetime + _low_card(df, all_num, 20)
        x_opts, y_opts, c_opts = x_cols, all_num, _low_card(df, types.categorical_like, 12)
        preferred = [c for c in x_cols if 2 <= df[c].nunique() <= 12]
        x_val = (preferred or x_cols or [None])[0]
    elif chart == "hist":
        x_opts, c_opts = all_num, _low_card(df, types.categorical_like, 8)
        x_val = (continuous or [None])[0]
    elif chart == "box":
        x_opts, y_opts = _low_card(df, types.categorical_like, 30), all_num
        y_val = (continuous or [None])[0]
        x_val = (_low_card(df, types.categorical_like, 8) or [None])[0]
    elif chart == "scatter":
        x_opts = y_opts = all_num
        x_val, y_val = _default_scatter_pair(df, continuous)
        c_opts = small_cat + [c for c in all_num if c not in small_cat]
    heat_val = good_num[:12]
    return (_opts(x_opts), x_val, _opts(y_opts), y_val, _opts(c_opts), c_val,
            _opts(all_num), heat_val, lx, ly, lc, *styles)


@callback(
    Output("main-chart", "figure"), Output("chart-title", "children"),
    Output("chart-stats", "children"), Output("chart-guide", "children"),
    Input("dataset-store", "data"), Input("chart-type", "value"),
    Input("ctl-x", "value"), Input("ctl-y", "value"), Input("ctl-color", "value"),
    Input("ctl-bins", "value"), Input("ctl-agg", "value"), Input("ctl-topn", "value"),
    Input("ctl-points", "value"), Input("ctl-trend", "value"), Input("ctl-annot", "value"),
    Input("ctl-heat", "value"), Input("ctl-method", "value"),
)
def render_chart(store, chart, x, y, color, bins, agg, topn, points, trend, annot, heat, method):
    df = _df(store)
    title, guide = CHART_TITLES[chart], GUIDES[chart]
    if df is None:
        return charts.empty_figure("Load a dataset to start exploring"), title, [], guide
    try:
        if chart == "bar":
            fig, stats = charts.bar_chart(df, x, y, agg, color, int(topn or 15))
        elif chart == "hist":
            fig, stats = charts.histogram_chart(df, x, bins, color)
        elif chart == "box":
            fig, stats = charts.box_chart(df, y, x, bool(points))
        elif chart == "scatter":
            fig, stats = charts.scatter_chart(df, x, y, color, bool(trend))
        else:
            fig, stats = charts.heatmap_chart(df, heat, method, bool(annot))
    except charts.ChartError as exc:
        return charts.empty_figure(str(exc)), title, [], guide
    except Exception as exc:  # noqa: BLE001 - never crash the UI on an odd column
        return charts.empty_figure(f"This selection could not be plotted ({type(exc).__name__}: {exc})"), title, [], guide
    return fig, title, [stat_chip(k, v) for k, v in stats], guide


# --------------------------------------------------------------------------- #
# 04 Statistics
# --------------------------------------------------------------------------- #
@callback(Output("st-num", "children"), Output("st-cat", "children"), Input("dataset-store", "data"))
def render_stats(store):
    df = _df(store)
    if df is None:
        return NO_DATA, NO_DATA
    num = prof.numeric_summary(df)
    cat = prof.categorical_summary(df)
    num_tbl = html_table(num, thousands=True, max_height=480) if not num.empty else alert("info", "No numeric columns in this dataset.")
    cat_tbl = html_table(cat, thousands=True, max_height=420) if not cat.empty else alert("info", "No categorical columns in this dataset.")
    return num_tbl, cat_tbl


@callback(Output("download-profile", "data"), Input("btn-profile", "n_clicks"),
          State("dataset-store", "data"), prevent_initial_call=True)
def download_profile(_, store):
    entry = loader.get_entry(store)
    if entry is None:
        return no_update
    merged = prof.column_profile(entry["df"])
    stem = Path(entry["name"]).stem
    return dcc.send_data_frame(merged.to_csv, f"{stem}_column_profile.csv", index=False)


# --------------------------------------------------------------------------- #
# 05 Insights
# --------------------------------------------------------------------------- #
@callback(Output("in-summary", "children"), Output("in-list", "children"), Input("dataset-store", "data"))
def render_insights(store):
    df = _df(store)
    if df is None:
        return [], NO_DATA
    items = prof.generate_insights(df)
    count = lambda lvl: sum(1 for i in items if i["level"] == lvl)  # noqa: E731
    summary = [kpi("Needs attention", str(count("warn")), "issues to review", "amber" if count("warn") else "green"),
               kpi("Notes", str(count("info")), "observations", "indigo"),
               kpi("Looks good", str(count("good")), "checks passed", "green")]
    return summary, [insight_card(i["level"], i["title"], i["text"]) for i in items]


@callback(Output("download-insights", "data"), Input("btn-insights", "n_clicks"),
          State("dataset-store", "data"), prevent_initial_call=True)
def download_insights(_, store):
    entry = loader.get_entry(store)
    if entry is None:
        return no_update
    text = prof.insights_to_markdown(prof.generate_insights(entry["df"]), entry["name"])
    return dcc.send_string(text, f"{Path(entry['name']).stem}_insights.md")


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8050")),
        debug=os.environ.get("DEBUG", "0") == "1",
    )
