"""Plotly figure builders for the five EDA views plus supporting quality charts.

Every builder returns ``(figure, stats)`` where ``stats`` is a list of
``(label, value)`` pairs shown as chips beside the chart. Builders raise
``ChartError`` with a friendly message when the selection cannot be plotted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .profiling import fmt_num, iqr_outlier_mask, split_columns, type_label

INDIGO, TEAL, AMBER, PINK = "#4F46E5", "#0EA5A4", "#F59E0B", "#EC4899"
SKY, VIOLET, GREEN, SLATE, RED = "#0EA5E9", "#8B5CF6", "#10B981", "#64748B", "#EF4444"
PALETTE = [INDIGO, TEAL, AMBER, PINK, SKY, VIOLET, GREEN, SLATE]
DIVERGING = [[0.0, "#1D4ED8"], [0.25, "#93C5FD"], [0.5, "#F8FAFC"], [0.75, "#FCA5A5"], [1.0, "#B91C1C"]]
SEQUENTIAL = ["#E0E7FF", "#818CF8", INDIGO, "#1E1B4B"]
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MISSING_LABEL = "(missing)"
MAX_POINTS = 20000


class ChartError(ValueError):
    """A selection that cannot be charted (shown to the user as a message)."""


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, title: str | None = None, height: int = 520) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family=FONT, size=12, color="#334155"),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16, color="#0F172A")) if title else None,
        colorway=PALETTE,
        margin=dict(l=64, r=24, t=64 if title else 24, b=56),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#F8FAFC", bordercolor="#0F172A"),
    )
    axis = dict(
        gridcolor="#EEF2F7",
        zerolinecolor="#E2E8F0",
        linecolor="#CBD5E1",
        ticks="outside",
        tickcolor="#CBD5E1",
        title_font=dict(size=12, color="#475569"),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def empty_figure(message: str = "Nothing to display", height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font=dict(size=14, color="#64748B"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=height)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require(df: pd.DataFrame, *cols: str | None) -> None:
    for c in cols:
        if c is not None and c not in df.columns:
            raise ChartError(f"Column '{c}' is not in this dataset.")


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    _require(df, col)
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        raise ChartError(f"'{col}' has no numeric values to plot.")
    return s


def _as_label(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), MISSING_LABEL).astype(str)


def _top_categories(s: pd.Series, k: int) -> list[str]:
    return s.value_counts().head(k).index.tolist()


# --------------------------------------------------------------------------- #
# Bar chart
# --------------------------------------------------------------------------- #
def bar_chart(df, x, y=None, agg="count", color=None, top_n=15):
    if not x:
        raise ChartError("Choose a category column for the X axis.")
    _require(df, x, y, color)
    use_value = bool(y) and agg != "count"
    split = color if color and color != x else None
    keys = [x] + ([split] if split else [])
    work = df[keys + ([y] if use_value else [])].copy()
    if use_value:
        work[y] = pd.to_numeric(work[y], errors="coerce")

    is_dt = pd.api.types.is_datetime64_any_dtype(work[x])
    if is_dt:
        mask = work[x].notna()
        work[x] = work[x].dt.to_period("M").astype(str).where(mask, MISSING_LABEL)
    else:
        work[x] = _as_label(work[x])
    if split:
        work[split] = _as_label(work[split])

    if use_value:
        g = work.groupby(keys, observed=True)[y].agg(agg).reset_index(name="value")
        ylab = f"{agg.title()} of {y}"
    else:
        g = work.groupby(keys, observed=True).size().reset_index(name="value")
        ylab = "Count"
    if g.empty:
        raise ChartError("No data available for this selection.")

    totals = g.groupby(x)["value"].agg("sum" if (not use_value or agg in ("count", "sum")) else "mean")
    if is_dt:
        order = sorted(totals.index)
        shown_note = ""
    else:
        order = totals.sort_values(ascending=False).index.tolist()
        total_cats = len(order)
        order = order[:top_n]
        shown_note = f" (top {top_n} of {total_cats})" if total_cats > top_n else ""
    g = g[g[x].isin(order)]

    fig = px.bar(
        g, x=x, y="value", color=split, barmode="group",
        category_orders={x: order}, labels={"value": ylab},
    )
    if not split and len(order) <= 20:
        fmt = "%{y:.3~s}" if g["value"].max() >= 1000 else "%{y:,.2~f}"
        fig.update_traces(texttemplate=fmt, textposition="outside", cliponaxis=False)
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title=x, type="category")
    fig.update_yaxes(title=ylab)
    style(fig, f"{ylab} by {x}{shown_note}")
    stats = [("Categories shown", f"{len(order):,}"), ("Rows used", f"{len(work):,}")]
    if is_dt:
        stats.append(("Grouped by", "month"))
    return fig, stats


# --------------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------------- #
def histogram_chart(df, x, bins=30, color=None):
    if not x:
        raise ChartError("Choose a numeric column.")
    s = _numeric(df, x).dropna()
    bins = int(bins or 30)
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    custom = np.column_stack([edges[:-1], edges[1:]])

    fig = go.Figure()

    def add(values, name, colour, opacity, show_legend):
        counts, _ = np.histogram(values, bins=edges)
        fig.add_bar(
            x=centers, y=counts, width=width * 0.96, name=name, marker_color=colour,
            opacity=opacity, customdata=custom, showlegend=show_legend,
            hovertemplate="Range: %{customdata[0]:,.4g} to %{customdata[1]:,.4g}<br>Count: %{y:,}"
                          f"<extra>{name}</extra>",
        )

    if color and color != x:
        _require(df, color)
        labels = _as_label(df.loc[s.index, color])
        for i, cat in enumerate(_top_categories(labels, 6)):
            add(s[labels == cat], str(cat), PALETTE[i % len(PALETTE)], 0.62, True)
        fig.update_layout(barmode="overlay")
    else:
        add(s, "All values", INDIGO, 0.92, False)
        mean, median = float(s.mean()), float(s.median())
        fig.add_vline(x=mean, line_dash="dash", line_color=AMBER, line_width=2,
                      annotation_text=f"Mean {fmt_num(mean)}", annotation_position="top right",
                      annotation_font=dict(color=AMBER, size=11))
        fig.add_vline(x=median, line_dash="dot", line_color=TEAL, line_width=2,
                      annotation_text=f"Median {fmt_num(median)}", annotation_position="top left",
                      annotation_font=dict(color=TEAL, size=11))
    fig.update_xaxes(title=x)
    fig.update_yaxes(title="Count")
    style(fig, f"Distribution of {x}")
    stats = [
        ("Values", f"{len(s):,}"), ("Mean", fmt_num(s.mean())), ("Median", fmt_num(s.median())),
        ("Std dev", fmt_num(s.std())), ("Skew", fmt_num(s.skew(), 2)),
    ]
    return fig, stats


# --------------------------------------------------------------------------- #
# Box plot
# --------------------------------------------------------------------------- #
def box_chart(df, y, x=None, show_points=False):
    if not y:
        raise ChartError("Choose a numeric column.")
    values = _numeric(df, y)
    cols = {"v": values}
    group = x if x and x != y else None
    if group:
        _require(df, group)
        cols["g"] = _as_label(df[group])
    work = pd.DataFrame(cols).dropna(subset=["v"])
    if len(work) > 50000:
        work = work.sample(50000, random_state=0)
    points = "all" if (show_points and len(work) <= 2000) else "outliers"

    if group:
        order = _top_categories(work["g"], 12)
        work = work[work["g"].isin(order)]
        fig = px.box(work, x="g", y="v", color="g", points=points, category_orders={"g": order},
                     color_discrete_sequence=PALETTE)
        fig.update_xaxes(title=group, type="category")
        fig.update_yaxes(title=y)
        fig.update_layout(showlegend=False)
        title = f"{y} by {group}"
    else:
        fig = px.box(work, x="v", points=points, color_discrete_sequence=[INDIGO])
        fig.update_xaxes(title=y)
        fig.update_yaxes(visible=False)
        title = f"Spread of {y}"
    fig.update_traces(boxmean=True, marker=dict(size=4, opacity=0.6), line=dict(width=1.6))
    style(fig, title)

    s = work["v"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    stats = [
        ("Values", f"{len(s):,}"), ("Median", fmt_num(s.median())), ("IQR", fmt_num(q3 - q1)),
        ("Flagged outliers", f"{int(iqr_outlier_mask(s).sum()):,}"),
    ]
    if group:
        stats.insert(1, ("Groups", f"{work['g'].nunique():,}"))
    return fig, stats


# --------------------------------------------------------------------------- #
# Scatter plot
# --------------------------------------------------------------------------- #
def scatter_chart(df, x, y, color=None, trendline=False):
    if not x or not y:
        raise ChartError("Choose numeric columns for both axes.")
    xs, ys = _numeric(df, x), _numeric(df, y)
    work = pd.DataFrame({x: xs, y: ys}) if x != y else pd.DataFrame({x: xs})
    if x == y:
        work[y + " "] = xs
        y_plot = y + " "
    else:
        y_plot = y
    color_kind = None
    if color:
        _require(df, color)
        c = df[color]
        if pd.api.types.is_numeric_dtype(c) and not pd.api.types.is_bool_dtype(c) and c.nunique() > 12:
            work["_c"], color_kind = pd.to_numeric(c, errors="coerce"), "continuous"
        else:
            labels = _as_label(c)
            keep = _top_categories(labels, 8)
            work["_c"] = labels.where(labels.isin(keep), "Other")
            color_kind = "discrete"
    work = work.dropna(subset=[x, y_plot])
    if work.empty:
        raise ChartError("No rows have values for both selected columns.")
    n_valid = len(work)
    valid_x, valid_y = work[x].to_numpy(float), work[y_plot].to_numpy(float)
    plot = work.sample(MAX_POINTS, random_state=0) if n_valid > MAX_POINTS else work

    kwargs = dict(x=x, y=y_plot, opacity=0.72)
    if color_kind == "continuous":
        kwargs.update(color="_c", color_continuous_scale=SEQUENTIAL, labels={"_c": color})
    elif color_kind == "discrete":
        kwargs.update(color="_c", labels={"_c": color}, color_discrete_sequence=PALETTE)
    else:
        kwargs.update(color_discrete_sequence=[INDIGO])
    fig = px.scatter(plot, **kwargs)
    fig.update_traces(marker=dict(size=7, line=dict(width=0.6, color="white")))
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)

    stats = [("Points", f"{n_valid:,}")]
    if n_valid > MAX_POINTS:
        stats.append(("Drawn (sampled)", f"{MAX_POINTS:,}"))
    if n_valid >= 3 and np.std(valid_x) > 0 and np.std(valid_y) > 0:
        r = float(np.corrcoef(valid_x, valid_y)[0, 1])
        rho = float(pd.Series(valid_x).corr(pd.Series(valid_y), method="spearman"))
        stats += [("Pearson r", f"{r:+.3f}"), ("Spearman ρ", f"{rho:+.3f}")]
        if trendline:
            slope, intercept = np.polyfit(valid_x, valid_y, 1)
            xx = np.array([valid_x.min(), valid_x.max()])
            fig.add_scatter(x=xx, y=slope * xx + intercept, mode="lines", name="Linear fit",
                            line=dict(color=RED, width=2.5, dash="dash"), hoverinfo="skip")
            stats.append(("R²", f"{r * r:.3f}"))
    style(fig, f"{y} vs {x}")
    return fig, stats


# --------------------------------------------------------------------------- #
# Correlation heatmap
# --------------------------------------------------------------------------- #
def heatmap_chart(df, cols, method="pearson", annotate=True):
    cols = [c for c in (cols or []) if c in df.columns]
    if len(cols) < 2:
        raise ChartError("Select at least two numeric columns.")
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = data.corr(method=method or "pearson")
    n = len(cols)
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
            colorscale=DIVERGING, xgap=2, ygap=2,
            colorbar=dict(title="r", thickness=14, outlinewidth=0),
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
            text=np.round(corr.to_numpy(), 2),
            texttemplate="%{text:.2f}" if (annotate and n <= 15) else None,
            textfont=dict(size=11),
        )
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(tickangle=-35, title=None, showgrid=False)
    style(fig, f"{(method or 'pearson').title()} correlation matrix", height=int(min(760, max(440, 54 * n + 170))))
    fig.update_yaxes(showgrid=False)

    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
             if pd.notna(corr.loc[a, b])]
    stats = [("Columns", f"{n}"), ("Method", (method or "pearson").title())]
    if pairs:
        a, b, r = max(pairs, key=lambda t: abs(t[2]))
        stats.append(("Strongest pair", f"{a} ↔ {b} ({r:+.2f})"))
    return fig, stats


# --------------------------------------------------------------------------- #
# Overview & quality charts
# --------------------------------------------------------------------------- #
TYPE_COLORS = {"Numeric": INDIGO, "Categorical": TEAL, "Datetime": AMBER, "Boolean": VIOLET}


def type_donut(df: pd.DataFrame) -> go.Figure:
    labels = pd.Series([type_label(df[c]) for c in df.columns]).value_counts()
    fig = go.Figure(
        go.Pie(
            labels=labels.index, values=labels.values, hole=0.64, sort=False,
            marker=dict(colors=[TYPE_COLORS[l] for l in labels.index], line=dict(color="white", width=3)),
            textinfo="value", textfont=dict(size=13, color="white"),
            hovertemplate="%{label}: %{value} column(s) (%{percent})<extra></extra>",
        )
    )
    fig.add_annotation(text=f"<b>{len(df.columns)}</b><br><span style='font-size:11px;color:#64748B'>columns</span>",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=24, color="#0F172A"))
    style(fig, height=320)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.04, yanchor="top", x=0.5, xanchor="center"))
    return fig


def missing_bar(df: pd.DataFrame) -> go.Figure:
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values()
    if miss.empty:
        return empty_figure("✓ No missing values found", 300)
    colors = [GREEN if v < 5 else AMBER if v < 20 else RED for v in miss.values]
    fig = go.Figure(
        go.Bar(
            x=miss.values, y=list(miss.index), orientation="h", marker_color=colors,
            text=[f"{v:.1f}%" for v in miss.values], textposition="outside", cliponaxis=False,
            customdata=[int(df[c].isna().sum()) for c in miss.index],
            hovertemplate="%{y}<br>%{x:.2f}% missing (%{customdata:,} rows)<extra></extra>",
        )
    )
    fig.update_xaxes(title="% of rows missing", range=[0, max(miss.max() * 1.25, 5)])
    fig.update_yaxes(title=None, showgrid=False)
    style(fig, height=int(max(300, 38 * len(miss) + 110)))
    fig.update_layout(margin=dict(l=10, r=40, t=10, b=50))
    return fig


def missing_matrix(df: pd.DataFrame, max_rows: int = 150) -> go.Figure:
    if int(df.isna().sum().sum()) == 0:
        return empty_figure("✓ No missing values found", 300)
    idx = np.linspace(0, len(df) - 1, min(len(df), max_rows)).astype(int)
    m = df.iloc[idx].isna().to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=m.astype(int), x=list(df.columns), y=idx + 1, xgap=1, showscale=False,
            colorscale=[[0, "#EEF2F7"], [1, INDIGO]],
            customdata=np.where(m, "missing", "present"),
            hovertemplate="Row %{y} · %{x}<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", title="Row # (evenly sampled)", showgrid=False)
    fig.update_xaxes(tickangle=-40, showgrid=False)
    style(fig, height=int(max(300, 38 * int((df.isna().sum() > 0).sum()) + 110)))
    fig.update_layout(margin=dict(l=60, r=10, t=10, b=90))
    return fig
