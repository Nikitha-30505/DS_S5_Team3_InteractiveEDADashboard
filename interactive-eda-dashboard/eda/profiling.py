"""Dataset profiling, data-quality checks and rule-based insights.

Everything here reports evidence; nothing modifies the dataset. Findings are
phrased as possibilities because their meaning depends on the data's context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Column typing
# --------------------------------------------------------------------------- #
@dataclass
class ColumnTypes:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)

    @property
    def categorical_like(self) -> list[str]:
        return self.categorical + self.boolean


def split_columns(df: pd.DataFrame) -> ColumnTypes:
    out = ColumnTypes()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            out.boolean.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            out.numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            out.datetime.append(col)
        else:
            out.categorical.append(col)
    return out


def type_label(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "Boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "Numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "Datetime"
    return "Categorical"


_ID_NAME = re.compile(r"(^|[_\s-])(id|uuid|key|index|idx|code|no|number)$|^(id|index)([_\s-]|$)", re.IGNORECASE)


def is_id_like(s: pd.Series, name: str | None = None) -> bool:
    """Integer column with a unique value per row that looks like an identifier.

    Unique integers alone are not enough (a rounded measurement can be unique), so
    the column must also be sequential (monotonic) or be named like an id.
    """
    if not (pd.api.types.is_integer_dtype(s) and len(s) > 20 and s.notna().all() and s.nunique() == len(s)):
        return False
    return bool(s.is_monotonic_increasing or s.is_monotonic_decreasing or (name and _ID_NAME.search(str(name))))


def analysis_numeric(df: pd.DataFrame) -> list[str]:
    """Numeric columns worth analysing (no ids, no constants, no all-NaN)."""
    cols = []
    for c in split_columns(df).numeric:
        s = df[c]
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1 or is_id_like(s, c):
            continue
        cols.append(c)
    return cols


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GB"


def fmt_num(v: float, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if float(v).is_integer() and abs(v) < 1e15:
        return f"{int(v):,}"
    text = f"{v:,.{digits}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


# --------------------------------------------------------------------------- #
# Overview and column profile
# --------------------------------------------------------------------------- #
def count_duplicates(df: pd.DataFrame) -> int:
    try:
        return int(df.duplicated().sum())
    except TypeError:  # unhashable cells (lists, dicts)
        return int(df.astype(str).duplicated().sum())


def overview_stats(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    cells = rows * cols
    types = split_columns(df)
    return {
        "rows": rows,
        "cols": cols,
        "numeric": len(types.numeric),
        "categorical": len(types.categorical_like),
        "datetime": len(types.datetime),
        "missing_cells": missing,
        "missing_pct": (missing / cells * 100) if cells else 0.0,
        "completeness": 100 - (missing / cells * 100 if cells else 0.0),
        "cols_with_missing": int((df.isna().sum() > 0).sum()),
        "duplicates": count_duplicates(df),
        "memory": int(df.memory_usage(deep=True).sum()),
    }


def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.astype(str).nunique())


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        kind = type_label(s)
        miss = int(s.isna().sum())
        if kind == "Numeric" and s.notna().any():
            summary = f"{fmt_num(s.min())} to {fmt_num(s.max())}"
        elif kind == "Datetime" and s.notna().any():
            summary = f"{s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}"
        elif s.notna().any():
            try:
                top = s.astype(str).value_counts().index[0]
            except Exception:  # noqa: BLE001
                top = "—"
            summary = f"Top: {top}"
        else:
            summary = "—"
        rows.append(
            {
                "Column": col,
                "Type": kind,
                "Non-null": n - miss,
                "Missing": miss,
                "Missing %": round(miss / n * 100, 2) if n else 0.0,
                "Unique": _safe_nunique(s),
                "Summary": summary,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def duplicate_rows(df: pd.DataFrame, limit: int = 30) -> tuple[int, int, pd.DataFrame]:
    """Return (extra duplicate rows, rows involved, preview of involved rows)."""
    try:
        mask = df.duplicated(keep=False)
        extra = int(df.duplicated().sum())
    except TypeError:
        as_text = df.astype(str)
        mask = as_text.duplicated(keep=False)
        extra = int(as_text.duplicated().sum())
    positions = np.flatnonzero(mask.to_numpy())
    preview = df.iloc[positions[:limit]].copy()
    preview.insert(0, "Row #", positions[:limit] + 1)
    return extra, int(mask.sum()), preview


def text_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Whitespace and spelling/case variants in text columns."""
    out = []
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        s = s.dropna()
        if s.empty or not s.map(lambda v: isinstance(v, str)).all():
            continue
        if s.nunique() > 2000:
            continue
        stripped = s.str.strip()
        whitespace = int((s != stripped).sum())
        canon = stripped.str.lower()
        groups = pd.DataFrame({"canon": canon, "raw": stripped}).groupby("canon")["raw"].agg(
            lambda x: sorted(set(x))
        )
        variants = groups[groups.map(len) > 1]
        if whitespace == 0 and variants.empty:
            continue
        example = ", ".join(f"'{v}'" for v in (variants.iloc[0][:3] if not variants.empty else []))
        out.append(
            {
                "Column": col,
                "Values with extra whitespace": whitespace,
                "Case/spelling variants": len(variants),
                "Example variants": example or "—",
            }
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def iqr_outlier_mask(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return pd.Series(False, index=s.index)
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def numeric_summary(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols if cols is not None else split_columns(df).numeric
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        n = int(s.notna().sum())
        if n == 0:
            continue
        out_n = int(iqr_outlier_mask(s).sum())
        rows.append(
            {
                "Column": c,
                "Count": n,
                "Mean": s.mean(),
                "Std": s.std(),
                "Min": s.min(),
                "Q1": s.quantile(0.25),
                "Median": s.median(),
                "Q3": s.quantile(0.75),
                "Max": s.max(),
                "Skew": s.skew(),
                "Kurtosis": s.kurt(),
                "Outliers (IQR)": out_n,
                "Outlier %": round(out_n / n * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in split_columns(df).categorical_like:
        s = df[c].dropna()
        if s.empty:
            continue
        vc = s.astype(str).value_counts()
        rows.append(
            {
                "Column": c,
                "Count": int(len(s)),
                "Unique": int(len(vc)),
                "Top value": vc.index[0],
                "Top frequency": int(vc.iloc[0]),
                "Top %": round(vc.iloc[0] / len(s) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def top_correlations(corr: pd.DataFrame, threshold: float = 0.7) -> list[tuple[str, str, float]]:
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append((a, b, float(r)))
    return sorted(pairs, key=lambda t: -abs(t[2]))


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
_LEVEL_RANK = {"warn": 0, "info": 1, "good": 2}


def generate_insights(df: pd.DataFrame) -> list[dict]:
    """Rule-based findings. Wording is deliberately cautious (evidence, not verdicts)."""
    out: list[dict] = []

    def add(level: str, title: str, text: str) -> None:
        out.append({"level": level, "title": title, "text": text})

    ov = overview_stats(df)
    n = ov["rows"]
    types = split_columns(df)
    add(
        "info",
        "Dataset structure",
        f"{ov['rows']:,} rows × {ov['cols']} columns: {ov['numeric']} numeric, "
        f"{ov['categorical']} categorical and {ov['datetime']} datetime. "
        f"Approximate memory use is {fmt_bytes(ov['memory'])}.",
    )

    # Missing values
    if ov["missing_cells"] == 0:
        add("good", "No missing values", "Every cell contains a value, so no completeness issues were detected.")
    else:
        miss = (df.isna().mean() * 100).sort_values(ascending=False)
        miss = miss[miss > 0]
        top = ", ".join(f"{c} ({v:.1f}%)" for c, v in miss.head(3).items())
        level = "warn" if (miss.iloc[0] >= 5 or ov["missing_pct"] >= 2) else "info"
        text = (
            f"{ov['missing_cells']:,} cells ({ov['missing_pct']:.2f}%) are empty across "
            f"{ov['cols_with_missing']} column(s). Most affected: {top}."
        )
        if miss.iloc[0] >= 40:
            text += " One column is missing more than 40% of its values; check whether the gaps are informative before using it."
        text += " Missing values are reported, not removed - decide on treatment with the data's context in mind."
        add(level, "Missing values detected", text)

    # Duplicates
    if ov["duplicates"] == 0:
        add("good", "No duplicate rows", "No fully repeated records were found.")
    else:
        pct = ov["duplicates"] / n * 100
        add(
            "warn",
            "Duplicate rows detected",
            f"{ov['duplicates']:,} row(s) ({pct:.2f}%) repeat an earlier record exactly. "
            "Check whether they are an input problem or a legitimate repeat before removing them.",
        )

    # Constant / identifier-like columns
    constants = [c for c in df.columns if _safe_nunique(df[c]) <= 1]
    if constants:
        add(
            "info",
            "Constant columns",
            f"{', '.join(constants[:6])} hold a single value (or are empty), so they cannot help explain differences between rows.",
        )
    id_like = [
        c for c in df.columns
        if (is_id_like(df[c], c) or (c in types.categorical and n > 20 and df[c].notna().all() and _safe_nunique(df[c]) == n))
    ]
    if id_like:
        add(
            "info",
            "Identifier-like columns",
            f"{', '.join(id_like[:6])} have a unique value on every row and look like identifiers. "
            "They are excluded from correlation analysis.",
        )

    # Text consistency
    ti = text_issues(df)
    if not ti.empty:
        cols = ", ".join(ti["Column"].head(4))
        add(
            "warn",
            "Inconsistent text values",
            f"Whitespace or case/spelling variants were found in: {cols}. "
            "The same category may be counted as several different values in charts.",
        )

    num_cols = analysis_numeric(df)

    # Skewness
    if num_cols:
        skews = df[num_cols].skew(numeric_only=True)
        skewed = skews[skews.abs() >= 1].sort_values(key=lambda x: -x.abs())
        if not skewed.empty:
            txt = ", ".join(f"{c} (skew {v:.2f})" for c, v in skewed.head(4).items())
            add(
                "info",
                "Skewed distributions",
                f"{txt} are noticeably asymmetric. Compare the mean and median in the histogram; "
                "a log scale or other transformation may be worth considering before modelling.",
            )

        # Outliers
        flagged = []
        for c in num_cols:
            mask = iqr_outlier_mask(df[c])
            k = int(mask.sum())
            if k:
                flagged.append((c, k, k / max(int(df[c].notna().sum()), 1) * 100))
        flagged.sort(key=lambda t: -t[2])
        if flagged:
            txt = ", ".join(f"{c} ({k:,} values, {p:.1f}%)" for c, k, p in flagged[:4])
            add(
                "info",
                "Possible outliers",
                f"The 1.5×IQR rule flags values outside the typical range in: {txt}. "
                "These may be valid events, measurement errors or a distinct subgroup - inspect them in the box plot.",
            )

    # Correlations
    if 2 <= len(num_cols) <= 60:
        corr = df[num_cols].corr()
        pairs = top_correlations(corr, 0.7)
        if pairs:
            txt = "; ".join(f"{a} ↔ {b} (r = {r:+.2f})" for a, b, r in pairs[:5])
            extra = ""
            if any(abs(r) >= 0.95 for _, _, r in pairs):
                extra = " Near-perfect pairs (|r| ≥ 0.95) may be redundant measurements of the same thing."
            add(
                "info",
                "Strong linear associations",
                f"{txt}.{extra} Correlation shows association only; it does not establish causation.",
            )
        else:
            add("info", "No strong linear correlations", "No numeric pair reaches |r| ≥ 0.7. Check the scatter plot for non-linear patterns.")

    # Dominant categories
    dominated = []
    for c in types.categorical_like:
        s = df[c].dropna()
        if s.empty or n <= 20:
            continue
        vc = s.astype(str).value_counts(normalize=True)
        if 2 <= len(vc) <= 50 and vc.iloc[0] >= 0.8:
            dominated.append(f"{c} ('{vc.index[0]}' = {vc.iloc[0] * 100:.0f}%)")
    if dominated:
        add(
            "info",
            "Dominant categories",
            f"{', '.join(dominated[:4])} are dominated by a single value, which can limit comparisons between groups.",
        )

    out.sort(key=lambda d: _LEVEL_RANK[d["level"]])
    return out


def insights_to_markdown(insights: list[dict], dataset_name: str) -> str:
    badge = {"warn": "Attention", "info": "Note", "good": "OK"}
    lines = [f"# EDA insights - {dataset_name}", ""]
    for item in insights:
        lines.append(f"## [{badge[item['level']]}] {item['title']}")
        lines.append(item["text"])
        lines.append("")
    lines.append("_Generated by the Interactive EDA Dashboard. Findings are evidence to investigate, not conclusions._")
    return "\n".join(lines)
