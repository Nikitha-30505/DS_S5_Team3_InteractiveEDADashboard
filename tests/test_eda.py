"""Tests for loading, profiling and chart builders (run: pytest -q)."""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eda import charts, loader  # noqa: E402
from eda import profiling as prof  # noqa: E402


@pytest.fixture(scope="module", params=list(loader.SAMPLES))
def sample(request):
    df, _ = loader.load_sample(request.param)
    return request.param, df


# ------------------------------- loader --------------------------------- #
def test_samples_load_with_expected_shape():
    df, _ = loader.load_sample("student_performance")
    assert df.shape == (1012, 14)
    df, _ = loader.load_sample("retail_sales")
    assert df.shape == (1508, 12)
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])  # ISO dates recognised


@pytest.mark.parametrize("ext,payload", [
    (".csv", b"a,b\n1,2\n3,4\n"),
    (".csv", b"a;b\n1;2\n3;4\n"),
    (".tsv", b"a\tb\n1\t2\n3\t4\n"),
    (".json", b'[{"a":1,"b":2},{"a":3,"b":4}]'),
    (".json", b'{"a":1,"b":2}\n{"a":3,"b":4}\n'),
])
def test_parse_formats(ext, payload):
    df = loader.parse_bytes(payload, "data" + ext)
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_parse_excel_and_upload_roundtrip():
    buf = io.BytesIO()
    pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).to_excel(buf, index=False)
    contents = "data:application/vnd.ms-excel;base64," + base64.b64encode(buf.getvalue()).decode()
    df = loader.from_upload(contents, "book.xlsx")
    assert df.shape == (3, 2)


def test_latin1_and_duplicate_headers():
    df = loader.parse_bytes("name,name,city\nJosé,1,Zürich\n".encode("latin-1"), "x.csv")
    assert len(set(df.columns)) == 3 and df.columns[0] == "name"  # duplicate header made unique
    assert df.loc[0, "city"] == "Zürich"


@pytest.mark.parametrize("name,payload", [("a.pdf", b"x"), ("empty.csv", b""), ("hdr.csv", b"a,b\n")])
def test_bad_files_raise_friendly_errors(name, payload):
    with pytest.raises(loader.LoadError):
        loader.parse_bytes(payload, name)


def test_cache_eviction():
    ids = [loader.register(pd.DataFrame({"a": [1]}), f"d{i}") for i in range(loader.MAX_CACHE + 3)]
    assert loader.get_entry({"id": ids[0]}) is None
    assert loader.get_entry({"id": ids[-1]}) is not None
    assert loader.get_entry(None) is None


# ------------------------------ profiling ------------------------------- #
def test_overview_and_quality_counts(sample):
    key, df = sample
    ov = prof.overview_stats(df)
    assert ov["rows"] == len(df) and ov["missing_cells"] > 0 and ov["duplicates"] > 0
    extra, involved, preview = prof.duplicate_rows(df)
    assert extra == ov["duplicates"] and involved >= 2 * extra and "Row #" in preview.columns


def test_text_issues_detects_injected_problems():
    df, _ = loader.load_sample("student_performance")
    ti = prof.text_issues(df)
    row = ti[ti["Column"] == "gender"].iloc[0]
    assert row["Values with extra whitespace"] > 0 and row["Case/spelling variants"] >= 1


def test_numeric_summary_matches_pandas():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100.0]})
    s = prof.numeric_summary(df).iloc[0]
    assert s["Mean"] == pytest.approx(22.0) and s["Median"] == 3 and s["Outliers (IQR)"] == 1


def test_insights_are_generated_and_cautious(sample):
    _, df = sample
    items = prof.generate_insights(df)
    titles = {i["title"] for i in items}
    assert {"Dataset structure", "Missing values detected", "Duplicate rows detected"} <= titles
    assert all(i["level"] in {"warn", "info", "good"} for i in items)
    md = prof.insights_to_markdown(items, "x.csv")
    assert md.startswith("# EDA insights")


def test_id_like_columns_excluded_from_analysis():
    df = pd.DataFrame({"id": range(50), "v": np.random.default_rng(0).normal(size=50), "c": 1.0})
    assert prof.analysis_numeric(df) == ["v"]


def test_unhashable_cells_do_not_crash():
    df = pd.DataFrame({"a": [[1], [1], [2]], "b": [1, 1, 2]})
    assert prof.count_duplicates(df) == 1
    prof.column_profile(df)


# -------------------------------- charts -------------------------------- #
def test_all_chart_types_build(sample):
    key, df = sample
    t = prof.split_columns(df)
    num = prof.analysis_numeric(df)
    cat = [c for c in t.categorical if df[c].nunique() <= 12][0]

    fig, st = charts.bar_chart(df, cat)
    assert len(fig.data) == 1 and st
    fig, _ = charts.bar_chart(df, cat, num[0], "mean", color=None)
    fig, _ = charts.histogram_chart(df, num[0], 25)
    fig, _ = charts.histogram_chart(df, num[0], 25, color=cat)
    assert len(fig.data) >= 2
    fig, _ = charts.box_chart(df, num[0])
    fig, _ = charts.box_chart(df, num[0], cat, show_points=True)
    fig, st = charts.scatter_chart(df, num[0], num[1], cat, trendline=True)
    assert any(k == "Pearson r" for k, _ in st)
    fig, _ = charts.scatter_chart(df, num[0], num[1], num[2])
    fig, st = charts.heatmap_chart(df, num, "spearman")
    assert fig.data[0].z.shape == (len(num), len(num))
    charts.type_donut(df), charts.missing_bar(df), charts.missing_matrix(df)


def test_datetime_bar_is_chronological():
    df, _ = loader.load_sample("retail_sales")
    fig, _ = charts.bar_chart(df, "order_date", "revenue", "sum")
    xs = list(fig.data[0].x)
    assert xs == sorted(xs) and len(xs) == 12


def test_chart_errors_are_friendly():
    df, _ = loader.load_sample("student_performance")
    for call in (lambda: charts.bar_chart(df, None), lambda: charts.histogram_chart(df, "gender"),
                 lambda: charts.heatmap_chart(df, ["math_score"]), lambda: charts.scatter_chart(df, "math_score", None)):
        with pytest.raises(charts.ChartError):
            call()
    assert charts.empty_figure("x").layout.annotations[0].text == "x"


def test_no_missing_gives_clean_message():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert "No missing" in charts.missing_bar(df).layout.annotations[0].text


def test_unique_integer_measurements_are_not_treated_as_ids():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"seq": range(50), "emp_id": rng.permutation(50) + 7000,
                       "income": rng.permutation(50) * 37 + 20000, "age": rng.integers(18, 70, 50)})
    assert prof.is_id_like(df["seq"], "seq") and prof.is_id_like(df["emp_id"], "emp_id")
    assert not prof.is_id_like(df["income"], "income")
    assert set(prof.analysis_numeric(df)) == {"income", "age"}
