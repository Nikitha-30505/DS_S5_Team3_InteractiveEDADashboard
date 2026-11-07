"""End-to-end callback tests: POST to Dash's real /_dash-update-component endpoint."""
import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as dash_app  # noqa: E402

client = dash_app.app.server.test_client()
DEPS = client.get("/_dash-dependencies").get_json()


def _find(output_prop: str):
    for dep in DEPS:
        if output_prop in dep["output"]:
            return dep
    raise KeyError(output_prop)


def call(output_prop: str, values: dict, changed: str | None = None):
    """Invoke the callback that produces `output_prop` with {"id.prop": value} inputs/state."""
    dep = _find(output_prop)
    out = dep["output"]
    ids = out.strip(".").split("...") if out.startswith("..") else [out]
    outputs = [{"id": o.rsplit(".", 1)[0], "property": o.rsplit(".", 1)[1]} for o in ids]
    body = {
        "output": out,
        "outputs": outputs if out.startswith("..") else outputs[0],
        "inputs": [{"id": i["id"], "property": i["property"], "value": values.get(f"{i['id']}.{i['property']}")}
                   for i in dep["inputs"]],
        "state": [{"id": s["id"], "property": s["property"], "value": values.get(f"{s['id']}.{s['property']}")}
                  for s in dep.get("state", [])],
        "changedPropIds": [changed or f"{dep['inputs'][0]['id']}.{dep['inputs'][0]['property']}"],
    }
    resp = client.post("/_dash-update-component", json=body)
    assert resp.status_code == 200, resp.get_data(as_text=True)[:500]
    return resp.get_json()["response"]


@pytest.fixture(scope="module")
def store():
    r = call("dataset-store.data", {"sample-select.value": "student_performance"})
    assert "Loaded student_performance.csv" in str(r["load-msg"]["children"])
    return r["dataset-store"]["data"]


def test_layout_serves():
    layout = client.get("/_dash-layout").get_json()
    assert "app-shell" in str(layout)
    assert client.get("/assets/style.css").status_code == 200


def test_upload_roundtrip_and_errors():
    csv = "a,b\n1,x\n2,y\n3,x\n"
    ok = call("dataset-store.data", {
        "upload.contents": "data:text/csv;base64," + base64.b64encode(csv.encode()).decode(),
        "upload.filename": "mine.csv"}, changed="upload.contents")
    assert ok["dataset-store"]["data"]["name"] == "mine.csv"
    bad = call("dataset-store.data", {"upload.contents": "data:text/plain;base64,eA==", "upload.filename": "bad.pdf"},
               changed="upload.contents")
    assert "Unsupported file type" in str(bad["load-msg"]["children"])


def test_navigation():
    r = call("page-visualize.style", {"nav-visualize.n_clicks": 1}, changed="nav-visualize.n_clicks")
    assert r["page-visualize"]["style"]["display"] == "flex" and r["page-overview"]["style"]["display"] == "none"
    assert r["nav-visualize"]["className"] == "nav-btn active" and r["page-title"]["children"] == "Visualize"


def test_pages_render(store):
    s = {"dataset-store.data": store, "preview-rows.value": 25}
    ov = call("ov-kpis.children", s)
    assert "1,012" in str(ov["ov-kpis"]["children"]) and ov["ov-donut"]["figure"]["data"]
    q = call("q-kpis.children", s)
    assert "Duplicate rows" in str(q["q-kpis"]["children"]) and "Row #" in str(q["q-dups"]["children"])
    st = call("st-num.children", s)
    assert "math_score" in str(st["st-num"]["children"])
    ins = call("in-list.children", s)
    assert "Duplicate rows detected" in str(ins["in-list"]["children"])
    assert "student_performance.csv" in str(call("dataset-chip.children", s)["dataset-chip"]["children"])


@pytest.mark.parametrize("chart", ["bar", "hist", "box", "scatter", "heatmap"])
def test_visualize_flow(store, chart):
    cfg = call("ctl-x.options", {"dataset-store.data": store, "chart-type.value": chart})
    assert cfg["ctl-heat"]["value"], "heatmap defaults should be populated"
    vals = {
        "dataset-store.data": store, "chart-type.value": chart,
        "ctl-x.value": cfg["ctl-x"]["value"], "ctl-y.value": cfg["ctl-y"]["value"],
        "ctl-color.value": cfg["ctl-color"]["value"], "ctl-bins.value": 30, "ctl-agg.value": "count",
        "ctl-topn.value": 15, "ctl-points.value": [], "ctl-trend.value": ["on"], "ctl-annot.value": ["on"],
        "ctl-heat.value": cfg["ctl-heat"]["value"], "ctl-method.value": "pearson",
    }
    r = call("main-chart.figure", vals)
    assert r["main-chart"]["figure"]["data"], f"{chart}: figure has no traces"
    assert r["chart-stats"]["children"], f"{chart}: no stat chips"
    # hidden controls follow the chart type
    shown = {w: cfg[f"wrap-{w}"]["style"].get("display") != "none" for w in dash_app.WRAPPERS}
    assert shown == {w: (w in dash_app.VISIBLE[chart]) for w in dash_app.WRAPPERS}


def test_invalid_selection_is_handled_not_crashing(store):
    vals = {"dataset-store.data": store, "chart-type.value": "hist", "ctl-x.value": "gender", "ctl-bins.value": 20}
    fig = call("main-chart.figure", vals)["main-chart"]["figure"]
    assert "no numeric values" in str(fig["layout"]["annotations"])


def test_downloads(store):
    p = call("download-profile.data", {"btn-profile.n_clicks": 1, "dataset-store.data": store})
    assert "student_performance_column_profile.csv" == p["download-profile"]["data"]["filename"]
    i = call("download-insights.data", {"btn-insights.n_clicks": 1, "dataset-store.data": store})
    assert "# EDA insights" in i["download-insights"]["data"]["content"]


def test_expired_cache_shows_empty_states():
    ghost = {"id": "doesnotexist", "name": "x"}
    ov = call("ov-kpis.children", {"dataset-store.data": ghost})
    assert "No dataset loaded" in str(ov["ov-columns"]["children"])
