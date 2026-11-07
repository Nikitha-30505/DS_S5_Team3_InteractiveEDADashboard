# Interactive EDA Dashboard

Data Science (24AD3107) · KL University · A.Y. 2026–2027
Y. Revanth (2420030605) · G. Rithwik (2420030415) · A. Nikitha (2420030505) · Guide: Dr. N. Sunanda

A web dashboard (Python, Dash, Plotly, pandas) that turns a raw tabular dataset into a profile, a data-quality
report and five exploratory charts, following the workflow in the project paper:
**Upload → Profiling → Quality Inspection → Chart Selection → Visualization → Interactive Interpretation.**

## Run

```bash
pip install -r requirements.txt
python generate_samples.py      # optional: regenerates the sample CSVs
python app.py                   # open http://127.0.0.1:8050
```
Environment variables: `HOST`, `PORT`, `DEBUG=1`. For deployment use a single-process server
(`gunicorn app:server --threads 8`), because uploaded data is cached in server memory.

## Features
| Page | What it does |
|---|---|
| 01 Overview | Rows, columns, types, completeness, memory, column profile, data preview |
| 02 Data Quality | Missing values (bar + pattern map), duplicate rows, whitespace/case inconsistencies |
| 03 Visualize | Bar chart, histogram, box plot, scatter plot (with trend line), correlation heatmap (Pearson/Spearman) |
| 04 Statistics | Mean, std, quartiles, skew, kurtosis, IQR outliers; categorical summaries; CSV export |
| 05 Insights | Rule-based findings (missing, duplicates, skew, outliers, correlations); Markdown export |

Uploads: CSV, TSV, Excel (.xlsx), JSON up to 25 MB; delimiter and encoding are auto-detected.
Charts export to PNG from the plot toolbar.

Design principle from the paper: the dashboard **reports evidence and never silently changes data**. Missing
values and duplicates are flagged, not removed; correlation is described as association, not causation.

## Structure
```
app.py            layout + callbacks
eda/loader.py     parsing and server-side dataset cache
eda/profiling.py  profiling, quality checks, insights
eda/charts.py     Plotly figure builders
eda/components.py UI building blocks
assets/style.css  design system
sample_data/      two synthetic datasets with realistic quality issues
tests/            pytest suite (unit + HTTP-level callback tests)
```
Run tests with `python -m pytest -q`.

## Notes
- The slides mention Streamlit; this implementation follows the final paper (Dash).
- Large datasets: scatter plots draw at most 20,000 sampled points (statistics use all rows); histograms are binned server-side.
- Measured performance/usability results are not included; evaluate with real users and datasets.
