"""Dataset loading and an in-memory, server-side dataset cache.

Uploaded data is parsed once and kept on the server; the browser only holds a
small dataset id. The loader never cleans or imputes values - it only tidies
column names and recognises ISO-style date columns so that they can be charted
sensibly. Source records are otherwise left untouched.
"""
from __future__ import annotations

import base64
import csv
import io
import re
import uuid
import warnings
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import pandas as pd

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

# key -> (label, filename)
SAMPLES: dict[str, tuple[str, str]] = {
    "student_performance": ("Student Performance (1,012 rows)", "student_performance.csv"),
    "retail_sales": ("Retail Sales (1,508 rows)", "retail_sales.csv"),
}

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".json")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_CACHE = 8

_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$")

_cache: "OrderedDict[str, dict]" = OrderedDict()
_lock = Lock()


class LoadError(ValueError):
    """Raised with a user-friendly message when a dataset cannot be loaded."""


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _read_delimited(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            head = raw[:20000].decode(encoding, errors="replace")
            try:
                sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
            except csv.Error:
                sep = ","
            return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise LoadError("The file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise LoadError(f"The file could not be parsed as a table: {exc}") from exc
    raise LoadError("The file encoding is not supported. Please save it as UTF-8.")


def _tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    names, seen = [], {}
    for i, col in enumerate(df.columns):
        name = str(col).strip() or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df = df.copy()
    df.columns = names
    return df


def _infer_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns that clearly hold ISO-style dates to datetime."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        non_null = s.dropna().astype(str)
        if non_null.empty:
            continue
        if non_null.head(200).str.match(_DATE_RE).mean() < 0.95:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if parsed.notna().sum() >= 0.9 * s.notna().sum():
            df[col] = parsed
    return df


def parse_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoadError(
            f"Unsupported file type '{ext or 'unknown'}'. "
            f"Please upload one of: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    try:
        if ext in (".csv", ".tsv", ".txt"):
            df = _read_delimited(raw)
        elif ext in (".xlsx", ".xlsm"):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            try:
                df = pd.read_json(io.BytesIO(raw))
            except ValueError:
                df = pd.read_json(io.BytesIO(raw), lines=True)
    except LoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any reader problem to the user
        raise LoadError(f"Could not read '{filename}': {exc}") from exc

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise LoadError("The file does not contain any rows or columns.")
    return _infer_dates(_tidy_columns(df))


def from_upload(contents: str, filename: str) -> pd.DataFrame:
    """Decode a ``dcc.Upload`` data URL into a DataFrame."""
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
    except Exception as exc:  # noqa: BLE001
        raise LoadError("The uploaded file could not be decoded.") from exc
    if len(raw) > MAX_UPLOAD_BYTES:
        raise LoadError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    return parse_bytes(raw, filename or "upload.csv")


def load_sample(key: str) -> tuple[pd.DataFrame, str]:
    if key not in SAMPLES:
        raise LoadError("Unknown sample dataset.")
    filename = SAMPLES[key][1]
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise LoadError(f"Sample file {filename} is missing. Run generate_samples.py.")
    return parse_bytes(path.read_bytes(), filename), filename


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def register(df: pd.DataFrame, name: str) -> str:
    ds_id = uuid.uuid4().hex[:12]
    with _lock:
        _cache[ds_id] = {"df": df, "name": name}
        while len(_cache) > MAX_CACHE:
            _cache.popitem(last=False)
    return ds_id


def get_entry(store: dict | None) -> dict | None:
    if not store or "id" not in store:
        return None
    with _lock:
        entry = _cache.get(store["id"])
        if entry is not None:
            _cache.move_to_end(store["id"])
        return entry


def get_df(store: dict | None) -> pd.DataFrame | None:
    entry = get_entry(store)
    return None if entry is None else entry["df"]
