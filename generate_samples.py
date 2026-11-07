"""Generate the two synthetic sample datasets used by the dashboard.

The data is synthetic (fixed random seed) and deliberately includes realistic
quality issues - missing values, duplicate rows, skewed values, outliers and
inconsistent text - so every part of the dashboard has something to show.

Usage:  python generate_samples.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "sample_data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


def inject_missing(df: pd.DataFrame, fractions: dict[str, float]) -> pd.DataFrame:
    for col, frac in fractions.items():
        df.loc[rng.random(len(df)) < frac, col] = np.nan
    return df


def student_performance(n: int = 1000) -> pd.DataFrame:
    gender = rng.choice(["Female", "Male"], n, p=[0.51, 0.49])
    school = rng.choice(["Public", "Private"], n, p=[0.72, 0.28])
    grade = rng.choice([9, 10, 11, 12], n)
    parent = rng.choice(["High School", "Some College", "Bachelor's", "Master's", "Doctorate"], n,
                        p=[0.28, 0.17, 0.34, 0.16, 0.05])
    internet = rng.choice(["Yes", "No"], n, p=[0.88, 0.12])
    extra = rng.choice(["Yes", "No"], n, p=[0.55, 0.45])
    study = np.clip(rng.gamma(4.0, 1.0, n), 0.5, 16).round(1)
    attendance = np.clip(rng.beta(8, 1.6, n) * 100, 40, 100).round(1)
    sleep = np.clip(rng.normal(7.1, 1.0, n), 3.5, 10).round(1)
    previous = np.clip(rng.normal(68, 12, n), 20, 100).round(0)

    math = np.clip(
        18 + 0.45 * previous + 2.6 * study + 0.12 * attendance + 1.5 * (sleep - 7)
        + 2 * (internet == "Yes") + 1.2 * (school == "Private") + rng.normal(0, 6, n), 0, 100,
    ).round(1)
    reading = np.clip(0.55 * math + 0.2 * previous + 14 + rng.normal(0, 7, n), 0, 100).round(1)
    science = np.clip(0.6 * math + 0.2 * study + 12 + rng.normal(0, 7, n), 0, 100).round(1)

    df = pd.DataFrame({
        "student_id": [f"S{1000 + i}" for i in range(n)],
        "gender": gender, "grade_level": grade, "school_type": school,
        "parental_education": parent, "internet_access": internet, "extracurricular": extra,
        "study_hours_per_week": study, "attendance_pct": attendance, "sleep_hours": sleep,
        "previous_score": previous, "math_score": math, "reading_score": reading, "science_score": science,
    })
    df = inject_missing(df, {"attendance_pct": 0.06, "sleep_hours": 0.04,
                             "parental_education": 0.03, "previous_score": 0.02})
    # inconsistent text
    df["gender"] = df["gender"].astype(object)
    df.loc[rng.choice(n, 8, replace=False), "gender"] = "female"
    df.loc[rng.choice(n, 5, replace=False), "gender"] = "Male "
    # duplicate rows
    return pd.concat([df, df.sample(12, random_state=3)], ignore_index=True)


def retail_sales(n: int = 1500) -> pd.DataFrame:
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    region = rng.choice(["North", "South", "East", "West", "Central"], n, p=[0.22, 0.20, 0.24, 0.20, 0.14])
    category = rng.choice(["Electronics", "Furniture", "Clothing", "Groceries", "Sports", "Beauty"], n,
                          p=[0.20, 0.12, 0.22, 0.26, 0.10, 0.10])
    segment = rng.choice(["Consumer", "Corporate", "Small Business"], n, p=[0.55, 0.25, 0.20])
    base = {"Electronics": 240, "Furniture": 180, "Clothing": 45, "Groceries": 12, "Sports": 60, "Beauty": 28}
    price = np.array([base[c] for c in category]) * rng.lognormal(0, 0.35, n)
    units = rng.poisson(3, n) + 1
    units[rng.choice(n, 6, replace=False)] = rng.integers(60, 90, 6)  # bulk orders (outliers)
    discount = rng.choice([0, 5, 10, 15, 20, 25], n, p=[0.35, 0.20, 0.20, 0.12, 0.08, 0.05])
    shipping = np.clip(rng.poisson(4, n) + 1, 1, 14)
    revenue = units * price * (1 - discount / 100)
    rating = np.clip(np.round(4.4 - 0.18 * shipping + rng.normal(0, 0.9, n)), 1, 5)
    returned = np.where(rng.random(n) < (0.04 + 0.02 * (discount >= 20) + 0.02 * (shipping > 7)), "Yes", "No")

    df = pd.DataFrame({
        "order_id": [f"ORD-{10000 + i}" for i in range(n)],
        "order_date": pd.Series(dates).dt.strftime("%Y-%m-%d"),
        "region": region, "category": category, "customer_segment": segment,
        "units_sold": units, "unit_price": price.round(2), "discount_pct": discount.astype(float),
        "revenue": revenue.round(2), "shipping_days": shipping, "rating": rating, "returned": returned,
    })
    df = inject_missing(df, {"rating": 0.09, "discount_pct": 0.04, "customer_segment": 0.02})
    return pd.concat([df, df.sample(8, random_state=5)], ignore_index=True)


if __name__ == "__main__":
    for name, frame in {"student_performance": student_performance(), "retail_sales": retail_sales()}.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
