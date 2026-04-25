"""
valuation.py
============
Determines whether each player is over- or undervalued relative to their
salary, using the performance metrics produced by analysis.py.

Approach
--------
1.  Estimate market-rate salary from performance using a regression model
    trained on the league-wide frame (WAR → salary, with Statcast corrections).
2.  Compute Value Gap = estimated_salary - actual_salary
    +  → player is underpaid  (team is getting a bargain)
    −  → player is overpaid   (team is paying a premium)
3.  Assign a Value Label: ELITE_VALUE / GOOD_VALUE / FAIR / OVERPAID / HIGHLY_OVERPAID
4.  Produce a sortable Valuation Leaderboard.

Notes on salary data
--------------------
The Lahman salary table lags by ~1 year. For the most recent season you may
need to supplement with a manually supplied CSV (path configurable below).
Expected CSV schema: name, salary (USD annual), year
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Path to an optional manually curated salary CSV
# Set to None to skip; must have columns: name, salary, year
MANUAL_SALARY_CSV: Path | None = Path("data/salaries_manual.csv")

# Salary model features — all must exist in the league frame
# We prefer Statcast-era metrics to anchor on true talent
SALARY_MODEL_FEATURES = [
    "WAR",
    "wRC_plus",
    "xwOBA",
    "barrel_pct",
    "craft_range_runs_saved",  # included if available
]

# Value tier thresholds ($ millions gap per season)
# Positive gap = underpaid, negative gap = overpaid
VALUE_TIERS = [
    ("ELITE VALUE",    10.0,  float("inf")),
    ("GOOD VALUE",      3.0,  10.0),
    ("FAIR",           -3.0,   3.0),
    ("OVERPAID",      -10.0,  -3.0),
    ("HIGHLY OVERPAID", float("-inf"), -10.0),
]

# Dollars per WAR — market rate escalates each year
# Keys are season years; add future years as needed
DOLLARS_PER_WAR: dict[int, float] = {
    2019: 8.0,
    2020: 8.5,
    2021: 9.0,
    2022: 9.5,
    2023: 9.0,
    2024: 9.5,
    2025: 10.0,
}
DEFAULT_DOLLARS_PER_WAR = 9.5


# ─────────────────────────────────────────────────────────────
#  SALARY LOADING
# ─────────────────────────────────────────────────────────────

def load_manual_salaries(year: int, path: Path | None = MANUAL_SALARY_CSV) -> pd.DataFrame:
    """
    Load a manually curated salary CSV if it exists.
    Expected columns: name, salary (annual USD), year

    Returns empty DataFrame if not found or wrong year.
    """
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "year" in df.columns:
            df = df[df["year"] == year]
        df["salary"]   = pd.to_numeric(df["salary"],   errors="coerce")
        df["salary_M"] = (df["salary"] / 1_000_000).round(2)
        return df[["name", "salary", "salary_M"]].dropna(subset=["name","salary"])
    except Exception as exc:
        print(f"[valuation] Failed to load manual salaries: {exc}")
        return pd.DataFrame()


def patch_salaries(league_frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Supplement Lahman salary data with manual CSV where available.
    Manual data takes priority over Lahman for matching player names.

    Parameters
    ----------
    league_frame : pd.DataFrame
    year         : int

    Returns
    -------
    pd.DataFrame  with salary / salary_M columns populated where possible
    """
    manual = load_manual_salaries(year)
    if manual.empty:
        return league_frame

    if "name" not in league_frame.columns:
        return league_frame

    # Build lookup dict from manual data
    sal_lookup = dict(zip(manual["name"].str.lower(), manual["salary_M"]))

    def _fill_salary(row):
        if pd.notna(row.get("salary_M")) and row["salary_M"] > 0:
            return row["salary_M"]
        return sal_lookup.get(str(row.get("name","")).lower(), np.nan)

    league_frame["salary_M"] = league_frame.apply(_fill_salary, axis=1)
    league_frame["salary"]   = league_frame["salary_M"] * 1_000_000
    return league_frame


# ─────────────────────────────────────────────────────────────
#  MARKET RATE ESTIMATION
# ─────────────────────────────────────────────────────────────

def _war_market_rate(war: float, year: int) -> float:
    """Simple $/WAR baseline estimate."""
    dpw = DOLLARS_PER_WAR.get(year, DEFAULT_DOLLARS_PER_WAR)
    # Pre-arb (~first 3 years) players earn ~$750k regardless of WAR
    # We don't have service time so we use salary floor proxy
    return max(0.72, war * dpw)   # $720k MLB min


def build_salary_model(
    league_frame: pd.DataFrame,
    year: int,
) -> Pipeline | None:
    """
    Train a robust regression (HuberRegressor) to estimate market-rate
    salary from performance features.

    Huber regression is used because salary distributions are heavily
    right-skewed and contain outliers (supermax contracts).

    Parameters
    ----------
    league_frame : pd.DataFrame  full league frame with salary_M populated
    year         : int

    Returns
    -------
    sklearn Pipeline  or None if insufficient training data
    """
    df = league_frame.copy()

    # Filter to players with salary data and min PA
    df = df[df["salary_M"].notna() & (df["salary_M"] > 0)]
    if "PA" in df.columns:
        df = df[df["PA"] >= 100]

    # Only use features that exist
    features = [f for f in SALARY_MODEL_FEATURES if f in df.columns]
    if "WAR" not in features or len(df) < 30:
        print(f"[valuation] Insufficient data for salary model ({len(df)} rows with salary). "
              "Falling back to $/WAR heuristic.")
        return None

    train = df[features + ["salary_M"]].dropna()
    if len(train) < 20:
        return None

    X = train[features].values
    y = np.log1p(train["salary_M"].values)   # log-scale for right-skewed salary

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    HuberRegressor(epsilon=1.5, max_iter=300)),
    ])
    model.fit(X, y)
    train_r2 = model.score(X, y)
    print(f"[valuation] Salary model trained on {len(train)} players | "
          f"features: {features} | train R²: {train_r2:.3f}")

    # Store feature list on the pipeline for inference
    model._features = features
    return model


def estimate_market_salary(
    row: pd.Series,
    model: Pipeline | None,
    year: int,
) -> float:
    """
    Estimate what the market would pay this player.

    Falls back to $/WAR heuristic if model is unavailable.

    Parameters
    ----------
    row   : pd.Series  player row
    model : Pipeline   from build_salary_model(), or None
    year  : int

    Returns
    -------
    float  estimated salary in $M
    """
    if model is not None:
        features = getattr(model, "_features", SALARY_MODEL_FEATURES)
        vals = [row.get(f, np.nan) for f in features]
        if all(pd.notna(v) for v in vals):
            log_pred = model.predict([vals])[0]
            return round(np.expm1(log_pred), 2)

    # Heuristic fallback
    war = row.get("WAR", np.nan)
    if pd.notna(war):
        return round(_war_market_rate(float(war), year), 2)

    return np.nan


# ─────────────────────────────────────────────────────────────
#  VALUE GAP & LABELS
# ─────────────────────────────────────────────────────────────

def compute_value_gap(estimated_M: float, actual_M: float) -> float:
    """
    Value Gap = estimated_salary - actual_salary ($M)
    Positive → underpaid (value for the team)
    Negative → overpaid
    """
    if pd.isna(estimated_M) or pd.isna(actual_M):
        return np.nan
    return round(estimated_M - actual_M, 2)


def assign_value_label(gap_M: float) -> str:
    """Assign a categorical value label from the gap in $M."""
    if pd.isna(gap_M):
        return "NO SALARY DATA"
    for label, lo, hi in VALUE_TIERS:
        if lo <= gap_M < hi:
            return label
    return "UNKNOWN"


def value_score(gap_M: float, max_gap: float = 20.0) -> float:
    """
    Normalise gap to 0-100 value score.
    50 = fair value, 100 = extreme bargain, 0 = extreme overpay.
    """
    if pd.isna(gap_M):
        return np.nan
    clamped = max(-max_gap, min(max_gap, gap_M))
    return round((clamped / max_gap + 1) / 2 * 100, 1)


# ─────────────────────────────────────────────────────────────
#  FULL VALUATION PIPELINE
# ─────────────────────────────────────────────────────────────

def run_valuation(
    league_frame: pd.DataFrame,
    year: int,
    min_pa: int = 100,
) -> pd.DataFrame:
    """
    Full valuation pipeline.

    Steps:
      1. Patch salaries with manual CSV if available
      2. Train salary model on players with known salaries
      3. Estimate market salary for every player
      4. Compute value gap and labels
      5. Return enriched frame sorted by value gap (best bargains first)

    Parameters
    ----------
    league_frame : pd.DataFrame  from analysis.analyse_league()
    year         : int
    min_pa       : int           minimum PA filter

    Returns
    -------
    pd.DataFrame  with added columns:
        estimated_salary_M, actual_salary_M, value_gap_M,
        value_label, value_score
    """
    df = league_frame.copy()

    # Patch with manual salary data
    df = patch_salaries(df, year)

    # Alias salary column
    if "salary_M" not in df.columns and "salary" in df.columns:
        df["salary_M"] = (df["salary"] / 1_000_000).round(2)

    # Filter by PA
    if "PA" in df.columns:
        qualified = df[df["PA"].fillna(0) >= min_pa].copy()
    else:
        qualified = df.copy()

    # Train salary model
    model = build_salary_model(qualified, year)

    # Estimate market salary for every player
    qualified["estimated_salary_M"] = qualified.apply(
        lambda r: estimate_market_salary(r, model, year), axis=1
    )

    # Value gap
    qualified["actual_salary_M"] = qualified.get("salary_M", np.nan)
    qualified["value_gap_M"] = qualified.apply(
        lambda r: compute_value_gap(
            r.get("estimated_salary_M", np.nan),
            r.get("actual_salary_M",    np.nan),
        ),
        axis=1,
    )
    qualified["value_label"] = qualified["value_gap_M"].apply(assign_value_label)
    qualified["value_score"] = qualified["value_gap_M"].apply(value_score)

    return qualified


# ─────────────────────────────────────────────────────────────
#  LEADERBOARDS
# ─────────────────────────────────────────────────────────────

LEADERBOARD_COLS = [
    "name", "team", "age", "PA",
    "wRC_plus", "xwOBA", "WAR",
    "craft_range_runs_saved",
    "hitting_score", "fielding_score", "total_score",
    "actual_salary_M", "estimated_salary_M",
    "value_gap_M", "value_label", "value_score",
]


def valuation_leaderboard(
    valued_frame: pd.DataFrame,
    top_n: int | None = None,
    label_filter: str | None = None,
    sort_by: str = "value_gap_M",
) -> pd.DataFrame:
    """
    Return a formatted leaderboard from the valued frame.

    Parameters
    ----------
    valued_frame  : pd.DataFrame  from run_valuation()
    top_n         : int | None    limit to top N rows
    label_filter  : str | None    e.g. 'ELITE VALUE' or 'HIGHLY OVERPAID'
    sort_by       : str           column to sort by (default: value_gap_M)

    Returns
    -------
    pd.DataFrame  leaderboard
    """
    df = valued_frame.copy()

    if label_filter:
        df = df[df["value_label"] == label_filter.upper()]

    avail_cols = [c for c in LEADERBOARD_COLS if c in df.columns]
    df = df[avail_cols]

    ascending = sort_by in ("actual_salary_M",)  # lower salary sorted first for bargain views
    df = df.sort_values(sort_by, ascending=ascending, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    if top_n:
        df = df.head(top_n)

    # Round display columns
    for col in ("actual_salary_M", "estimated_salary_M", "value_gap_M"):
        if col in df.columns:
            df[col] = df[col].round(1)
    for col in ("wRC_plus", "xwOBA", "WAR", "hitting_score", "total_score",
                "craft_range_runs_saved"):
        if col in df.columns:
            df[col] = df[col].round(2)

    return df


def best_bargains(valued_frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N most undervalued players (largest positive gap)."""
    return valuation_leaderboard(valued_frame, top_n=n, sort_by="value_gap_M")


def biggest_overpays(valued_frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N most overpaid players (largest negative gap)."""
    df = valued_frame.sort_values("value_gap_M", ascending=True, na_position="last").head(n)
    avail_cols = [c for c in LEADERBOARD_COLS if c in df.columns]
    df = df[avail_cols].reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def tier_summary(valued_frame: pd.DataFrame) -> pd.DataFrame:
    """Count and average stats per value tier."""
    if "value_label" not in valued_frame.columns:
        return pd.DataFrame()

    agg = (
        valued_frame
        .groupby("value_label", observed=True)
        .agg(
            count=("value_label", "count"),
            avg_salary_M=("actual_salary_M", "mean"),
            avg_estimated_M=("estimated_salary_M", "mean"),
            avg_gap_M=("value_gap_M", "mean"),
            avg_WAR=("WAR", "mean") if "WAR" in valued_frame.columns else ("value_gap_M", "count"),
            avg_wRC_plus=("wRC_plus", "mean") if "wRC_plus" in valued_frame.columns else ("value_gap_M", "count"),
            avg_total_score=("total_score", "mean") if "total_score" in valued_frame.columns else ("value_gap_M", "count"),
        )
        .round(2)
        .reset_index()
    )

    # Sort by tier logic
    tier_order = {t[0]: i for i, t in enumerate(VALUE_TIERS)}
    agg["_order"] = agg["value_label"].map(tier_order).fillna(99)
    agg = agg.sort_values("_order").drop(columns="_order")
    return agg
