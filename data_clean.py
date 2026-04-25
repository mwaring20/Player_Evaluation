"""
data_clean.py
=============
Takes raw DataFrames from data_pull and produces a single, analysis-ready
master DataFrame. Responsibilities:

  1. Normalise column names across sources
  2. Cast columns to correct dtypes
  3. Merge all sources on player_id / name
  4. Join salary data
  5. Derive helper columns (BB% as float, age buckets, etc.)
  6. Flag data quality issues (low PA, missing salary, etc.)

All functions are pure: they take DataFrames in and return DataFrames out.
"""

import re
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  COLUMN RENAME MAPS
# ─────────────────────────────────────────────────────────────

# FanGraphs batting columns → standard names
FG_RENAME = {
    "Name":      "name",
    "Team":      "team",
    "Age":       "age",
    "G":         "G",
    "PA":        "PA",
    "AB":        "AB",
    "H":         "H",
    "HR":        "HR",
    "R":         "R",
    "RBI":       "RBI",
    "SB":        "SB",
    "CS":        "CS",
    "AVG":       "AVG",
    "OBP":       "OBP",
    "SLG":       "SLG",
    "OPS":       "OPS",
    "wOBA":      "wOBA",
    "wRC+":      "wRC_plus",
    "BB%":       "BB_pct",
    "K%":        "K_pct",
    "ISO":       "ISO",
    "BABIP":     "BABIP",
    "WAR":       "WAR",
    "Off":       "Off",
    "Def":       "Def",
    "Spd":       "Spd",
    "IDfg":      "fg_id",
}

# Statcast EV/barrel columns → standard names
EV_RENAME = {
    "player_id":              "mlbam_id",
    "batter":                 "mlbam_id",
    "last_name":              "last_name",
    "first_name":             "first_name",
    "avg_hit_speed":          "avg_EV",
    "max_hit_speed":          "max_EV",
    "avg_hit_angle":          "avg_LA",
    "anglesweetspotpercent":  "hard_hit_pct",
    "brl_percent":            "barrel_pct",
    "brl_pa":                 "barrels_per_pa",
}

# Statcast expected stats columns → standard names
XS_RENAME = {
    "player_id":   "mlbam_id",
    "batter":      "mlbam_id",
    "last_name":   "last_name",
    "first_name":  "first_name",
    "est_ba":      "xBA",
    "est_slg":     "xSLG",
    "est_woba":    "xwOBA",
    "ba":          "BA_actual",
    "slg":         "SLG_actual",
    "woba":        "wOBA_actual",
}

# Statcast sprint speed → standard names
SPEED_RENAME = {
    "player_id":       "mlbam_id",
    "batter":          "mlbam_id",
    "last_name":       "last_name",
    "first_name":      "first_name",
    "sprint_speed":    "sprint_speed",
    "competitive_runs":"competitive_runs",
    "percent_rank":    "speed_pct_rank",
}


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _safe_rename(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename only columns that exist; ignore the rest."""
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing)


def _pct_to_float(series: pd.Series) -> pd.Series:
    """Convert '12.3%' strings or 0-1 floats to 0-100 floats."""
    def convert(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            val = val.replace("%", "").strip()
            return float(val)
        if isinstance(val, (int, float)):
            # FanGraphs stores as 0–1 fractions
            return float(val) * 100 if float(val) <= 1.0 else float(val)
        return np.nan
    return series.apply(convert)


def _id_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ─────────────────────────────────────────────────────────────
#  INDIVIDUAL CLEANERS
# ─────────────────────────────────────────────────────────────

def clean_fg_batting(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise FanGraphs batting leaderboard."""
    if df.empty:
        return df
    df = _safe_rename(df, FG_RENAME)

    # pct columns to numeric 0–100
    for col in ("BB_pct", "K_pct"):
        if col in df.columns:
            df[col] = _pct_to_float(df[col])

    # numeric coercions
    numeric_cols = [
        "PA","AB","G","H","HR","R","RBI","SB","CS",
        "AVG","OBP","SLG","OPS","wOBA","wRC_plus",
        "ISO","BABIP","WAR","Off","Def","Spd","age",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fg_id as string
    if "fg_id" in df.columns:
        df["fg_id"] = df["fg_id"].astype(str).str.strip()

    return df


def clean_statcast_ev(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Statcast exit-velo / barrel leaderboard."""
    if df.empty:
        return df
    # unify id column name before rename
    if "batter" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"batter": "player_id"})
    df = _safe_rename(df, EV_RENAME)

    # build full name if split
    if "last_name" in df.columns and "first_name" in df.columns:
        df["name"] = (df["first_name"].str.title() + " " +
                      df["last_name"].str.title())

    numeric_cols = ["avg_EV","max_EV","avg_LA","hard_hit_pct","barrel_pct","barrels_per_pa"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_statcast_expected(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Statcast expected stats leaderboard."""
    if df.empty:
        return df
    if "batter" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"batter": "player_id"})
    df = _safe_rename(df, XS_RENAME)

    if "last_name" in df.columns and "first_name" in df.columns:
        df["name"] = (df["first_name"].str.title() + " " +
                      df["last_name"].str.title())

    for col in ("xBA","xSLG","xwOBA","BA_actual","SLG_actual","wOBA_actual"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # luck differentials
    if "xBA" in df.columns and "BA_actual" in df.columns:
        df["xBA_diff"] = df["xBA"] - df["BA_actual"]
    if "xwOBA" in df.columns and "wOBA_actual" in df.columns:
        df["xwOBA_diff"] = df["xwOBA"] - df["wOBA_actual"]

    return df


def clean_statcast_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Statcast percentile ranks."""
    if df.empty:
        return df
    if "batter" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"batter": "player_id"})
    if "mlbam_id" not in df.columns and "player_id" in df.columns:
        df = df.rename(columns={"player_id": "mlbam_id"})

    # all numeric percentile columns
    skip = {"mlbam_id","player_name","year","_source"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_statcast_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Statcast sprint speed leaderboard."""
    if df.empty:
        return df
    if "batter" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"batter": "player_id"})
    df = _safe_rename(df, SPEED_RENAME)

    if "last_name" in df.columns and "first_name" in df.columns:
        df["name"] = (df["first_name"].str.title() + " " +
                      df["last_name"].str.title())

    for col in ("sprint_speed","competitive_runs","speed_pct_rank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # convert 0-1 rank to 0-100
    if "speed_pct_rank" in df.columns:
        if df["speed_pct_rank"].max() <= 1.0:
            df["speed_pct_rank"] = (df["speed_pct_rank"] * 100).round(0)

    return df


def clean_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Lahman salary table."""
    if df.empty:
        return df
    rename = {"playerID": "lahman_id", "salary": "salary", "yearID": "year"}
    df = _safe_rename(df, rename)
    if "salary" in df.columns:
        df["salary"]   = pd.to_numeric(df["salary"], errors="coerce")
        df["salary_M"] = (df["salary"] / 1_000_000).round(2)
    return df


# ─────────────────────────────────────────────────────────────
#  MERGE PIPELINE
# ─────────────────────────────────────────────────────────────

def _merge_statcast_onto_fg(
    fg: pd.DataFrame,
    statcast: pd.DataFrame,
    mlbam_id: int,
    suffix: str,
) -> pd.DataFrame:
    """
    Merge a single Statcast leaderboard onto the FG master frame.
    Uses mlbam_id to locate the player row, then appends columns.
    """
    if statcast.empty or "mlbam_id" not in statcast.columns:
        return fg
    player_row = statcast[statcast["mlbam_id"] == mlbam_id]
    if player_row.empty:
        return fg
    # drop duplicated / internal columns before merging
    drop_cols = [c for c in player_row.columns
                 if c in fg.columns or c in ("name","last_name","first_name","_source")]
    player_row = player_row.drop(columns=drop_cols, errors="ignore")
    # broadcast the single row across every column
    for col in player_row.columns:
        fg[col] = player_row.iloc[0][col]
    return fg


def build_player_frame(
    player_info: dict,
    raw: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Construct a single-row analysis-ready DataFrame for one player
    by cleaning and merging all data sources.

    Parameters
    ----------
    player_info : dict   from data_pull.lookup_player()
    raw         : dict   from data_pull.pull_all()

    Returns
    -------
    pd.DataFrame with one row
    """
    mlbam  = player_info["key_mlbam"]
    fg_id  = player_info["key_fangraphs"]
    name   = player_info["name_full"]

    # ── Clean each source ──────────────────────────────────
    fg      = clean_fg_batting(raw.get("fg_batting", pd.DataFrame()))
    ev      = clean_statcast_ev(raw.get("statcast_ev", pd.DataFrame()))
    xs      = clean_statcast_expected(raw.get("statcast_expected", pd.DataFrame()))
    pcts    = clean_statcast_percentiles(raw.get("statcast_percentiles", pd.DataFrame()))
    speed   = clean_statcast_speed(raw.get("statcast_speed", pd.DataFrame()))
    salary  = clean_salary(raw.get("fg_salary", pd.DataFrame()))

    # ── Locate FG row ─────────────────────────────────────
    base = pd.DataFrame()
    if not fg.empty:
        row = pd.DataFrame()
        if "fg_id" in fg.columns:
            row = fg[fg["fg_id"] == fg_id]
        if row.empty and "name" in fg.columns:
            row = fg[fg["name"].str.lower() == name.lower()]
        if not row.empty:
            base = row.iloc[[0]].copy()

    if base.empty:
        base = pd.DataFrame([{"name": name}])

    base = base.reset_index(drop=True)
    base["mlbam_id"] = mlbam
    base["fg_id"]    = fg_id

    # ── Merge Statcast sources ────────────────────────────
    for sc_df, label in [(ev,"ev"), (xs,"xs"), (speed,"spd")]:
        if not sc_df.empty and "mlbam_id" in sc_df.columns:
            player_sc = sc_df[sc_df["mlbam_id"] == mlbam]
            if not player_sc.empty:
                drop = [c for c in player_sc.columns
                        if c in base.columns
                        or c in ("name","last_name","first_name","_source","mlbam_id")]
                player_sc = player_sc.drop(columns=drop, errors="ignore").reset_index(drop=True)
                base = pd.concat([base, player_sc], axis=1)

    # ── Merge percentile ranks ────────────────────────────
    if not pcts.empty and "mlbam_id" in pcts.columns:
        player_pcts = pcts[pcts["mlbam_id"] == mlbam]
        if not player_pcts.empty:
            pct_cols = [c for c in player_pcts.columns
                        if c not in ("mlbam_id","player_name","year","_source")]
            pct_row = player_pcts[pct_cols].reset_index(drop=True)
            pct_row.columns = [f"pct_{c}" for c in pct_cols]
            base = pd.concat([base, pct_row], axis=1)

    # ── Salary join (Lahman uses playerID, not mlbam) ─────
    # Best-effort: match on name if no direct ID bridge
    if not salary.empty:
        sal_match = pd.DataFrame()
        if "name" in salary.columns:
            sal_match = salary[salary["name"].str.lower() == name.lower()]
        if not sal_match.empty:
            base["salary"]   = sal_match.iloc[0].get("salary",   np.nan)
            base["salary_M"] = sal_match.iloc[0].get("salary_M", np.nan)
        else:
            base["salary"]   = np.nan
            base["salary_M"] = np.nan

    # ── Derived columns ───────────────────────────────────
    base = _add_derived_columns(base)

    return base


def build_league_frame(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the full league-wide master frame (all players) from cleaned
    FG batting as the spine, enriched with Statcast columns.

    Used for computing league-average benchmarks and valuation Z-scores.
    """
    fg    = clean_fg_batting(raw.get("fg_batting", pd.DataFrame()))
    ev    = clean_statcast_ev(raw.get("statcast_ev", pd.DataFrame()))
    xs    = clean_statcast_expected(raw.get("statcast_expected", pd.DataFrame()))
    speed = clean_statcast_speed(raw.get("statcast_speed", pd.DataFrame()))

    if fg.empty:
        return pd.DataFrame()

    # merge Statcast onto FG using player name (no reliable shared ID for bulk)
    for sc_df in [ev, xs, speed]:
        if sc_df.empty or "name" not in sc_df.columns:
            continue
        merge_cols = [c for c in sc_df.columns if c not in ("_source",)]
        fg = fg.merge(
            sc_df[merge_cols],
            on="name",
            how="left",
            suffixes=("", "_sc"),
        )

    # salary
    salary = clean_salary(raw.get("fg_salary", pd.DataFrame()))
    if not salary.empty and "name" in salary.columns:
        fg = fg.merge(
            salary[["name","salary","salary_M"]],
            on="name",
            how="left",
        )

    fg = _add_derived_columns(fg)
    return fg


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed helper columns used by analysis and valuation."""
    # OPS+ proxy (FG doesn't supply it but wRC+ is better anyway)
    if "AVG" in df.columns and "OBP" in df.columns:
        df["contact_score"] = (df["AVG"].fillna(0) + df["OBP"].fillna(0)) / 2

    # Plate discipline composite (higher = better eye)
    if "BB_pct" in df.columns and "K_pct" in df.columns:
        df["discipline"] = df["BB_pct"].fillna(0) - df["K_pct"].fillna(0)

    # Power composite
    if "barrel_pct" in df.columns and "ISO" in df.columns:
        df["power_score"] = (
            df["barrel_pct"].fillna(0) * 0.6 +
            df["ISO"].fillna(0) * 100 * 0.4
        )

    # Luck indicator: positive means player is being unlucky (xwOBA > wOBA)
    if "xwOBA" in df.columns and "wOBA_actual" in df.columns:
        df["luck_index"] = df["xwOBA"] - df["wOBA_actual"]
    elif "xwOBA" in df.columns and "wOBA" in df.columns:
        df["luck_index"] = df["xwOBA"] - df["wOBA"]

    return df


# ─────────────────────────────────────────────────────────────
#  DATA QUALITY REPORT
# ─────────────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of completeness for each column.
    Useful for diagnosing which data sources loaded successfully.
    """
    report = pd.DataFrame({
        "column":    df.columns,
        "dtype":     [str(df[c].dtype) for c in df.columns],
        "non_null":  [df[c].notna().sum() for c in df.columns],
        "null_pct":  [(df[c].isna().sum() / len(df) * 100).round(1)
                      for c in df.columns],
    })
    return report.sort_values("null_pct", ascending=False).reset_index(drop=True)
