"""
data_pull.py
============
Responsible for fetching all raw data from external sources:
  - FanGraphs  : traditional + advanced batting stats, salary
  - Baseball Savant (Statcast): exit velo, barrels, expected stats,
                                percentile ranks, sprint speed

All functions return raw DataFrames with no transformation applied.
Errors are caught gracefully and return empty DataFrames so the
pipeline can continue with partial data.
"""

import warnings
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import pybaseball as pb
    pb.cache.enable()
except ImportError:
    raise ImportError("Run:  pip install pybaseball")


# ─────────────────────────────────────────────────────────────
#  PLAYER LOOKUP
# ─────────────────────────────────────────────────────────────

def lookup_player(name: str) -> dict | None:
    """
    Resolve a player name to their MLBAM and FanGraphs IDs.

    Parameters
    ----------
    name : str
        Full name, e.g. "Aaron Judge"

    Returns
    -------
    dict with keys {key_mlbam, key_fangraphs, name_full} or None
    """
    parts = name.strip().split()
    if len(parts) < 2:
        print(f"[lookup] Need first AND last name — got: '{name}'")
        return None
    last, first = parts[-1], " ".join(parts[:-1])
    try:
        res = pb.playerid_lookup(last, first, fuzzy=True)
    except Exception as exc:
        print(f"[lookup] playerid_lookup failed: {exc}")
        return None
    if res is None or res.empty:
        print(f"[lookup] No match for '{name}'")
        return None
    row = res.iloc[0]
    return {
        "key_mlbam":      int(row.get("key_mlbam", 0)),
        "key_fangraphs":  str(int(row.get("key_fangraphs", 0))),
        "name_full": (
            f"{str(row.get('name_first','')).title()} "
            f"{str(row.get('name_last','')).title()}"
        ).strip(),
    }


# ─────────────────────────────────────────────────────────────
#  FANGRAPHS — BATTING
# ─────────────────────────────────────────────────────────────

def fetch_fg_batting(year: int, min_pa: int = 30) -> pd.DataFrame:
    """
    FanGraphs batting leaderboard: traditional + advanced stats.
    Includes wOBA, wRC+, BB%, K%, ISO, BABIP, WAR, Off, Def, Spd.

    Parameters
    ----------
    year   : int   season
    min_pa : int   minimum plate appearances filter
    """
    try:
        df = pb.batting_stats(year, qual=min_pa)
        df["_source"] = "fg_batting"
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[fetch_fg_batting] {exc}")
        return pd.DataFrame()


def fetch_fg_salary(year: int) -> pd.DataFrame:
    """
    FanGraphs salary data via Lahman/Cots.
    Returns player salaries for the given season.

    Parameters
    ----------
    year : int
    """
    try:
        # pybaseball wraps the Lahman salary table
        df = pb.lahman.salaries()
        if df is None or df.empty:
            return pd.DataFrame()
        df = df[df["yearID"] == year].copy()
        df["salary_M"] = df["salary"] / 1_000_000
        df["_source"] = "lahman_salary"
        return df
    except Exception as exc:
        print(f"[fetch_fg_salary] {exc}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
#  STATCAST — LEADERBOARDS
# ─────────────────────────────────────────────────────────────

def fetch_statcast_expected(year: int, min_pa: int = 25) -> pd.DataFrame:
    """
    Baseball Savant expected stats: xBA, xSLG, xwOBA, xISO.
    These are quality-of-contact metrics removing luck from outcomes.
    """
    try:
        df = pb.statcast_batter_expected_stats(year, minPA=min_pa)
        if df is not None:
            df["_source"] = "statcast_expected"
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[fetch_statcast_expected] {exc}")
        return pd.DataFrame()


def fetch_statcast_ev_barrels(year: int, min_bbe: int = 25) -> pd.DataFrame:
    """
    Baseball Savant exit velocity & barrel leaderboard.
    Avg EV, Max EV, Avg LA, Hard Hit%, Barrel%, Barrels/PA.
    """
    try:
        df = pb.statcast_batter_exitvelo_barrels(year, minBBE=min_bbe)
        if df is not None:
            df["_source"] = "statcast_ev"
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[fetch_statcast_ev_barrels] {exc}")
        return pd.DataFrame()


def fetch_statcast_percentiles(year: int) -> pd.DataFrame:
    """
    Baseball Savant percentile ranks for all tracked metrics.
    One row per player with 0–100 percentile for each Statcast stat.
    """
    try:
        df = pb.statcast_batter_percentile_ranks(year)
        if df is not None:
            df["_source"] = "statcast_pct"
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[fetch_statcast_percentiles] {exc}")
        return pd.DataFrame()


def fetch_statcast_sprint_speed(year: int, min_opp: int = 10) -> pd.DataFrame:
    """
    Baseball Savant sprint speed leaderboard (ft/s).
    """
    try:
        df = pb.statcast_sprint_speed(year, min_opp=min_opp)
        if df is not None:
            df["_source"] = "statcast_speed"
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[fetch_statcast_sprint_speed] {exc}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
#  CONVENIENCE — PULL ALL AT ONCE
# ─────────────────────────────────────────────────────────────

def pull_all(year: int) -> dict[str, pd.DataFrame]:
    """
    Pull every data source for a given season in one call.
    Returns a dict keyed by source name. Safe to call even if
    some endpoints fail — those will return empty DataFrames.

    Parameters
    ----------
    year : int

    Returns
    -------
    dict with keys:
        fg_batting, fg_salary, statcast_expected,
        statcast_ev, statcast_percentiles, statcast_speed
    """
    print(f"[data_pull] Fetching {year} season data…")
    return {
        "fg_batting":           fetch_fg_batting(year),
        "fg_salary":            fetch_fg_salary(year),
        "statcast_expected":    fetch_statcast_expected(year),
        "statcast_ev":          fetch_statcast_ev_barrels(year),
        "statcast_percentiles": fetch_statcast_percentiles(year),
        "statcast_speed":       fetch_statcast_sprint_speed(year),
    }
