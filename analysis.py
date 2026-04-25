"""
analysis.py
===========
Core analytical layer. Takes the clean master DataFrame from data_clean.py
and produces:

  1. 20-80 scouting grades for key hitting dimensions
  2. Composite hitting score (wRC+ anchored, Statcast-enhanced)
  3. CRAFT fielding integration — reads craft_range_{season}.csv outputs
     and appends range_runs_saved / range_runs_per_150 to player profiles
  4. Total Player Score = hitting + fielding composite
  5. Z-score rankings vs the league-wide frame

CRAFT integration notes
-----------------------
CRAFT outputs are keyed on Statcast player_id (MLBAM), which is the same
mlbam_id we carry through data_pull / data_clean. The join is therefore
direct. CRAFT Phase 1 only covers the Range Component; this module is
structured so that Arm, Reliability, and Positioning components can be
appended as new CRAFT phases are released.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

# Default path where CRAFT outputs its score CSVs
CRAFT_SCORES_DIR = Path("craft_output/scores")

# Runs per win (standard sabermetric constant)
RUNS_PER_WIN = 10.0

# 20-80 grade boundaries per metric (higher_is_better, [20,40,45,50,55,60,80] thresholds)
GRADE_TABLE: dict[str, tuple[bool, list]] = {
    "wRC_plus":      (True,  [60,  80,  90,  100, 110, 120, 150]),
    "K_pct":         (False, [12,  15,  17,  20,  23,  26,  32]),   # lower = better
    "BB_pct":        (True,  [4,   6,   7,   9,   11,  13,  17]),
    "hard_hit_pct":  (True,  [28,  34,  37,  40,  43,  47,  55]),
    "barrel_pct":    (True,  [2,   5,   7,   9,   12,  15,  20]),
    "xwOBA":         (True,  [.270,.310,.325,.340,.360,.385,.430]),
    "sprint_speed":  (True,  [25,  26.5,27,  27.8,28.5,29,  30]),
    "avg_EV":        (True,  [83,  86,  87,  88.5,90,  91.5,95]),
    "ISO":           (True,  [.100,.140,.160,.175,.200,.230,.280]),
    "wOBA":          (True,  [.270,.310,.320,.330,.350,.375,.420]),
}

GRADE_LABELS = [20, 40, 45, 50, 55, 60, 80]

# Composite hitting score weights (must sum to 1.0)
HITTING_WEIGHTS = {
    "wRC_plus":     0.30,   # overall offensive value, park/league adjusted
    "xwOBA":        0.20,   # quality of contact, luck-neutral
    "barrel_pct":   0.15,   # true power
    "BB_pct":       0.10,   # plate discipline (positive)
    "K_pct":        0.10,   # plate discipline (negative — inverted)
    "hard_hit_pct": 0.10,   # contact quality
    "sprint_speed": 0.05,   # speed / baserunning
}

# CRAFT fielding weight in Total Player Score (0.0 – 1.0)
# Phase 1 only covers Range; we keep this conservative until full CRAFT is built.
CRAFT_WEIGHT = 0.20
HITTING_TOTAL_WEIGHT = 1.0 - CRAFT_WEIGHT


# ─────────────────────────────────────────────────────────────
#  20-80 GRADING
# ─────────────────────────────────────────────────────────────

def grade_metric(stat: str, value: float) -> int:
    """
    Convert a raw stat value to a 20-80 scouting grade.

    Parameters
    ----------
    stat  : str    must be a key in GRADE_TABLE
    value : float  raw stat value

    Returns
    -------
    int  one of [20, 40, 45, 50, 55, 60, 80]
    """
    if stat not in GRADE_TABLE or pd.isna(value):
        return 50  # unknown → average

    higher_is_better, thresholds = GRADE_TABLE[stat]

    if higher_is_better:
        for grade, cut in zip(reversed(GRADE_LABELS), reversed(thresholds)):
            if value >= cut:
                return grade
        return 20
    else:
        for grade, cut in zip(GRADE_LABELS, thresholds):
            if value <= cut:
                return grade
        return 20


def compute_grades(row: pd.Series) -> dict[str, int]:
    """
    Compute 20-80 grades for all gradeable metrics in a player row.

    Returns a dict like {"wRC_plus": 60, "barrel_pct": 55, ...}
    """
    grades = {}
    for stat in GRADE_TABLE:
        val = row.get(stat, np.nan)
        if pd.notna(val):
            grades[stat] = grade_metric(stat, float(val))
    return grades


# ─────────────────────────────────────────────────────────────
#  COMPOSITE HITTING SCORE
# ─────────────────────────────────────────────────────────────

def _normalise_wrc(wrc: float) -> float:
    """Map wRC+ to a 0-100 scale (100 wRC+ → 50, each 10pts ≈ 5pts on scale)."""
    return max(0.0, min(100.0, (wrc - 50) / 100 * 50 + 50))


def _normalise_xwoba(xwoba: float) -> float:
    """Map xwOBA to 0-100 scale (league avg ~.320 → 50)."""
    return max(0.0, min(100.0, (xwoba - 0.250) / 0.200 * 100))


def _normalise_barrel(barrel: float) -> float:
    """Map barrel% to 0-100 scale (15% → ~85)."""
    return max(0.0, min(100.0, barrel / 18.0 * 100))


def _normalise_bb(bb: float) -> float:
    """BB% 0-100 scale (league avg ~8.5% → 50)."""
    return max(0.0, min(100.0, bb / 17.0 * 100))


def _normalise_k(k: float) -> float:
    """K% inverted — lower strikeout rate is better."""
    return max(0.0, min(100.0, (1 - k / 35.0) * 100))


def _normalise_hh(hh: float) -> float:
    """Hard-hit% 0-100 scale (55% → 100)."""
    return max(0.0, min(100.0, hh / 55.0 * 100))


def _normalise_speed(spd: float) -> float:
    """Sprint speed 0-100 scale (30 ft/s → 100)."""
    return max(0.0, min(100.0, (spd - 22) / 10.0 * 100))


NORMALISERS = {
    "wRC_plus":     _normalise_wrc,
    "xwOBA":        _normalise_xwoba,
    "barrel_pct":   _normalise_barrel,
    "BB_pct":       _normalise_bb,
    "K_pct":        _normalise_k,
    "hard_hit_pct": _normalise_hh,
    "sprint_speed": _normalise_speed,
}


def compute_hitting_score(row: pd.Series) -> float:
    """
    Weighted composite hitting score on a 0-100 scale.

    Uses normalised versions of each component, weighted per HITTING_WEIGHTS.
    Missing metrics reduce effective weight proportionally rather than
    penalising the player with a zero.

    Returns
    -------
    float  hitting composite (0-100), or NaN if no metrics available
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for stat, weight in HITTING_WEIGHTS.items():
        val = row.get(stat, np.nan)
        if pd.isna(val):
            continue
        norm_fn = NORMALISERS.get(stat)
        if norm_fn is None:
            continue
        normalised = norm_fn(float(val))
        weighted_sum  += normalised * weight
        total_weight  += weight

    if total_weight < 0.30:   # fewer than 30% of metrics available
        return np.nan

    return round(weighted_sum / total_weight, 2)


# ─────────────────────────────────────────────────────────────
#  CRAFT INTEGRATION
# ─────────────────────────────────────────────────────────────

def load_craft_scores(season: int, scores_dir: Path = CRAFT_SCORES_DIR) -> pd.DataFrame:
    """
    Load CRAFT range component scores for a given season.

    Expects the CSV produced by CRAFT's 03_range_component.py:
        craft_output/scores/craft_range_{season}.csv

    Columns used:
        player_id           — Statcast MLBAM ID (matches our mlbam_id)
        fielder_pos         — defensive position
        opportunities       — total batted balls fielded
        out_differential    — actual_outs - expected_outs
        range_runs_saved    — out_differential * 0.092
        range_runs_per_150  — rate stat per 150 games

    Parameters
    ----------
    season     : int
    scores_dir : Path  directory containing craft_range_{season}.csv

    Returns
    -------
    pd.DataFrame  cleaned CRAFT scores, or empty DataFrame if file not found
    """
    path = scores_dir / f"craft_range_{season}.csv"
    if not path.exists():
        print(f"[CRAFT] No scores found at {path} — fielding component will be skipped.")
        print(f"[CRAFT] Run CRAFT scripts first:  python 03_range_component.py --season {season}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    required = {"player_id", "range_runs_saved"}
    if not required.issubset(df.columns):
        print(f"[CRAFT] Unexpected schema in {path} — missing {required - set(df.columns)}")
        return pd.DataFrame()

    df = df.rename(columns={"player_id": "mlbam_id"})
    df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce")
    df["range_runs_saved"]   = pd.to_numeric(df["range_runs_saved"],   errors="coerce")
    df["range_runs_per_150"] = pd.to_numeric(df.get("range_runs_per_150", np.nan), errors="coerce")

    print(f"[CRAFT] Loaded {len(df)} player-position records for {season}")
    return df


def attach_craft_to_player(player_frame: pd.DataFrame, craft: pd.DataFrame) -> pd.DataFrame:
    """
    Merge CRAFT range scores into a single-player frame.

    If the player has multiple positions in CRAFT (e.g. OF/1B utility),
    we take the position with the highest opportunity count so the
    primary position drives the fielding score.

    Parameters
    ----------
    player_frame : pd.DataFrame  single-row player master frame
    craft        : pd.DataFrame  from load_craft_scores()

    Returns
    -------
    pd.DataFrame  player_frame with craft columns appended
    """
    if craft.empty or "mlbam_id" not in player_frame.columns:
        player_frame["craft_range_runs_saved"]   = np.nan
        player_frame["craft_range_per_150"]      = np.nan
        player_frame["craft_position"]           = np.nan
        player_frame["craft_opportunities"]      = np.nan
        return player_frame

    mlbam = int(player_frame["mlbam_id"].iloc[0])
    player_craft = craft[craft["mlbam_id"] == mlbam].copy()

    if player_craft.empty:
        player_frame["craft_range_runs_saved"]   = np.nan
        player_frame["craft_range_per_150"]      = np.nan
        player_frame["craft_position"]           = np.nan
        player_frame["craft_opportunities"]      = np.nan
        return player_frame

    # Pick primary position (most opportunities)
    if "opportunities" in player_craft.columns:
        player_craft = player_craft.sort_values("opportunities", ascending=False)
    best = player_craft.iloc[0]

    player_frame["craft_range_runs_saved"] = best.get("range_runs_saved",   np.nan)
    player_frame["craft_range_per_150"]    = best.get("range_runs_per_150", np.nan)
    player_frame["craft_position"]         = best.get("fielder_pos",        np.nan)
    player_frame["craft_opportunities"]    = best.get("opportunities",       np.nan)

    return player_frame


def attach_craft_to_league(league_frame: pd.DataFrame, craft: pd.DataFrame) -> pd.DataFrame:
    """
    Bulk-merge CRAFT scores onto the full league frame.
    Used when computing league-wide valuation Z-scores.
    """
    if craft.empty:
        return league_frame

    # Aggregate to one row per player (primary position)
    if "opportunities" in craft.columns:
        craft_primary = (
            craft.sort_values("opportunities", ascending=False)
                 .groupby("mlbam_id", as_index=False)
                 .first()
        )
    else:
        craft_primary = craft.groupby("mlbam_id", as_index=False).first()

    craft_merge = craft_primary[[
        "mlbam_id", "range_runs_saved", "range_runs_per_150", "fielder_pos", "opportunities"
    ]].rename(columns={
        "range_runs_saved":   "craft_range_runs_saved",
        "range_runs_per_150": "craft_range_per_150",
        "fielder_pos":        "craft_position",
        "opportunities":      "craft_opportunities",
    })

    if "mlbam_id" not in league_frame.columns:
        return league_frame

    return league_frame.merge(craft_merge, on="mlbam_id", how="left")


# ─────────────────────────────────────────────────────────────
#  CRAFT FIELDING SCORE (0-100)
# ─────────────────────────────────────────────────────────────

def compute_fielding_score(row: pd.Series, craft_std: float = 5.0) -> float:
    """
    Convert CRAFT range_runs_saved to a 0-100 fielding score.

    Logic: league average = 0 runs saved → score of 50.
    Each standard deviation (~5 runs) moves score by ~15 points.

    Parameters
    ----------
    row       : pd.Series  player row with craft_range_runs_saved
    craft_std : float      std dev of range_runs_saved across league
                           (default 5.0 runs; recalibrate from league_frame)

    Returns
    -------
    float  fielding score 0-100
    """
    rrs = row.get("craft_range_runs_saved", np.nan)
    if pd.isna(rrs):
        return np.nan
    score = 50.0 + (float(rrs) / craft_std) * 15.0
    return round(max(0.0, min(100.0, score)), 2)


# ─────────────────────────────────────────────────────────────
#  TOTAL PLAYER SCORE
# ─────────────────────────────────────────────────────────────

def compute_total_score(
    hitting_score: float,
    fielding_score: float,
    craft_available: bool = True,
) -> float:
    """
    Blend hitting and fielding into a single Total Player Score (0-100).

    If CRAFT is not available, 100% weight goes to hitting.
    If CRAFT is available, applies HITTING_TOTAL_WEIGHT / CRAFT_WEIGHT split.

    Parameters
    ----------
    hitting_score   : float  0-100
    fielding_score  : float  0-100, or NaN
    craft_available : bool   whether CRAFT data was loaded

    Returns
    -------
    float  total score 0-100
    """
    if not craft_available or pd.isna(fielding_score):
        return round(hitting_score, 2) if pd.notna(hitting_score) else np.nan

    if pd.isna(hitting_score):
        return round(fielding_score, 2)

    total = (hitting_score * HITTING_TOTAL_WEIGHT +
             fielding_score * CRAFT_WEIGHT)
    return round(total, 2)


# ─────────────────────────────────────────────────────────────
#  Z-SCORE RANKINGS
# ─────────────────────────────────────────────────────────────

ZSCORE_METRICS = [
    "wRC_plus", "wOBA", "xwOBA", "barrel_pct", "hard_hit_pct",
    "BB_pct", "K_pct", "ISO", "avg_EV", "sprint_speed",
    "craft_range_runs_saved", "hitting_score", "fielding_score", "total_score",
]


def compute_zscores(league_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add a z-score column for each metric in ZSCORE_METRICS.
    Negative metrics (K_pct) are inverted so higher z = better.

    Parameters
    ----------
    league_frame : pd.DataFrame  full league master frame

    Returns
    -------
    pd.DataFrame  with additional z_{metric} columns
    """
    df = league_frame.copy()
    invert = {"K_pct"}  # lower is better

    for metric in ZSCORE_METRICS:
        if metric not in df.columns:
            continue
        col = df[metric].dropna()
        if len(col) < 5:
            continue
        mu, sigma = col.mean(), col.std()
        if sigma == 0:
            continue
        z = (df[metric] - mu) / sigma
        if metric in invert:
            z = -z
        df[f"z_{metric}"] = z.round(3)

    return df


def rank_players(
    league_frame: pd.DataFrame,
    min_pa: int = 100,
    rank_by: str = "total_score",
) -> pd.DataFrame:
    """
    Rank all qualifying players by a chosen composite metric.

    Parameters
    ----------
    league_frame : pd.DataFrame
    min_pa       : int   minimum plate appearances
    rank_by      : str   column to rank by

    Returns
    -------
    pd.DataFrame  sorted leaderboard with rank column
    """
    df = league_frame.copy()

    if "PA" in df.columns:
        df = df[df["PA"].fillna(0) >= min_pa]

    if rank_by not in df.columns:
        print(f"[rank_players] '{rank_by}' not in frame — ranking by hitting_score")
        rank_by = "hitting_score"

    df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# ─────────────────────────────────────────────────────────────
#  FULL ANALYSIS PIPELINE FOR A PLAYER
# ─────────────────────────────────────────────────────────────

def analyse_player(
    player_frame: pd.DataFrame,
    craft: pd.DataFrame,
    craft_std: float = 5.0,
) -> pd.DataFrame:
    """
    Run the full analysis pipeline on a single-player frame:
      1. Attach CRAFT fielding scores
      2. Compute 20-80 grades
      3. Compute hitting score
      4. Compute fielding score
      5. Compute total score

    Parameters
    ----------
    player_frame : pd.DataFrame  single-row frame from data_clean
    craft        : pd.DataFrame  from load_craft_scores()
    craft_std    : float         std dev of range_runs_saved (from league frame)

    Returns
    -------
    pd.DataFrame  enriched single-row frame
    """
    pf = attach_craft_to_player(player_frame.copy(), craft)
    row = pf.iloc[0]

    grades = compute_grades(row)
    for metric, grade in grades.items():
        pf[f"grade_{metric}"] = grade

    pf["hitting_score"]  = compute_hitting_score(row)
    craft_avail = pd.notna(row.get("craft_range_runs_saved", np.nan))
    pf["fielding_score"] = compute_fielding_score(row, craft_std=craft_std)
    pf["total_score"]    = compute_total_score(
        float(pf["hitting_score"].iloc[0]) if pd.notna(pf["hitting_score"].iloc[0]) else np.nan,
        float(pf["fielding_score"].iloc[0]) if pd.notna(pf["fielding_score"].iloc[0]) else np.nan,
        craft_available=craft_avail,
    )

    return pf


def analyse_league(
    league_frame: pd.DataFrame,
    craft: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the full analysis pipeline across the entire league frame.
    Used to compute Z-scores and valuation benchmarks.

    Parameters
    ----------
    league_frame : pd.DataFrame  from data_clean.build_league_frame()
    craft        : pd.DataFrame  from load_craft_scores()

    Returns
    -------
    pd.DataFrame  enriched league frame with scores and z-scores
    """
    df = attach_craft_to_league(league_frame.copy(), craft)

    # Compute craft std from real data
    craft_std = 5.0
    if "craft_range_runs_saved" in df.columns:
        s = df["craft_range_runs_saved"].dropna().std()
        if s > 0:
            craft_std = s

    # Scores
    df["hitting_score"]  = df.apply(compute_hitting_score, axis=1)
    df["fielding_score"] = df.apply(
        lambda r: compute_fielding_score(r, craft_std=craft_std), axis=1
    )
    craft_avail = "craft_range_runs_saved" in df.columns

    df["total_score"] = df.apply(
        lambda r: compute_total_score(
            r.get("hitting_score", np.nan),
            r.get("fielding_score", np.nan),
            craft_available=craft_avail,
        ),
        axis=1,
    )

    # Grades for each player
    grade_cols = list(GRADE_TABLE.keys())
    for stat in grade_cols:
        if stat in df.columns:
            df[f"grade_{stat}"] = df.apply(
                lambda r, s=stat: grade_metric(s, r[s]) if pd.notna(r[s]) else np.nan,
                axis=1,
            )

    df = compute_zscores(df)
    return df
