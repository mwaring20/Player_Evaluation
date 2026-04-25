"""
output.py
=========
All display and export logic. Completely decoupled from analysis —
takes DataFrames in, produces formatted output.

Supports:
  - Rich terminal player cards (single player)
  - Rich terminal leaderboard tables
  - Side-by-side player comparison
  - CSV export
  - Markdown report export
  - Plain-text fallback (no rich dependency)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional rich ──────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box as rbox
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None


# ─────────────────────────────────────────────────────────────
#  CONSTANTS / DISPLAY CONFIG
# ─────────────────────────────────────────────────────────────

GRADE_STYLE = {
    80: "bold bright_cyan",
    60: "bold green",
    55: "green",
    50: "yellow",
    45: "dark_orange",
    40: "red",
    20: "bold red",
}

VALUE_STYLE = {
    "ELITE VALUE":     "bold bright_cyan",
    "GOOD VALUE":      "bold green",
    "FAIR":            "yellow",
    "OVERPAID":        "red",
    "HIGHLY OVERPAID": "bold red",
    "NO SALARY DATA":  "dim",
}

PCT_BAR_WIDTH = 22


def _pct_bar(pct: float, width: int = PCT_BAR_WIDTH) -> str:
    """ASCII percentile bar for terminal display."""
    if pd.isna(pct):
        return "—"
    filled = round(float(pct) / 100 * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {int(pct):>3}th"


def _fmt(val, decimals: int = 3, pct: bool = False) -> str:
    """Safe formatter for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if pct:
        return f"{float(val):.1f}%"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _grade_str(grade: int | float) -> str:
    """20-80 grade with nearest valid grade."""
    valid = [20, 40, 45, 50, 55, 60, 80]
    if pd.isna(grade):
        return "—"
    g = int(grade)
    nearest = min(valid, key=lambda x: abs(x - g))
    return str(nearest)


# ─────────────────────────────────────────────────────────────
#  PLAYER CARD — RICH
# ─────────────────────────────────────────────────────────────

def _player_card_rich(row: pd.Series):
    name        = row.get("name", "Unknown")
    team        = row.get("team", "—")
    age         = row.get("age", "—")
    year        = row.get("year", "")
    hit_score   = _fmt(row.get("hitting_score"),   2)
    fld_score   = _fmt(row.get("fielding_score"),  2)
    total_score = _fmt(row.get("total_score"),     2)
    v_label     = str(row.get("value_label", "NO SALARY DATA"))
    v_gap       = row.get("value_gap_M", np.nan)
    sal_actual  = row.get("actual_salary_M",    row.get("salary_M", np.nan))
    sal_est     = row.get("estimated_salary_M", np.nan)

    # Header panel
    gap_str = f"{v_gap:+.1f}M" if pd.notna(v_gap) else "—"
    v_col   = VALUE_STYLE.get(v_label, "white")
    console.print()
    console.print(Panel(
        Text(f"⚾  {name}  |  {team}  |  Age {age}  |  {year}", justify="center",
             style="bold white"),
        subtitle=f"[{v_col}]{v_label}  (gap: {gap_str})[/{v_col}]",
        style="bold blue", expand=False,
    ))

    # Score summary line
    score_table = Table(box=rbox.SIMPLE, show_header=False, padding=(0,2))
    score_table.add_column("", style="dim")
    score_table.add_column("", justify="center", style="bold")
    score_table.add_row("Hitting Score",  f"[yellow]{hit_score}[/yellow] / 100")
    score_table.add_row("Fielding Score", f"[cyan]{fld_score}[/cyan] / 100")
    score_table.add_row("Total Score",    f"[bold white]{total_score}[/bold white] / 100")
    score_table.add_row("Actual Salary",  f"${_fmt(sal_actual, 1)}M")
    score_table.add_row("Est. Market",    f"${_fmt(sal_est, 1)}M")
    console.print(score_table)

    # Traditional stats
    _section_table(row, "Traditional Stats", [
        ("G",   _fmt(row.get("G"),  0)),
        ("PA",  _fmt(row.get("PA"), 0)),
        ("HR",  _fmt(row.get("HR"), 0)),
        ("RBI", _fmt(row.get("RBI"),0)),
        ("SB",  _fmt(row.get("SB"), 0)),
        ("AVG", _fmt(row.get("AVG"), 3)),
        ("OBP", _fmt(row.get("OBP"), 3)),
        ("SLG", _fmt(row.get("SLG"), 3)),
        ("OPS", _fmt(row.get("OPS"), 3)),
    ])

    # Advanced
    _section_table(row, "Advanced Stats", [
        ("wOBA",    _fmt(row.get("wOBA"),     3)),
        ("wRC+",    _fmt(row.get("wRC_plus"), 0)),
        ("BB%",     _fmt(row.get("BB_pct"),   1) + "%"),
        ("K%",      _fmt(row.get("K_pct"),    1) + "%"),
        ("ISO",     _fmt(row.get("ISO"),      3)),
        ("BABIP",   _fmt(row.get("BABIP"),    3)),
        ("WAR",     _fmt(row.get("WAR"),      1)),
        ("Off",     _fmt(row.get("Off"),      1)),
        ("Def",     _fmt(row.get("Def"),      1)),
    ])

    # Statcast / contact quality
    _section_table(row, "Statcast Quality", [
        ("Avg EV (mph)",  _fmt(row.get("avg_EV"),      1)),
        ("Max EV (mph)",  _fmt(row.get("max_EV"),      1)),
        ("Avg LA (°)",    _fmt(row.get("avg_LA"),      1)),
        ("Hard Hit%",     _fmt(row.get("hard_hit_pct"),1) + "%"),
        ("Barrel%",       _fmt(row.get("barrel_pct"),  1) + "%"),
        ("xBA",           _fmt(row.get("xBA"),         3)),
        ("xSLG",          _fmt(row.get("xSLG"),        3)),
        ("xwOBA",         _fmt(row.get("xwOBA"),       3)),
        ("xwOBA − wOBA",  _fmt(row.get("xwOBA_diff"),  3)),
    ])

    # CRAFT fielding
    craft_rrs  = row.get("craft_range_runs_saved", np.nan)
    craft_p150 = row.get("craft_range_per_150",    np.nan)
    craft_pos  = row.get("craft_position",         "—")
    craft_opp  = row.get("craft_opportunities",     np.nan)
    _section_table(row, "CRAFT Fielding (Phase 1 — Range)", [
        ("Primary Position",     str(craft_pos)),
        ("Opportunities",        _fmt(craft_opp, 0)),
        ("Range Runs Saved",     _fmt(craft_rrs, 2)),
        ("Range Runs / 150G",    _fmt(craft_p150, 2)),
        ("Fielding Score",       f"{fld_score} / 100"),
    ], note="CRAFT Phase 1 only. Arm/Reliability components pending." if pd.isna(craft_rrs) else None)

    # Speed
    _section_table(row, "Speed", [
        ("Sprint Speed (ft/s)", _fmt(row.get("sprint_speed"), 1)),
        ("Competitive Runs",    _fmt(row.get("competitive_runs"), 0)),
    ])

    # 20-80 grades
    _grades_table(row)

    # Percentile ranks
    _percentiles_table(row)


def _section_table(row: pd.Series, title: str, rows: list[tuple], note: str | None = None):
    t = Table(title=title, box=rbox.SIMPLE_HEAD,
              title_style="bold magenta", show_header=True,
              header_style="bold dim")
    t.add_column("Metric", style="dim", width=24)
    t.add_column("Value",  justify="right", width=14)
    for metric, val in rows:
        t.add_row(metric, str(val))
    if note:
        t.caption = f"[dim italic]{note}[/dim italic]"
    console.print(t)


def _grades_table(row: pd.Series):
    grade_metrics = [c for c in row.index if c.startswith("grade_")]
    if not grade_metrics:
        return
    t = Table(title="20-80 Scouting Grades", box=rbox.SIMPLE_HEAD,
              title_style="bold magenta", show_header=True,
              header_style="bold dim")
    t.add_column("Metric",  style="dim", width=24)
    t.add_column("Grade",   justify="center", width=8)
    for col in grade_metrics:
        label = col.replace("grade_", "").replace("_", " ").title()
        g     = row.get(col, np.nan)
        g_str = _grade_str(g)
        style = GRADE_STYLE.get(int(g_str) if g_str != "—" else 50, "white")
        t.add_row(label, f"[{style}]{g_str}[/{style}]")
    console.print(t)


def _percentiles_table(row: pd.Series):
    pct_metrics = [c for c in row.index if c.startswith("pct_")]
    if not pct_metrics:
        return
    t = Table(title="MLB Percentile Ranks", box=rbox.SIMPLE_HEAD,
              title_style="bold magenta", show_header=True,
              header_style="bold dim")
    t.add_column("Metric",  style="dim",   width=24)
    t.add_column("Pct",     justify="right", width=6)
    t.add_column("Bar",     width=PCT_BAR_WIDTH + 6)
    for col in pct_metrics:
        label = col.replace("pct_", "").replace("_", " ").title()
        pct   = row.get(col, np.nan)
        if pd.isna(pct):
            continue
        bar   = _pct_bar(pct)
        color = "green" if pct >= 70 else ("yellow" if pct >= 40 else "red")
        t.add_row(label, str(int(pct)), f"[{color}]{bar}[/{color}]")
    console.print(t)


# ─────────────────────────────────────────────────────────────
#  PLAYER CARD — PLAIN TEXT
# ─────────────────────────────────────────────────────────────

def _player_card_plain(row: pd.Series):
    name  = row.get("name", "Unknown")
    team  = row.get("team", "—")
    sep   = "=" * 60

    print(f"\n{sep}")
    print(f"  {name.upper()}  |  {team}")
    print(f"  Hitting: {_fmt(row.get('hitting_score'),2)}/100  "
          f"Fielding: {_fmt(row.get('fielding_score'),2)}/100  "
          f"Total: {_fmt(row.get('total_score'),2)}/100")
    print(f"  Value: {row.get('value_label','—')}  "
          f"(Gap: {_fmt(row.get('value_gap_M'),1)}M)")
    print(sep)

    sections = {
        "Traditional":  [("AVG",row.get("AVG")),("OBP",row.get("OBP")),
                         ("SLG",row.get("SLG")),("HR",row.get("HR")),
                         ("WAR",row.get("WAR"))],
        "Advanced":     [("wRC+",row.get("wRC_plus")),("wOBA",row.get("wOBA")),
                         ("BB%",row.get("BB_pct")),("K%",row.get("K_pct")),
                         ("ISO",row.get("ISO"))],
        "Statcast":     [("xwOBA",row.get("xwOBA")),("Barrel%",row.get("barrel_pct")),
                         ("Avg EV",row.get("avg_EV")),("Hard Hit%",row.get("hard_hit_pct"))],
        "CRAFT":        [("Range Runs Saved",row.get("craft_range_runs_saved")),
                         ("Range/150G",row.get("craft_range_per_150")),
                         ("Position",row.get("craft_position"))],
        "Salary":       [("Actual ($M)",row.get("actual_salary_M")),
                         ("Estimated ($M)",row.get("estimated_salary_M")),
                         ("Gap ($M)",row.get("value_gap_M"))],
    }
    for section, items in sections.items():
        print(f"\n── {section} ──")
        for k, v in items:
            print(f"  {k:<28} {_fmt(v, 2)}")
    print()


# ─────────────────────────────────────────────────────────────
#  PUBLIC: PRINT PLAYER CARD
# ─────────────────────────────────────────────────────────────

def print_player_card(player_frame: pd.DataFrame):
    """
    Print a full player card for a single-row DataFrame.

    Parameters
    ----------
    player_frame : pd.DataFrame  single-row enriched frame from analysis.analyse_player()
    """
    row = player_frame.iloc[0]
    if RICH:
        _player_card_rich(row)
    else:
        _player_card_plain(row)


# ─────────────────────────────────────────────────────────────
#  LEADERBOARD TABLE
# ─────────────────────────────────────────────────────────────

def print_leaderboard(
    df: pd.DataFrame,
    title: str = "Player Leaderboard",
    cols: list[str] | None = None,
):
    """
    Print a rich or plain leaderboard table.

    Parameters
    ----------
    df    : pd.DataFrame  output from valuation.valuation_leaderboard()
    title : str
    cols  : list | None   columns to display (defaults to a curated set)
    """
    default_cols = [
        "rank", "name", "team", "PA", "wRC_plus", "xwOBA",
        "craft_range_runs_saved", "WAR",
        "hitting_score", "fielding_score", "total_score",
        "actual_salary_M", "estimated_salary_M", "value_gap_M", "value_label",
    ]
    show_cols = [c for c in (cols or default_cols) if c in df.columns]

    if RICH:
        t = Table(title=title, box=rbox.SIMPLE_HEAD,
                  title_style="bold blue", show_header=True,
                  header_style="bold dim")
        for col in show_cols:
            t.add_column(col, justify="right" if col not in ("name","team","value_label") else "left")

        for _, row in df.iterrows():
            cells = []
            for col in show_cols:
                val = row.get(col, np.nan)
                if col == "value_label":
                    style = VALUE_STYLE.get(str(val), "white")
                    cells.append(f"[{style}]{val}[/{style}]")
                elif col == "value_gap_M" and pd.notna(val):
                    color = "green" if float(val) > 0 else "red"
                    cells.append(f"[{color}]{val:+.1f}[/{color}]")
                else:
                    cells.append(_fmt(val, 2))
            t.add_row(*cells)
        console.print(t)

    else:
        print(f"\n{'─'*80}")
        print(f"  {title}")
        print(f"{'─'*80}")
        header = "  " + "  ".join(f"{c[:14]:<14}" for c in show_cols)
        print(header)
        for _, row in df.iterrows():
            line = "  " + "  ".join(
                f"{str(row.get(c,'—'))[:14]:<14}" for c in show_cols
            )
            print(line)
        print()


# ─────────────────────────────────────────────────────────────
#  COMPARISON TABLE
# ─────────────────────────────────────────────────────────────

def print_comparison(frames: list[pd.DataFrame], title: str = "Player Comparison"):
    """
    Side-by-side comparison of multiple player frames.

    Parameters
    ----------
    frames : list[pd.DataFrame]  each is a single-row analysed frame
    title  : str
    """
    if not frames:
        return

    names = [f.iloc[0].get("name", f"Player {i+1}") for i, f in enumerate(frames)]
    combined = pd.concat([f.iloc[[0]] for f in frames], ignore_index=True)

    COMPARE_SECTIONS = {
        "Performance": ["wRC_plus","wOBA","xwOBA","barrel_pct","hard_hit_pct","WAR"],
        "Plate Disc.": ["BB_pct","K_pct","ISO","BABIP"],
        "Statcast":    ["avg_EV","max_EV","sprint_speed"],
        "CRAFT":       ["craft_range_runs_saved","craft_range_per_150","craft_position"],
        "Scores":      ["hitting_score","fielding_score","total_score"],
        "Valuation":   ["actual_salary_M","estimated_salary_M","value_gap_M","value_label"],
    }

    if RICH:
        console.print(Panel(Text(title, justify="center", style="bold white"),
                            style="bold blue", expand=False))
        for section, metrics in COMPARE_SECTIONS.items():
            avail = [m for m in metrics if m in combined.columns]
            if not avail:
                continue
            t = Table(title=section, box=rbox.SIMPLE_HEAD,
                      title_style="bold magenta", show_header=True,
                      header_style="bold dim")
            t.add_column("Metric", style="dim", width=24)
            for n in names:
                t.add_column(n[:16], justify="right", width=14)
            for metric in avail:
                label = metric.replace("_", " ").title()
                cells = [label]
                vals  = [combined.iloc[i].get(metric, np.nan) for i in range(len(frames))]
                # highlight best value
                numeric_vals = [float(v) for v in vals if pd.notna(v) and str(v).replace(".","").replace("-","").isdigit()]
                best = max(numeric_vals) if numeric_vals else None
                for val in vals:
                    fmt = _fmt(val, 2)
                    try:
                        is_best = best is not None and abs(float(val) - best) < 0.001
                        cells.append(f"[bold green]{fmt}[/bold green]" if is_best else fmt)
                    except (ValueError, TypeError):
                        cells.append(fmt)
                t.add_row(*cells)
            console.print(t)
    else:
        print(f"\n{'─'*70}\n  {title}\n{'─'*70}")
        for section, metrics in COMPARE_SECTIONS.items():
            avail = [m for m in metrics if m in combined.columns]
            if not avail:
                continue
            print(f"\n── {section} ──")
            header = f"  {'Metric':<24}" + "".join(f"{n[:14]:>16}" for n in names)
            print(header)
            for metric in avail:
                row_str = f"  {metric.replace('_',' ').title():<24}"
                for i in range(len(frames)):
                    val = combined.iloc[i].get(metric, np.nan)
                    row_str += f"{_fmt(val, 2):>16}"
                print(row_str)
        print()


# ─────────────────────────────────────────────────────────────
#  EXPORT
# ─────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, path: str | Path):
    """Export any DataFrame to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"[output] CSV saved → {p}")


def export_markdown(
    player_frame: pd.DataFrame,
    path: str | Path,
    valued_frame: pd.DataFrame | None = None,
):
    """
    Export a markdown report for a single player.

    Parameters
    ----------
    player_frame : pd.DataFrame  single-row analysed frame
    path         : str | Path    output .md file path
    valued_frame : pd.DataFrame  optional, for valuation section
    """
    p   = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = player_frame.iloc[0]

    name  = row.get("name", "Player")
    team  = row.get("team", "—")
    year  = row.get("year", "—")

    lines = [
        f"# {name} — {year} Full Player Analysis",
        f"**Team:** {team}  |  **Age:** {row.get('age','—')}  |  **Season:** {year}",
        "",
        "---",
        "",
        "## Summary Scores",
        "",
        f"| Dimension     | Score (0–100) |",
        f"|---------------|---------------|",
        f"| Hitting       | {_fmt(row.get('hitting_score'),  2)} |",
        f"| Fielding      | {_fmt(row.get('fielding_score'), 2)} |",
        f"| **Total**     | **{_fmt(row.get('total_score'), 2)}** |",
        "",
    ]

    # Valuation
    v_label = row.get("value_label", "NO SALARY DATA")
    v_gap   = row.get("value_gap_M", np.nan)
    sal_act = row.get("actual_salary_M", row.get("salary_M", np.nan))
    sal_est = row.get("estimated_salary_M", np.nan)
    lines += [
        "## Valuation",
        "",
        f"| Metric             | Value     |",
        f"|--------------------|-----------|",
        f"| Actual Salary      | ${_fmt(sal_act, 1)}M |",
        f"| Estimated Market   | ${_fmt(sal_est, 1)}M |",
        f"| Value Gap          | {_fmt(v_gap, 1)}M |",
        f"| **Value Label**    | **{v_label}** |",
        "",
    ]

    def md_section(title, items):
        rows = [f"## {title}", "", "| Metric | Value |", "|--------|-------|"]
        for k, v in items:
            rows.append(f"| {k} | {_fmt(v, 3)} |")
        rows.append("")
        return rows

    lines += md_section("Traditional Stats", [
        ("G",   row.get("G")),  ("PA",  row.get("PA")),
        ("HR",  row.get("HR")), ("AVG", row.get("AVG")),
        ("OBP", row.get("OBP")),("SLG", row.get("SLG")),
        ("OPS", row.get("OPS")),
    ])
    lines += md_section("Advanced Stats", [
        ("wOBA",  row.get("wOBA")),    ("wRC+",  row.get("wRC_plus")),
        ("BB%",   row.get("BB_pct")),  ("K%",    row.get("K_pct")),
        ("ISO",   row.get("ISO")),     ("BABIP", row.get("BABIP")),
        ("WAR",   row.get("WAR")),
    ])
    lines += md_section("Statcast", [
        ("xBA",      row.get("xBA")),
        ("xSLG",     row.get("xSLG")),
        ("xwOBA",    row.get("xwOBA")),
        ("Avg EV",   row.get("avg_EV")),
        ("Barrel%",  row.get("barrel_pct")),
        ("Hard Hit%",row.get("hard_hit_pct")),
    ])
    lines += md_section("CRAFT Fielding", [
        ("Position",        row.get("craft_position")),
        ("Opportunities",   row.get("craft_opportunities")),
        ("Range Runs Saved",row.get("craft_range_runs_saved")),
        ("Range / 150G",    row.get("craft_range_per_150")),
    ])

    # Grades
    grade_cols = [c for c in row.index if c.startswith("grade_")]
    if grade_cols:
        lines += ["## 20-80 Scouting Grades", "", "| Metric | Grade |", "|--------|-------|"]
        for col in grade_cols:
            label = col.replace("grade_","").replace("_"," ").title()
            lines.append(f"| {label} | {_grade_str(row.get(col, np.nan))} |")
        lines.append("")

    lines += [
        "---",
        "*Generated by Hitter Analysis Pipeline + CRAFT Fielding*",
    ]

    p.write_text("\n".join(lines))
    print(f"[output] Markdown report saved → {p}")
