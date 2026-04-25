"""
main.py
=======
Entry point for the full Hitter Analysis + CRAFT + Valuation pipeline.

Usage
-----
# Single player card
python main.py --name "Aaron Judge" --year 2024

# Comparison
python main.py --compare "Aaron Judge" "Juan Soto" "Mookie Betts" --year 2024

# League-wide leaderboard (top 20 bargains)
python main.py --leaderboard --year 2024 --top 20

# Filter to a value tier
python main.py --leaderboard --year 2024 --tier "ELITE VALUE"

# Export player report
python main.py --name "Aaron Judge" --year 2024 --export

# Interactive
python main.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from data_pull   import lookup_player, pull_all
from data_clean  import build_player_frame, build_league_frame
from analysis    import (
    analyse_player, analyse_league,
    load_craft_scores,
)
from valuation   import run_valuation, valuation_leaderboard, best_bargains, biggest_overpays, tier_summary
from output      import (
    print_player_card, print_leaderboard,
    print_comparison, export_csv, export_markdown,
)


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _league_pipeline(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull, clean, analyse and value the full league frame."""
    raw    = pull_all(year)
    league = build_league_frame(raw)

    if league.empty:
        print("❌  League frame is empty — check your data sources.")
        sys.exit(1)

    craft  = load_craft_scores(year)

    # Get craft std before analysing so fielding scores are calibrated
    craft_std = 5.0
    if not craft.empty and "range_runs_saved" in craft.columns:
        s = craft["range_runs_saved"].std()
        if s > 0:
            craft_std = s

    analysed = analyse_league(league, craft)
    valued   = run_valuation(analysed, year)
    return analysed, valued


# ─────────────────────────────────────────────────────────────
#  SINGLE PLAYER
# ─────────────────────────────────────────────────────────────

def run_single(name: str, year: int, export: bool = False):
    print(f"\n📡  Fetching {year} data for {name}…")
    raw  = pull_all(year)

    info = lookup_player(name)
    if info is None:
        sys.exit(1)

    craft = load_craft_scores(year)
    craft_std = 5.0
    if not craft.empty and "range_runs_saved" in craft.columns:
        s = craft["range_runs_saved"].std()
        if s > 0:
            craft_std = s

    pf       = build_player_frame(info, raw)
    pf["year"] = year
    analysed = analyse_player(pf, craft, craft_std=craft_std)

    # Valuation — need league frame for salary model
    league   = build_league_frame(raw)
    league_a = analyse_league(league, craft)
    valued_l = run_valuation(league_a, year)

    # Patch valuation columns back onto the player frame
    if "name" in analysed.columns and not valued_l.empty:
        player_val = valued_l[valued_l.get("name","").str.lower() == info["name_full"].lower()] \
            if "name" in valued_l.columns else pd.DataFrame()
        for col in ("estimated_salary_M","actual_salary_M","value_gap_M","value_label","value_score"):
            if not player_val.empty and col in player_val.columns:
                analysed[col] = player_val.iloc[0][col]
            else:
                analysed[col] = None

    print_player_card(analysed)

    if export:
        safe = info["name_full"].replace(" ", "_")
        export_csv(analysed, f"outputs/{safe}_{year}_profile.csv")
        export_markdown(analysed, f"outputs/{safe}_{year}_report.md", valued_l)


# ─────────────────────────────────────────────────────────────
#  COMPARISON
# ─────────────────────────────────────────────────────────────

def run_comparison(names: list[str], year: int, export: bool = False):
    print(f"\n📡  Fetching {year} data…")
    raw   = pull_all(year)
    craft = load_craft_scores(year)
    craft_std = 5.0
    if not craft.empty and "range_runs_saved" in craft.columns:
        s = craft["range_runs_saved"].std()
        if s > 0:
            craft_std = s

    frames = []
    for name in names:
        info = lookup_player(name)
        if info is None:
            continue
        pf       = build_player_frame(info, raw)
        pf["year"] = year
        analysed = analyse_player(pf, craft, craft_std=craft_std)
        frames.append(analysed)
        print(f"  ✅  {info['name_full']}")

    if not frames:
        print("❌  No players found.")
        return

    print_comparison(frames, title=f"Player Comparison — {year}")

    if export:
        combined = pd.concat(frames, ignore_index=True)
        export_csv(combined, f"outputs/comparison_{year}.csv")


# ─────────────────────────────────────────────────────────────
#  LEADERBOARD
# ─────────────────────────────────────────────────────────────

def run_leaderboard(
    year: int,
    top_n: int = 30,
    tier: str | None = None,
    sort_by: str = "value_gap_M",
    export: bool = False,
):
    print(f"\n📡  Building {year} league-wide valuation leaderboard…")
    _, valued = _league_pipeline(year)

    board = valuation_leaderboard(valued, top_n=top_n, label_filter=tier, sort_by=sort_by)

    title = f"{year} Valuation Leaderboard"
    if tier:
        title += f" — {tier}"
    print_leaderboard(board, title=title)

    print("\n📊  Tier Summary:")
    ts = tier_summary(valued)
    if not ts.empty:
        print(ts.to_string(index=False))

    print("\n🔎  Top Bargains:")
    print_leaderboard(best_bargains(valued, 10), title="Top 10 Bargains")

    print("\n💸  Biggest Overpays:")
    print_leaderboard(biggest_overpays(valued, 10), title="Top 10 Overpaid")

    if export:
        export_csv(valued, f"outputs/valuation_leaderboard_{year}.csv")


# ─────────────────────────────────────────────────────────────
#  INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────

def interactive():
    print("\n⚾  HITTER ANALYSIS + CRAFT + VALUATION")
    print("─" * 45)
    print("  1. Single player card")
    print("  2. Compare multiple players")
    print("  3. League leaderboard")
    choice = input("\nChoice [1/2/3]: ").strip()

    year_raw = input("Season year [2024]: ").strip()
    year     = int(year_raw) if year_raw.isdigit() else 2024

    exp = input("Export results? [y/N]: ").strip().lower() == "y"

    if choice == "1":
        name = input("Player name: ").strip()
        run_single(name, year, export=exp)

    elif choice == "2":
        raw   = input("Player names (comma-separated): ").strip()
        names = [n.strip() for n in raw.split(",") if n.strip()]
        run_comparison(names, year, export=exp)

    elif choice == "3":
        top_raw = input("Top N players [30]: ").strip()
        top_n   = int(top_raw) if top_raw.isdigit() else 30
        tier    = input("Filter tier? (leave blank for all): ").strip() or None
        run_leaderboard(year, top_n=top_n, tier=tier, export=exp)

    else:
        print("Invalid choice.")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full Hitter Analysis + CRAFT Fielding + Valuation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--name",        type=str,      help="Single player name")
    parser.add_argument("--compare",     nargs="+",     help="2+ player names for comparison")
    parser.add_argument("--leaderboard", action="store_true", help="Build league leaderboard")
    parser.add_argument("--year",        type=int,      default=2024)
    parser.add_argument("--top",         type=int,      default=30, help="Top N for leaderboard")
    parser.add_argument("--tier",        type=str,      default=None,
                        help="Filter leaderboard to tier: 'ELITE VALUE','GOOD VALUE','FAIR','OVERPAID','HIGHLY OVERPAID'")
    parser.add_argument("--sort",        type=str,      default="value_gap_M")
    parser.add_argument("--export",      action="store_true", help="Export results to files")

    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    if args.compare:
        run_comparison(args.compare, args.year, export=args.export)
    elif args.name:
        run_single(args.name, args.year, export=args.export)
    elif args.leaderboard:
        run_leaderboard(args.year, top_n=args.top, tier=args.tier,
                        sort_by=args.sort, export=args.export)
    else:
        interactive()
