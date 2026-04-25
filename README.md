# ⚾ Hitter Analysis Pipeline

A modular data science project for full position player analysis, combining Statcast hitting metrics, CRAFT defensive ratings, and a salary-based valuation model.

---

## Project Structure

```
hitter_analysis/
├── main.py              # CLI entry point — run everything from here
├── data_pull.py         # Raw data fetching (FanGraphs + Statcast via pybaseball)
├── data_clean.py        # Normalisation, merging, derived columns
├── analysis.py          # Scoring, grading, CRAFT integration
├── valuation.py         # Market salary estimation + over/undervalue ranking
├── output.py            # Terminal display (rich) + CSV/Markdown export
├── data/
│   └── salaries_manual.csv   # Optional: manually curated salary overrides
└── outputs/             # Auto-created on first run
```

---

## Setup

```bash
pip install pybaseball pandas numpy scipy scikit-learn rich tabulate
```

**CRAFT Integration** — run the CRAFT pipeline first so fielding scores are available:

```bash
# In your CRAFT project directory
python 01_pull_data.py --season 2024
python 02_feature_engineering.py --season 2024
python 03_range_component.py --season 2024
```

Then point `analysis.py` at your CRAFT output directory by setting `CRAFT_SCORES_DIR`.
Default path: `craft_output/scores/craft_range_{year}.csv`

---

## Usage

### Single player card
```bash
python main.py --name "Aaron Judge" --year 2024
python main.py --name "Aaron Judge" --year 2024 --export
```

### Side-by-side comparison
```bash
python main.py --compare "Aaron Judge" "Juan Soto" "Mookie Betts" --year 2024
```

### League-wide valuation leaderboard
```bash
python main.py --leaderboard --year 2024 --top 30
python main.py --leaderboard --year 2024 --tier "ELITE VALUE"
python main.py --leaderboard --year 2024 --sort total_score --export
```

### Interactive mode
```bash
python main.py
```

---

## Metrics Covered

### Hitting
| Category | Metrics |
|---|---|
| Traditional | G, PA, AB, H, HR, R, RBI, SB, AVG, OBP, SLG, OPS |
| Advanced | wOBA, wRC+, BB%, K%, ISO, BABIP, WAR, Off, Def |
| Statcast contact | Avg EV, Max EV, Avg LA, Hard Hit%, Barrel%, Barrels/PA |
| Expected stats | xBA, xSLG, xwOBA, luck differentials vs actuals |
| Speed | Sprint speed (ft/s), competitive runs |
| Plate discipline | BB%, K%, discipline composite |

### Fielding (CRAFT — Phase 1)
| Metric | Description |
|---|---|
| `craft_range_runs_saved` | Runs saved above average from Range Component |
| `craft_range_per_150` | Rate stat per 150 games |
| `craft_position` | Primary defensive position |
| `craft_opportunities` | Total batted balls fielded |

> CRAFT Phases 2–4 (Arm, Reliability, Positioning) will be integrated as they are published.

### Composite Scores
| Score | Method |
|---|---|
| Hitting Score (0-100) | Weighted composite: wRC+ (30%), xwOBA (20%), Barrel% (15%), BB% (10%), K% (10%), Hard Hit% (10%), Speed (5%) |
| Fielding Score (0-100) | CRAFT range_runs_saved normalised to 0-100 (avg=50) |
| Total Score (0-100) | 80% Hitting + 20% Fielding (CRAFT weight increases as more phases are added) |

### 20-80 Scouting Grades
Grades for: wRC+, K%, BB%, Hard Hit%, Barrel%, xwOBA, Sprint Speed, Avg EV, ISO, wOBA

---

## Valuation Model

### Approach
1. Train a **HuberRegressor** (robust to outliers) on players with known salaries
2. Features: WAR, wRC+, xwOBA, Barrel%, CRAFT Range Runs Saved
3. Estimate market salary for every qualifying player
4. Compute **Value Gap** = Estimated Salary − Actual Salary

### Value Tiers
| Label | Value Gap |
|---|---|
| 🔵 ELITE VALUE | > +$10M |
| 🟢 GOOD VALUE | +$3M to +$10M |
| 🟡 FAIR | -$3M to +$3M |
| 🔴 OVERPAID | -$10M to -$3M |
| 🔴 HIGHLY OVERPAID | < -$10M |

### Salary Data
- Primary: Lahman salary table (via pybaseball) — lags ~1 year
- Supplement: `data/salaries_manual.csv` (name, salary, year) for current season
- Fallback: $/WAR heuristic (~$9.5M/WAR for 2024)

---

## CRAFT Integration Notes

The CRAFT `player_id` column is a Statcast MLBAM ID — the same `mlbam_id` used throughout this pipeline, so the join is direct with no crosswalk needed.

If a player has multiple defensive positions in CRAFT (e.g., utility OF/1B), the position with the highest opportunity count is used as their primary fielding profile.

The fielding score weight in Total Player Score is currently **20%** (conservative, reflecting that CRAFT Phase 1 covers Range only). This weight is configurable in `analysis.py`:

```python
CRAFT_WEIGHT = 0.20          # fielding share of total score
HITTING_TOTAL_WEIGHT = 0.80  # hitting share
```

---

## Outputs

When `--export` is passed:
- `outputs/{Player}_{year}_profile.csv` — full stat row
- `outputs/{Player}_{year}_report.md` — formatted markdown report
- `outputs/valuation_leaderboard_{year}.csv` — full league valuation table
