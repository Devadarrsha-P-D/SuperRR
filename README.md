# SupeRR Selector S2 - Round 1: Most reliable players under pressure

Pipeline that defines pressure from a win-probability model, scores every delivery in run value above what the
match state predicts, and ranks batters and bowlers with a pressure-adjusted ridge model.

## Run (≈10 min)
```
pip install -r requirements.txt
python -I src/01_pressure_index.py      # states, win-prob model, leverage, pressure tiers, run values -> outputs/balls.pkl
python -I src/02_ratings.py --boot 100  # ridge ratings + bootstrap -> outputs/ratings_*.csv
python -I src/03_validate.py --sens     # held-out validation + sensitivity -> outputs/validation_*.csv
python -I src/04_report_assets.py       # top-10 tables + figures
```
Data goes in `data/raw/` (the six CSVs from the data pack).

## Method in one paragraph
1. **Pressure Index** - a gradient-boosted win-probability model (5-fold, grouped by match, so no ball is scored by a model that saw its
   match) gives each ball a *leverage* = expected |change in win probability| of the next delivery. Leverage is ranked within innings among
   live states (win prob 3-97%) and cut into Ordinary (<60th pct), Elevated (60-90), High (90-97), Extreme (97+). First-innings
   pressure is half-weighted (weaker model). A mild directional tilt puts more pressure on batters when their side is behind and on
   bowlers when the batting side is ahead.
2. **Performance** - run value above state-expectation: `V = (runs - E[runs|state]) - C(state)*(wicket - E[wicket|state])`, with C the
   expected runs lost to a wicket. Win probability only *defines* pressure; it does not score players (batter win-prob credit
   drifted -1.0 pts/100 balls in the Extreme tier and barely persisted, rho~0.1, so it was dropped as a scoring metric).
3. **Rating** - ridge model with crossed batter and bowler effects (each player: baseline + pressure slope), batting-slot and pace/spin
   role effects, feeder events at 0.5 weight, pressure balls up-weighted. Rating = baseline + shrunk pressure slope, in runs/ball above expected.
4. **Uncertainty** - 100 match-level bootstraps of the whole pipeline: score percentile band, P(top-10), median rank.
5. **Validation** - rating is checked against held-out pressure results (temporal 2022-24 -> 2025-26 and 10 random 50/50 match splits).

## Key honest finding
Pressure-specific components (slope vs own baseline, hit-rate consistency) did **not** predict held-out pressure performance; baseline
skill estimated on role- and opposition-adjusted, pressure-weighted balls did. Final score is therefore the pressure rating only;
delta and consistency are reported as diagnostics. See `outputs/validation_predictive.csv`.

## Files
- `outputs/top10_batters.csv`, `top10_bowlers.csv` - final lists with evidence columns
- `outputs/ratings_*_full.csv` - all eligible players (88 batters, 97 bowlers)
- `outputs/figures/` - report figures
- `src/exploration/` - scratch diagnostics behind design decisions
