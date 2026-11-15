"""Stage 5 - compact JSON feed for the website and the PDF."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"
b = pd.read_pickle(OUT / "balls.pkl")
r = lambda x, n=3: None if pd.isna(x) else round(float(x), n)

def players(side):
    df = pd.read_csv(OUT / f"ratings_{'batters' if side=='bat' else 'bowlers'}_full.csv")
    rows = []
    for x in df.itertuples():
        d = dict(id=int(x.player_id), name=x.name, rank=int(x.rank), rating=r(x.rating_runs_per_over, 2), z=r(x.score, 2),
                 z10=r(x.score_p10, 2), z90=r(x.score_p90, 2), pt10=r(x.p_top10, 2), medrank=r(x.median_rank, 1),
                 pb=int(x.pressure_balls), pm=int(x.pressure_matches), cons=r(x.cons, 2), last=str(x.last_match),
                 naive=int(x.naive_rank_SR if side == "bat" else x.naive_rank_ECO))
        if side == "bat":
            d.update(p=r(x.SR_pressure, 1), o=r(x.SR_ordinary, 1), bdy=r(x.boundary_pct_pressure, 1), out=r(x.dismissal_rate_pressure, 1), rae=r(x.runs_above_exp_per_over_pressure, 2))
        else:
            d.update(p=r(x.ECO_pressure, 2), o=r(x.ECO_ordinary, 2), dot=r(x.dot_pct_pressure, 1), out=r(x.wicket_rate_pressure, 1), rae=r(x.runs_saved_per_over_pressure, 2))
        rows.append(d)
    return rows

# heat map: chase, live balls, mean leverage (win-prob points) by over x chasing-side win prob
x = b[b.live & (b.innings == 2)].copy()
labels = ["3-15", "15-30", "30-45", "45-55", "55-70", "70-85", "85-97"]
x["wpb"] = pd.cut(x.wp, [0.03, .15, .3, .45, .55, .7, .85, .97], labels=labels)
M = x.pivot_table(index="wpb", columns="overid", values="lev", aggfunc="mean", observed=False) * 100
N = x.pivot_table(index="wpb", columns="overid", values="lev", aggfunc="size", observed=False)
heat = [[(r(M.loc[l, o], 2) if N.loc[l, o] >= 30 else None) for o in range(1, 21)] for l in labels]
# state expectations by phase (for the "value of a ball" widget)
b["ph"] = np.where(b.overid <= 6, "Powerplay", np.where(b.overid <= 15, "Middle", "Death"))
ph = {k: dict(er=r(g.E_runs.mean(), 2), ew=r(g.E_w.mean() * 100, 1), C=r(g.C.mean(), 1)) for k, g in b[b.legal == 1].groupby("ph")}
T = pd.read_csv(OUT / "tier_validation.csv", index_col=0)
tiers = {k: dict(balls=int(v.balls), rpb=r(v.runs_per_ball, 2), dot=r(v["dot"] * 100, 1), bdy=r(v["boundary"] * 100, 1), wkt=r(v["wicket"] * 100, 1)) for k, v in T.iterrows()}
V = pd.read_csv(OUT / "validation_predictive.csv")
val = [dict(kind=v.kind, side=v.side, tier=v.test_tier, pred=v.predictor, rho=r(v.rho), n=r(v.n, 0)) for v in V.itertuples()
       if v.predictor in ("level", "naive_overall_mean", "naive_pressure_mean", "delta", "cons") and "role-adj" not in v.test_tier]
S = pd.read_csv(OUT / "validation_sensitivity.csv"); sens = S.round(3).to_dict("records")
meta = dict(matches=int(b.match_id.nunique()), balls=int(len(b)), ipl=int(b[b.is_ipl == 1].match_id.nunique()), t20i=int(b[b.is_ipl == 0].match_id.nunique()),
            first=str(b.match_date.min().date()), last=str(b.match_date.max().date()))
data = dict(meta=meta, bat=players("bat"), bowl=players("bowl"), heat=heat, heat_rows=labels, phases=ph, tiers=tiers, val=val, sens=sens)
(ROOT / "site" / "data.json").write_text(json.dumps(data, separators=(",", ":")))
print(meta, ph, len(json.dumps(data)))
