"""Stage 3 - out-of-sample validation + robustness.
(a) temporal: fit on <2025, test on >=2025;  (b) repeated random 50/50 splits by match;  (c) sensitivity of top-10 to settings.
Outputs: outputs/validation_predictive.csv, validation_sensitivity.csv"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent)); import ratings as R
OUT = Path(__file__).resolve().parents[1] / "outputs"
d = R.prepare(pd.read_pickle(OUT / "balls.pkl"))
bat_ids = np.sort(d.striker_id.unique()); bowl_ids = np.sort(d.bowler_id.unique())
TIERS = {"High+": ["High", "Extreme"], "Elev+": ["Elevated", "High", "Extreme"]}
MIN_TEST = {"High+": 20, "Elev+": 40}

def tgt(df, side, tiers, adj=False):
    p = df[df.tier.isin(tiers)].copy()
    if side == "bat": p = p[~p.is_wide]
    if adj:   # remove structural role effect: batting slot (bat) / pace-vs-spin (bowl)
        key = "slot" if side == "bat" else "spin"; p["y"] = p.y - p.groupby(key).y.transform("mean")
    if side == "bat": return p.groupby("striker_id").y.agg(n="size", v="mean")
    return p.groupby("bowler_id").y.agg(n="size", v=lambda s: -s.mean())

def evaluate(train, test, label, rows):
    elig = R.eligible(train, mb=(25, 5), mk=(20, 5))          # looser eligibility inside half-samples
    rb, rk = R.score(train, bat_ids, bowl_ids, elig)
    for side, t in (("bat", rb), ("bowl", rk)):
        naive = tgt(train, side, TIERS["High+"]).v.rename("naive_pressure_mean")
        allm = tgt(train, side, ["Ordinary", "Elevated", "High", "Extreme"]).v.rename("naive_overall_mean")
        for tn, tiers in TIERS.items():
            for adj in (False, True):
                T = tgt(test, side, tiers, adj)
                j = t.join(T, how="inner").join(naive).join(allm); j = j[j.n >= MIN_TEST[tn]]
                comps = dict(base=j.base, level=j.level, delta=j.delta, cons=j.cons, composite=j.score,
                             c_level_only=R.composite(j.level, j.delta, j.cons, (1, 0, 0)),
                             c_85_15=R.composite(j.level, j.delta, j.cons, (0.85, 0, 0.15)),
                             c_70_30=R.composite(j.level, j.delta, j.cons, (0.70, 0, 0.30)),
                             naive_pressure_mean=j.naive_pressure_mean, naive_overall_mean=j.naive_overall_mean)
                for c, sr in comps.items():
                    rows.append(dict(split=label, side=side, test_tier=tn + ("/role-adj" if adj else ""), predictor=c, n=len(j),
                                     rho=spearmanr(sr, j.v, nan_policy="omit")[0]))

rows = []
cut = pd.Timestamp("2025-01-01")
evaluate(d[d.match_date < cut], d[d.match_date >= cut], "temporal(<2025 -> >=2025)", rows)
rng = np.random.default_rng(1); mids = d.match_id.unique()
for r in range(10):
    half = set(rng.choice(mids, len(mids) // 2, replace=False))
    a = d[d.match_id.isin(half)]; b = d[~d.match_id.isin(half)]
    evaluate(a, b, f"random{r}", rows)
V = pd.DataFrame(rows); V.to_csv(OUT / "validation_predictive_raw.csv", index=False)
V["kind"] = np.where(V.split.str.startswith("temporal"), "temporal", "random-split mean")
S = V.groupby(["kind", "side", "test_tier", "predictor"]).agg(rho=("rho", "mean"), rho_sd=("rho", "std"), n=("n", "mean")).round(3)
S.to_csv(OUT / "validation_predictive.csv"); pd.set_option("display.width", 200)
print(S.unstack("predictor")["rho"].to_string())


if '--sens' in sys.argv:
    # sensitivity of final top-10 to hyper-parameters and composite weights
    elig = R.eligible(d); base = R.score(d, bat_ids, bowl_ids, elig); sens = []
    for lab, hp, w in (("default", R.HP, R.W_DEFAULT), ("alpha x3", {**R.HP, "alpha": 3000}, R.W_DEFAULT), ("alpha /3", {**R.HP, "alpha": 333}, R.W_DEFAULT),
                       ("no pressure re-weight (k=0)", {**R.HP, "k": 0.0}, R.W_DEFAULT), ("slope f=1", {**R.HP, "f": 1.0}, R.W_DEFAULT),
                       ("level only", R.HP, (1, 0, 0)), ("0.8/0/0.2", R.HP, (0.8, 0, 0.2)), ("0.5/0.25/0.25", R.HP, (0.5, 0.25, 0.25))):
        cur = R.score(d, bat_ids, bowl_ids, elig, hp, w)
        for side, b0, c0 in (("bat", base[0], cur[0]), ("bowl", base[1], cur[1])):
            t0 = set(b0.score.nlargest(10).index); t1 = set(c0.score.nlargest(10).index)
            sens.append(dict(setting=lab, side=side, top10_overlap=len(t0 & t1), rho_full_ranking=spearmanr(b0.score, c0.score.reindex(b0.index))[0]))
    Sn = pd.DataFrame(sens); Sn.to_csv(OUT / "validation_sensitivity.csv", index=False); print(Sn.round(3).to_string())
