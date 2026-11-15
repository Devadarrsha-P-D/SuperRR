import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent)); import ratings as R
OUT = Path(__file__).resolve().parents[1] / "outputs"
d = R.prepare(pd.read_pickle(OUT / "balls.pkl")); cut = pd.Timestamp("2025-01-01")
tr, te = d[d.match_date < cut], d[d.match_date >= cut]
bat_ids = np.sort(d.striker_id.unique()); bowl_ids = np.sort(d.bowler_id.unique())
def tgt(df, side, tiers):
    p = df[df.tier.isin(tiers)]
    if side == "bat": p = p[~p.is_wide]; return p.groupby("striker_id").y.agg(n="size", v="mean")
    return p.groupby("bowler_id").y.agg(n="size", v=lambda s: -s.mean())
rows = []
for alpha in (300, 1000, 3000):
  for f in (1.0, 3.0):
    for k in (0.0, 3.0):
        c, _, _ = R.fit(tr, bat_ids, bowl_ids, alpha, f, k)
        for side, ids, bb, ss in (("bat", bat_ids, c["bat_base"], c["bat_slope"]), ("bowl", bowl_ids, c["bowl_base"], c["bowl_slope"])):
            ex = pd.DataFrame(dict(base=bb, slope=ss, level=bb + ss), index=ids)
            for tn, tiers, mn in (("High+", ["High", "Extreme"], 20), ("Elev+", ["Elevated", "High", "Extreme"], 40)):
                T = tgt(te, side, tiers); trn = tgt(tr, side, tiers)
                j = ex.join(trn.n.rename("ntr")).join(T, how="inner"); j = j[(j.ntr >= 15) & (j.n >= mn)]
                for comp in ("base", "slope", "level"):
                    rows.append(dict(alpha=alpha, f=f, k=k, side=side, test=tn, comp=comp, n=len(j), rho=spearmanr(j[comp], j.v)[0]))
r = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(r.pivot_table(index=["side", "test", "alpha", "f", "k"], columns="comp", values="rho").round(3).to_string())
print(r.groupby(["side", "test"]).n.first())
for side in ("bat", "bowl"):
    for tn, tiers, mn in (("High+", ["High", "Extreme"], 20), ("Elev+", ["Elevated", "High", "Extreme"], 40)):
        T = tgt(te, side, tiers); trn = tgt(tr, side, tiers); j = trn.join(T, lsuffix="_tr", how="inner"); j = j[(j.n_tr >= 15) & (j.n >= mn)]
        print(side, tn, "naive train pressure mean V ->", round(spearmanr(j.v_tr, j.v)[0], 3), "n", len(j))
