import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent)); import ratings as R
OUT = Path(__file__).resolve().parents[1] / "outputs"
d = R.prepare(pd.read_pickle(OUT / "balls.pkl")); cut = pd.Timestamp("2025-01-01")
d["yr"] = d.tot - d.E_runs; d["yw"] = -d.C * (d.bowler_wkt - d.E_w)
tr, te = d[d.match_date < cut].copy(), d[d.match_date >= cut]
bat_ids = np.sort(d.striker_id.unique()); bowl_ids = np.sort(d.bowler_id.unique())
def tgt(df, side, tiers, col):
    p = df[df.tier.isin(tiers)]
    if side == "bat": p = p[~p.is_wide]; return p.groupby("striker_id")[col].agg(n="size", v="mean")
    return p.groupby("bowler_id")[col].agg(n="size", v=lambda s: -s.mean())
out = []
for lam in (0, 0.25, 0.5, 1.0):
    tr["y"] = (tr.yr + lam * tr.yw).clip(-12, 12)
    c, _, _ = R.fit(tr, bat_ids, bowl_ids, 1000, 3.0, 3.0)
    for side, ids, bb, ss in (("bat", bat_ids, c["bat_base"], c["bat_slope"]), ("bowl", bowl_ids, c["bowl_base"], c["bowl_slope"])):
        ex = pd.DataFrame(dict(level=bb + ss), index=ids)
        for tn, tiers, mn in (("High+", ["High", "Extreme"], 20), ("Elev+", ["Elevated", "High", "Extreme"], 40)):
            for tcol in ("V", "yr"):
                te2 = te.copy(); te2["V"] = te2.yr + te2.yw
                T = tgt(te2, side, tiers, tcol); trn = tgt(tr, side, tiers, "y")
                j = ex.join(trn.n.rename("ntr")).join(T, how="inner"); j = j[(j.ntr >= 15) & (j.n >= mn)]
                out.append(dict(lam=lam, side=side, test=tn, target=tcol, n=len(j), rho=spearmanr(j.level, j.v)[0]))
o = pd.DataFrame(out); print(o.pivot_table(index=["side", "test", "target"], columns="lam", values="rho").round(3))
