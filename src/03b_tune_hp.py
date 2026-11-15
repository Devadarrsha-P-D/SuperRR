"""Tune ridge strength (alpha) and pressure re-weighting (k) by held-out predictive accuracy (random 50/50 match splits)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent)); import ratings as R
OUT = Path(__file__).resolve().parents[1] / "outputs"
d = R.prepare(pd.read_pickle(OUT / "balls.pkl"))
bat_ids = np.sort(d.striker_id.unique()); bowl_ids = np.sort(d.bowler_id.unique())
def tgt(df, side, tiers):
    p = df[df.tier.isin(tiers)]
    if side == "bat": return p[~p.is_wide].groupby("striker_id").y.agg(n="size", v="mean")
    return p.groupby("bowler_id").y.agg(n="size", v=lambda s: -s.mean())
rng = np.random.default_rng(7); mids = d.match_id.unique(); rows = []
for r in range(8):
    half = set(rng.choice(mids, len(mids) // 2, replace=False)); A = d[d.match_id.isin(half)]; B = d[~d.match_id.isin(half)]
    for alpha in (300, 1000, 3000):
        for k in (0.0, 3.0, 8.0):
            c, _, _ = R.fit(A, bat_ids, bowl_ids, alpha, 0.5, k)
            for side, ids, base in (("bat", bat_ids, c["bat_base"]), ("bowl", bowl_ids, c["bowl_base"])):
                ex = pd.Series(base, index=ids)
                for tn, tiers, mn in (("Elev+", ["Elevated", "High", "Extreme"], 40), ("High+", ["High", "Extreme"], 20)):
                    T = tgt(B, side, tiers); nA = tgt(A, side, tiers).n.rename("nA")
                    j = T.join(ex.rename("b")).join(nA); j = j[(j.n >= mn) & (j.nA >= 15)]
                    rows.append(dict(alpha=alpha, k=k, side=side, tier=tn, rho=spearmanr(j.b, j.v)[0]))
t = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(t.pivot_table(index=["alpha", "k"], columns=["side", "tier"], values="rho").round(3).assign(mean=lambda x: x.mean(axis=1)).round(3))
