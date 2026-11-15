import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent)); import ratings as R
OUT = Path(__file__).resolve().parents[1] / "outputs"
b = pd.read_pickle(OUT / "balls.pkl"); d = b[b.live & ~(b.is_wicket & b.dismissal_type.isin(R.NON_BOWLER_DISMISSALS))].copy()
d["is_wide"] = d.extras_type.isin(["wide"]); cut = pd.Timestamp("2025-01-01")
print("mean V by tier (bat side):", d.groupby("tier").V.mean().round(3).to_dict())
def pers(filt, col, key, minn, label, sign=1):
    a, bb = d[d.match_date < cut], d[d.match_date >= cut]
    f = lambda x: filt(x).groupby(key)[col].agg(m="mean", n="size")
    j = f(a).join(f(bb), lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.n_a >= minn) & (j.n_b >= minn)]
    print(f"{label:48s} n={len(j):3d} rho={spearmanr(j.m_a, j.m_b)[0]:+.3f}")
nw = lambda x: x[~x.is_wide]
E = ["Elevated", "High", "Extreme"]; H = ["High", "Extreme"]
for lab, tiers, mn in (("all-live", None, 40), ("Elev+", E, 15), ("High+", H, 10)):
    ft = (lambda x, t=tiers: x if t is None else x[x.tier.isin(t)])
    pers(lambda x: nw(ft(x)), "V", "striker_id", mn, f"BAT {lab} V/ball (min {mn})")
    pers(lambda x: ft(x), "V", "bowler_id", mn, f"BOWL {lab} V/ball (min {mn})")
