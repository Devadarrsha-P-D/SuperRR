"""Pressure-adjusted ability model (crossed batter x bowler ridge model) + scoring helpers."""
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import Ridge

NON_BOWLER_DISMISSALS = ["RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"]
FIXED_SCALE = 30.0                 # large column scale == (almost) unpenalised
Y_CLIP = 12.0
LEV_FLOOR = 0.003
TIER_PRESSURE = ["High", "Extreme"]


def prepare(balls: pd.DataFrame) -> pd.DataFrame:
    """Rows used by the rating model: live states only, bowler-attributable dismissals only."""
    d = balls[balls.live].copy()
    d = d[~(d.is_wicket & d.dismissal_type.isin(NON_BOWLER_DISMISSALS))]
    d["y"] = d.V.clip(-Y_CLIP, Y_CLIP)                                      # run value above state-expected
    d["is_wide"] = d.extras_type.isin(["wide", "byes", "legbyes"])   # not off the bat -> no batter credit
    d["sw"] = np.where(d.is_feeder == 1, 0.5, 1.0)                           # feeder-event down-weight
    d["pmax"] = np.maximum(d.P_bat, d.P_bowl)
    d["slot"] = pd.cut(d.batting_order, [0, 2, 4, 6, 12], labels=False)
    d["spin"] = (d.pace_or_spin == "Spin").astype(int)
    return d.reset_index(drop=True)


def design(d, bat_ids, bowl_ids, f_slope):
    n = len(d); r = np.arange(n)
    bi = pd.Series(bat_ids).reset_index(drop=True); bi = pd.Series(np.arange(len(bi)), index=bi)
    ki = pd.Series(np.arange(len(bowl_ids)), index=bowl_ids)
    bc = d.striker_id.map(bi).values; kc = d.bowler_id.map(ki).values
    bm = (~d.is_wide).values.astype(float)           # batter is not credited/blamed for wides
    nb, nk = len(bat_ids), len(bowl_ids)
    mats = [
        sp.csr_matrix((bm, (r, bc)), shape=(n, nb)),                          # batter baseline
        sp.csr_matrix((f_slope * bm * d.P_bat.values, (r, bc)), shape=(n, nb)),  # batter pressure slope
        sp.csr_matrix((-np.ones(n), (r, kc)), shape=(n, nk)),                 # bowler baseline (sign flipped)
        sp.csr_matrix((-f_slope * d.P_bowl.values, (r, kc)), shape=(n, nk)),  # bowler pressure slope
    ]
    # role fixed effects: batting slot and pace/spin, baseline and x pressure
    slot = np.zeros((n, 4)); slot[r, d.slot.fillna(3).astype(int).values] = FIXED_SCALE * bm
    slotP = slot * d.P_bat.values[:, None]
    sp_ = np.c_[d.spin.values, 1 - d.spin.values] * FIXED_SCALE
    spP = sp_ * d.P_bowl.values[:, None]
    mats += [sp.csr_matrix(slot), sp.csr_matrix(slotP), sp.csr_matrix(-sp_), sp.csr_matrix(-spP)]
    return sp.hstack(mats, format="csr"), nb, nk


def fit(d, bat_ids, bowl_ids, alpha, f_slope, k_pw=0.0):
    X, nb, nk = design(d, bat_ids, bowl_ids, f_slope)
    m = Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg", tol=1e-4, max_iter=400)
    m.fit(X, d.y.values, sample_weight=(d.sw * (1 + k_pw * d.pmax)).values)
    c = m.coef_
    out = dict(bat_base=c[:nb], bat_slope=c[nb:2 * nb] * f_slope,
               bowl_base=c[2 * nb:2 * nb + nk], bowl_slope=c[2 * nb + nk:2 * nb + 2 * nk] * f_slope)
    return out, m, X


def predict(d, bat_ids, bowl_ids, coefs, f_slope, model):
    X, _, _ = design(d, bat_ids, bowl_ids, f_slope)
    return X @ model.coef_ + model.intercept_


def exposure_and_consistency(d, min_balls_match=3):
    """Pressure exposure (effective balls) and match-level hit rate on High/Extreme-tier balls."""
    pr = d[d.tier.isin(TIER_PRESSURE)]
    bat = pr[~pr.is_wide]
    bat_e = bat.groupby("striker_id").agg(p_balls=("w_inn", "sum"), p_matches=("match_id", "nunique"))
    bowl = pr[pr.legal == 1]
    bowl_e = bowl.groupby("bowler_id").agg(p_balls=("w_inn", "sum"), p_matches=("match_id", "nunique"))
    bm = bat.groupby(["striker_id", "match_id"]).agg(v=("y", "sum"), n=("y", "size")).reset_index()
    bm = bm[bm.n >= min_balls_match]
    km = pr.groupby(["bowler_id", "match_id"]).agg(v=("y", "sum"), n=("y", "size")).reset_index()
    km = km[km.n >= min_balls_match]; km["v"] = -km.v
    out = []
    for e, mm, key in ((bat_e, bm, "striker_id"), (bowl_e, km, "bowler_id")):
        base = (mm.v > 0).mean()
        h = mm.groupby(key).agg(hits=("v", lambda s: (s > 0).sum()), n_m=("v", "size"))
        K = 6
        h["cons"] = (h.hits + K * base) / (h.n_m + K)
        out.append((e.join(h, how="left").fillna({"n_m": 0, "hits": 0, "cons": base}), base))
    return out


def zs(s):
    return (s - s.mean()) / s.std(ddof=0)


def composite(level, delta, cons, w=(0.6, 0.2, 0.2)):
    return w[0] * zs(level) + w[1] * zs(delta) + w[2] * zs(cons)


# ---------------------------------------------------------------- shared scoring pipeline
HP = dict(alpha=1000.0, f=0.5, k=3.0)            # ridge strength, slope column scale, pressure-weight multiplier
W_DEFAULT = (1.0, 0.0, 0.0)                      # score = pressure rating (level); delta & consistency are diagnostics (see validation)
MIN_BAT_BALLS, MIN_BAT_MATCHES = 40, 8           # effective pressure balls / matches with pressure balls
MIN_BOWL_BALLS, MIN_BOWL_MATCHES = 36, 8


def eligible(d, mb=(MIN_BAT_BALLS, MIN_BAT_MATCHES), mk=(MIN_BOWL_BALLS, MIN_BOWL_MATCHES)):
    (be, _), (ke, _) = exposure_and_consistency(d)
    return dict(bat=be[(be.p_balls >= mb[0]) & (be.p_matches >= mb[1])].index,
                bowl=ke[(ke.p_balls >= mk[0]) & (ke.p_matches >= mk[1])].index)


def score(d, bat_ids, bowl_ids, elig, hp=HP, w=W_DEFAULT):
    """level = rating at full pressure (baseline+pressure slope); delta = pressure slope (vs own baseline);
    cons = shrunk share of pressure matches with positive net run value; score = weighted z-composite."""
    coefs, _, _ = fit(d, bat_ids, bowl_ids, hp["alpha"], hp["f"], hp["k"])
    (be, _), (ke, _) = exposure_and_consistency(d)
    bat = pd.DataFrame(dict(base=coefs["bat_base"], slope=coefs["bat_slope"]), index=pd.Index(bat_ids, name="player_id"))
    bowl = pd.DataFrame(dict(base=coefs["bowl_base"], slope=coefs["bowl_slope"]), index=pd.Index(bowl_ids, name="player_id"))
    res = []
    for tab, e, ids in ((bat, be, elig["bat"]), (bowl, ke, elig["bowl"])):
        t = tab.join(e, how="left").loc[tab.index.intersection(ids)].copy()
        t["level"] = t.base + t.slope; t["delta"] = t.slope
        t["cons"] = t.cons.fillna(t.cons.mean())
        t["score"] = composite(t.level, t.delta, t.cons, w)
        res.append(t)
    return res
