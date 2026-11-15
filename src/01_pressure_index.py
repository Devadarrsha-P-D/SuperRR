"""Stage 1 - build ball-level match states, an out-of-fold win-probability (WP) model,
a leverage-based Pressure Index and directional pressure tiers.

Run:  python -I src/01_pressure_index.py
Output: outputs/balls.pkl   (one row per delivery, innings 1-2, clean matches only)
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "outputs"
SEED = 0
FEATS = ["innings", "balls_left", "wk", "cum", "req", "rrr", "crr", "is_ipl"]
FEEDER = ["Emerging Teams", "Rising Stars"]          # lower-level events, down-weighted later
LIVE_LO, LIVE_HI = 0.03, 0.97                         # outside this WP band the game is "decided"


def state_features(innings, cum, wk, balls_left, target, is_ipl):
    """Single source of truth for model features (used for real and hypothetical states)."""
    balls_left = np.clip(balls_left, 0, 120)
    req = np.where(innings == 2, target - cum, np.nan)
    rrr = np.where(innings == 2, req / (np.clip(balls_left, 1, None) / 6), np.nan)
    crr = cum / (np.clip(120 - balls_left, 1, None) / 6)
    return pd.DataFrame(dict(innings=innings, balls_left=balls_left, wk=wk, cum=cum,
                             req=req, rrr=rrr, crr=crr, is_ipl=is_ipl))


def load():
    b = pd.read_csv(RAW / "ball_by_ball.csv", low_memory=False)
    m = pd.read_csv(RAW / "matches.csv")
    c = pd.read_csv(RAW / "competitions.csv")
    m = m.merge(c[["comp_id", "comp_name", "match_format", "season"]], on="comp_id")
    return b, m


def build_states(b, m):
    b = b[b.innings.isin([1, 2])].copy()                       # drop super overs
    b["legal"] = (~b.extras_type.isin(["wide", "Overstep"])).astype(int)
    b["tot"] = b.runs + b.extras1_runs + b.extras_2_runs       # team runs off the ball
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    g = b.groupby(["match_id", "innings"])
    b["cum"] = g.tot.cumsum() - b.tot
    b["wk"] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
    b["balls_left"] = 120 - (g.legal.cumsum() - b.legal)
    tot1 = b[b.innings == 1].groupby("match_id").tot.sum().rename("tot1")
    meta = m[["match_id", "match_date", "match_format", "comp_name", "season", "win_team_id", "stage",
              "knock_out", "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]]
    b = b.merge(tot1, on="match_id").merge(meta, on="match_id")
    n0 = b.match_id.nunique()
    dls = b.first_inn_revised_overs.notna() | b.second_inn_revised_overs.notna() | b.revised_target.notna()
    b = b[~dls]
    b = b[(b.win_team_id == b.batting_team_id) | (b.win_team_id == b.bowling_team_id)].copy()  # drop ties/NR
    print(f"matches: {n0} with ball data -> {b.match_id.nunique()} after removing DLS/no-result/tied")
    b["target"] = np.where(b.innings == 2, b.tot1 + 1, np.nan)
    b["is_ipl"] = (b.match_format == "T20").astype(int)
    b["bat_win"] = (b.win_team_id == b.batting_team_id).astype(int)
    b["is_feeder"] = b.comp_name.str.contains("|".join(FEEDER)).astype(int)
    b["match_date"] = pd.to_datetime(b.match_date)
    return b.reset_index(drop=True)


def wp_and_leverage(b):
    """5-fold grouped-by-match WP model. Every ball is scored by a model that never saw its match."""
    X = state_features(b.innings.values, b["cum"].values, b.wk.values, b.balls_left.values,
                       b.target.values, b.is_ipl.values)
    y = b.bat_win.values
    # league outcome mix by phase -> used to compute expected |change in WP| of the next ball
    b["phase"] = np.where(b.overid <= 6, "PP", np.where(b.overid <= 15, "Mid", "Death"))
    b["out_cat"] = np.where(b.is_wicket, "W", b.tot.clip(upper=6).astype(str))
    mix = b.groupby("phase").out_cat.value_counts(normalize=True).unstack().fillna(0)
    wp0 = np.zeros(len(b)); lev = np.zeros(len(b))
    for tr, te in GroupKFold(5).split(X, y, b.match_id):
        clf = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=250,
                                             min_samples_leaf=200, l2_regularization=1.0, random_state=SEED)
        clf.fit(X.iloc[tr], y[tr])
        Xt = X.iloc[te]; p0 = clf.predict_proba(Xt)[:, 1]; wp0[te] = p0
        bt = b.iloc[te]
        for cat in mix.columns:
            r = 0 if cat == "W" else int(cat)
            nxt = state_features(bt.innings.values, bt["cum"].values + r, bt.wk.values + (cat == "W"),
                                 bt.balls_left.values - 1, bt.target.values, bt.is_ipl.values)
            lev[te] += bt.phase.map(mix[cat]).values * np.abs(clf.predict_proba(nxt)[:, 1] - p0)
    print(f"WP out-of-fold: AUC={roc_auc_score(y, wp0):.3f} (inn1 {roc_auc_score(y[b.innings==1], wp0[b.innings==1]):.3f}, "
          f"inn2 {roc_auc_score(y[b.innings==2], wp0[b.innings==2]):.3f})  logloss={log_loss(y, wp0):.3f}")
    b["wp"] = wp0; b["lev"] = lev
    # WP after the ball: next ball of same innings; last ball of inn2 = result; last ball of inn1 = 1 - first WP of inn2
    nxt = b.groupby(["match_id", "innings"]).wp.shift(-1)
    first2 = b[b.innings == 2].groupby("match_id").wp.first()
    last_inn1 = 1 - b.match_id.map(first2)
    is_last = nxt.isna()
    b["wp_after"] = np.where(is_last, np.where(b.innings == 2, b.bat_win, last_inn1), nxt)
    b["dwp"] = b.wp_after - b.wp
    return b


NON_BOWLER = ["RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"]


def add_run_values(b):
    """Run-value of every ball above what an average batter/bowler produces in the SAME match state.
    V = (runs - E[runs|s]) - C(s) * (wicket - E[wicket|s]); C(s) = expected runs lost when a wicket falls at s.
    Win probability is used only to DEFINE pressure; performance is scored in run value (see report)."""
    X = state_features(b.innings.values, b["cum"].values, b.wk.values, b.balls_left.values, b.target.values, b.is_ipl.values)
    Xf = X[["innings", "balls_left", "wk", "cum", "crr", "is_ipl"]]
    fut = b.groupby(["match_id", "innings"]).tot.transform("sum") - b["cum"]
    W = (b.is_wicket & ~b.dismissal_type.isin(NON_BOWLER)).astype(int).values
    ok = ~(b.is_wicket & b.dismissal_type.isin(NON_BOWLER)).values
    Er = np.zeros(len(b)); Ew = np.zeros(len(b)); C = np.zeros(len(b))
    for tr, te in GroupKFold(5).split(X, groups=b.match_id):
        trk = tr[ok[tr]]
        kw = dict(max_depth=5, learning_rate=0.06, max_iter=200, min_samples_leaf=200, l2_regularization=1.0, random_state=SEED)
        Er[te] = HistGradientBoostingRegressor(**kw).fit(X.iloc[trk], b.tot.values[trk]).predict(X.iloc[te])
        Ew[te] = HistGradientBoostingClassifier(**kw).fit(X.iloc[trk], W[trk]).predict_proba(X.iloc[te])[:, 1]
        f = HistGradientBoostingRegressor(**kw).fit(Xf.iloc[trk], fut.values[trk])
        a = Xf.iloc[te]; a1 = a.copy(); a1["wk"] = a1.wk + 1
        C[te] = np.clip(f.predict(a) - f.predict(a1), 0, None)
    b["E_runs"], b["E_w"], b["C"] = Er, Ew, C
    b["bowler_wkt"] = W
    b["V"] = (b.tot - Er) - C * (W - Ew)
    print(f"run-value: mean V={b.V.mean():+.4f}, mean wicket cost C={b.C.mean():.2f} runs")
    return b


def pressure_tiers(b):
    """Leverage percentile within innings among LIVE balls -> tiers, continuous intensity, direction."""
    b["live"] = (b.wp > LIVE_LO) & (b.wp < LIVE_HI)
    b["lev_pct"] = np.nan
    for i in (1, 2):
        mk = (b.innings == i) & b.live
        b.loc[mk, "lev_pct"] = b.loc[mk, "lev"].rank(pct=True)
    b["tier"] = pd.cut(b.lev_pct, [0, .60, .90, .97, 1.0001], labels=["Ordinary", "Elevated", "High", "Extreme"],
                       include_lowest=True).astype(object)
    b.loc[~b.live, "tier"] = "Decided"
    # innings-1 pressure is inferred from a weaker model -> half weight
    b["w_inn"] = np.where(b.innings == 1, 0.5, 1.0)
    ramp = ((b.lev_pct - 0.60) / (0.97 - 0.60)).clip(0, 1).fillna(0)
    P = ramp * b.w_inn
    # direction: batting side behind the game -> more pressure on batters; batting side ahead -> pressure on bowlers
    b["P_bat"] = (P * (0.6 + 0.8 * (1 - b.wp))).clip(0, 1)
    b["P_bowl"] = (P * (0.6 + 0.8 * b.wp)).clip(0, 1)
    b["P"] = P
    return b


if __name__ == "__main__":
    raw_b, m = load()
    b = pressure_tiers(add_run_values(wp_and_leverage(build_states(raw_b, m))))
    keep = ["match_id", "innings", "overid", "innings_ball_num", "batting_team_id", "bowling_team_id", "striker_id",
            "striker_name", "bowler_id", "bowler_name", "batting_order", "pace_or_spin", "batting_type",
            "runs", "tot", "legal", "extras_type", "is_four", "is_six", "is_wicket", "dismissal_type", "dismissed_player_id",
            "wp", "wp_after", "dwp", "V", "E_runs", "E_w", "C", "bowler_wkt", "lev", "lev_pct", "tier", "live", "P", "P_bat", "P_bowl", "w_inn",
            "phase", "match_date", "match_format", "comp_name", "season", "is_ipl", "is_feeder", "knock_out", "bat_win"]
    b[keep].to_pickle(OUT / "balls.pkl")
    t = b[b.legal == 1].copy()
    t["bdy"] = t.is_four | t.is_six
    t["dot"] = t.tot == 0
    tab = t.groupby("tier").agg(balls=("tot", "size"), runs_per_ball=("tot", "mean"), dot=("dot", "mean"),
                                boundary=("bdy", "mean"), wicket=("is_wicket", "mean"), mean_wp=("wp", "mean"),
                                share_chase=("innings", lambda s: (s == 2).mean())).round(3)
    tab = tab.reindex(["Ordinary", "Elevated", "High", "Extreme", "Decided"])
    print(tab); tab.to_csv(OUT / "tier_validation.csv")
