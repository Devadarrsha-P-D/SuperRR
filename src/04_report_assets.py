"""Stage 4 - evidence tables and figures for the report."""
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent)); import ratings as R
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; FIG = OUT / "figures"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
b = pd.read_pickle(OUT / "balls.pkl"); d = R.prepare(b)
bat = pd.read_csv(OUT / "ratings_batters.csv").rename(columns={"Unnamed: 0": "player_id"}); bowl = pd.read_csv(OUT / "ratings_bowlers.csv").rename(columns={"Unnamed: 0": "player_id"})

# ---------- evidence tables
H = ["High", "Extreme"]
def bat_stats(pid):
    x = d[(d.striker_id == pid) & ~d.is_wide]; p = x[x.tier.isin(H)]; o = x[x.tier.isin(["Ordinary"])]
    sr = lambda z: z.runs.sum() / max(len(z), 1) * 100
    return dict(pressure_balls=len(p), pressure_matches=p.match_id.nunique(), SR_pressure=sr(p), SR_ordinary=sr(o),
                boundary_pct_pressure=(p.is_four | p.is_six).mean() * 100, dismissal_rate_pressure=p.bowler_wkt.mean() * 100,
                runs_above_exp_per_over_pressure=p.y.mean() * 6)
def bowl_stats(pid):
    x = d[d.bowler_id == pid]; p = x[x.tier.isin(H)]; o = x[x.tier.isin(["Ordinary"])]
    eco = lambda z: z.tot.sum() / max(z.legal.sum(), 1) * 6
    return dict(pressure_balls=int(p.legal.sum()), pressure_matches=p.match_id.nunique(), ECO_pressure=eco(p), ECO_ordinary=eco(o),
                dot_pct_pressure=((p.tot == 0) & (p.legal == 1)).sum() / max(p.legal.sum(), 1) * 100,
                wicket_rate_pressure=p.bowler_wkt.sum() / max(p.legal.sum(), 1) * 100, runs_saved_per_over_pressure=-p.y.mean() * 6)
# naive rankings among the same eligible pool
bs = pd.DataFrame({pid: bat_stats(pid) for pid in bat.player_id}).T
ks = pd.DataFrame({pid: bowl_stats(pid) for pid in bowl.player_id}).T
bat = bat.set_index("player_id").join(bs); bowl = bowl.set_index("player_id").join(ks)
bat["naive_rank_SR"] = bat.SR_pressure.rank(ascending=False).astype(int); bowl["naive_rank_ECO"] = bowl.ECO_pressure.rank().astype(int)
bat["rating_runs_per_over"] = bat.level * 6; bowl["rating_runs_per_over"] = bowl.level * 6
cols_b = ["rank", "name", "rating_runs_per_over", "score", "score_p10", "score_p90", "p_top10", "median_rank", "pressure_balls", "pressure_matches",
          "SR_pressure", "SR_ordinary", "boundary_pct_pressure", "dismissal_rate_pressure", "runs_above_exp_per_over_pressure", "cons", "naive_rank_SR", "last_match"]
cols_k = ["rank", "name", "rating_runs_per_over", "score", "score_p10", "score_p90", "p_top10", "median_rank", "pressure_balls", "pressure_matches",
          "ECO_pressure", "ECO_ordinary", "dot_pct_pressure", "wicket_rate_pressure", "runs_saved_per_over_pressure", "cons", "naive_rank_ECO", "last_match"]
tb = bat.sort_values("rank")[cols_b].head(10); tk = bowl.sort_values("rank")[cols_k].head(10)
tb.round(3).to_csv(OUT / "top10_batters.csv", index=False); tk.round(3).to_csv(OUT / "top10_bowlers.csv", index=False)
bat.sort_values("rank").round(3).to_csv(OUT / "ratings_batters_full.csv"); bowl.sort_values("rank").round(3).to_csv(OUT / "ratings_bowlers_full.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(tb.round(2).to_string()); print(tk.round(2).to_string())
nb = set(bat.nsmallest(10, "naive_rank_SR").name); nk = set(bowl.nsmallest(10, "naive_rank_ECO").name)
print("overlap with naive top10 (pressure SR / economy): bat", len(nb & set(tb.name)), "bowl", len(nk & set(tk.name)))
print("naive top10 bat:", sorted(nb)); print("naive top10 bowl:", sorted(nk))

# ---------- fig1 tier validation
T = pd.read_csv(OUT / "tier_validation.csv", index_col=0).loc[["Ordinary", "Elevated", "High", "Extreme"]]
fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.0))
for a, (col, ttl) in zip(ax, (("dot", "Dot-ball share (%)"), ("boundary", "Boundary share (%)"), ("wicket", "Wicket rate (% of balls)"))):
    v = T[col] * 100; cols = [BLUE if t in ("High", "Extreme") else "#9db8dc" for t in T.index]
    a.bar(T.index, v, color=cols, width=0.62); a.set_title(ttl, fontsize=10, color=INK, loc="left")
    for i, x in enumerate(v): a.text(i, x + v.max() * 0.015, f"{x:.1f}", ha="center", fontsize=9, color=INK)
    a.set_ylim(0, v.max() * 1.18); a.tick_params(axis="x", labelsize=8.5); a.grid(axis="y", color=GRID, lw=.6); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / "fig1_pressure_tiers.png", dpi=200); plt.close(fig)

# ---------- fig2 top-10 forest
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
for a, t, ttl in ((ax[0], tb, "Batters"), (ax[1], tk, "Bowlers")):
    y = np.arange(len(t))[::-1]
    a.hlines(y, t.score_p10, t.score_p90, color="#9db8dc", lw=3); a.scatter(t.score, y, s=42, color=BLUE, zorder=3, edgecolor="white", lw=1.5)
    a.set_yticks(y); a.set_yticklabels([f"{r}. {n}" for r, n in zip(t["rank"], t.name)], fontsize=9, color=INK)
    a.axvline(0, color=GRID, lw=1); a.set_title(f"{ttl}: rating (z), 10th-90th pct range", fontsize=9, color=INK, loc="left")
    a.grid(axis="x", color=GRID, lw=.6); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / "fig2_top10_uncertainty.png", dpi=200); plt.close(fig)

# ---------- fig3 predictive validity
V = pd.read_csv(OUT / "validation_predictive.csv"); V = V[(V.kind == "random-split mean")]
lab = {"naive_pressure_mean": "Raw pressure-only average", "delta": "Pressure delta vs own baseline", "cons": "Hit-rate consistency",
       "level": "Pressure rating (ours)", "naive_overall_mean": "Raw overall average", "base": "Baseline skill (ridge)"}
order = ["naive_pressure_mean", "cons", "delta", "level", "naive_overall_mean"]
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4), sharex=True)
for a, side, ttl in ((ax[0], "bat", "Batters"), (ax[1], "bowl", "Bowlers")):
    s = V[(V.side == side) & (V.test_tier == "Elev+")].set_index("predictor").rho.reindex(order)
    y = np.arange(len(order))[::-1]
    a.barh(y, s.values, color=[ORANGE if p == "level" else "#9db8dc" if p != "naive_overall_mean" else "#b8b7b0" for p in order], height=0.55)
    a.set_yticks(y); a.set_yticklabels([lab[p] for p in order], fontsize=9, color=INK); a.axvline(0, color=MUTED, lw=.8)
    for yy, v in zip(y, s.values): a.text(v + (0.008 if v >= 0 else -0.008), yy, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    a.set_title(f"{ttl}: rank corr. with held-out results", fontsize=9, color=INK, loc="left"); a.grid(axis="x", color=GRID, lw=.6); a.set_axisbelow(True)
    a.set_xlim(-0.16, 0.38)
fig.tight_layout(); fig.savefig(FIG / "fig3_predictive_validity.png", dpi=200); plt.close(fig)

# ---------- fig4 leverage heatmap (over x win-prob)
x = b[b.live & (b.innings == 2)].copy(); x["wpb"] = pd.cut(x.wp, [0.03, .15, .3, .45, .55, .7, .85, .97], labels=["3-15", "15-30", "30-45", "45-55", "55-70", "70-85", "85-97"])
M = x.pivot_table(index="wpb", columns="overid", values="lev", aggfunc="mean", observed=False) * 100
fig, a = plt.subplots(figsize=(9.2, 3.3)); im = a.imshow(M.values, aspect="auto", cmap="Blues", origin="lower")
a.set_xticks(range(M.shape[1])); a.set_xticklabels(M.columns, fontsize=8); a.set_yticks(range(M.shape[0])); a.set_yticklabels(M.index, fontsize=8)
a.set_xlabel("Over of the chase"); a.set_ylabel("Chasing side win prob. (%)"); a.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=a, pad=0.01); cb.set_label("Expected win-prob swing of next ball (pts)", fontsize=8)
a.set_title("Where the pressure sits in a chase: late overs, close games", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig4_pressure_map.png", dpi=200); plt.close(fig)
print("figures done")
