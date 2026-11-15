"""Stage 2 - fit pressure-adjusted ratings on all data, bootstrap uncertainty by match, rank.
Run: python -I src/02_ratings.py [--boot 100]
Outputs: outputs/ratings_batters.csv, ratings_bowlers.csv
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import ratings as R

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"
N_BOOT = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 100

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = R.prepare(pd.read_pickle(OUT / "balls.pkl"))
    bat_ids = np.sort(d.striker_id.unique()); bowl_ids = np.sort(d.bowler_id.unique())
    elig = R.eligible(d)
    print("rating rows:", len(d), "| eligible batters:", len(elig["bat"]), "bowlers:", len(elig["bowl"]), "| hp:", R.HP)
    bat, bowl = R.score(d, bat_ids, bowl_ids, elig)
    # bootstrap over matches (whole pipeline re-run each time; eligibility fixed from the full sample)
    ids = d.match_id.unique(); groups = {m: g for m, g in d.groupby("match_id")}
    bs_b, bs_k = [], []
    for i in range(N_BOOT):
        pick = rng.choice(ids, len(ids), replace=True)
        rb, rk = R.score(pd.concat([groups[m] for m in pick], ignore_index=True), bat_ids, bowl_ids, elig)
        bs_b.append(rb.score.rename(i)); bs_k.append(rk.score.rename(i))
        if i % 20 == 0: print("boot", i, flush=True)
    for tab, bs in ((bat, bs_b), (bowl, bs_k)):
        B = pd.concat(bs, axis=1).reindex(tab.index); ranks = B.rank(ascending=False, axis=0)
        tab["score_sd"] = B.std(axis=1); tab["score_p10"] = B.quantile(.10, axis=1); tab["score_p90"] = B.quantile(.90, axis=1)
        tab["p_top10"] = (ranks <= 10).mean(axis=1); tab["median_rank"] = ranks.median(axis=1)
    nm_b = d.drop_duplicates("striker_id").set_index("striker_id").striker_name
    nm_k = d.drop_duplicates("bowler_id").set_index("bowler_id").bowler_name
    bat["name"] = nm_b.reindex(bat.index); bowl["name"] = nm_k.reindex(bowl.index)
    bat["last_match"] = d.groupby("striker_id").match_date.max().reindex(bat.index).dt.date
    bowl["last_match"] = d.groupby("bowler_id").match_date.max().reindex(bowl.index).dt.date
    for tab, key, fn in ((bat, "striker_id", "ratings_batters.csv"), (bowl, "bowler_id", "ratings_bowlers.csv")):
        tab.sort_values("score", ascending=False, inplace=True); tab["rank"] = range(1, len(tab) + 1)
        tab.to_csv(OUT / fn)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["name", "rank", "level", "delta", "cons", "score", "score_p10", "p_top10", "median_rank", "p_balls", "p_matches", "last_match"]
    print("\nTOP BATTERS\n", bat[cols].head(15).round(3).to_string()); print("\nTOP BOWLERS\n", bowl[cols].head(15).round(3).to_string())
