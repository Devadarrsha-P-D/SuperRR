"""Stage 6 - build the Unstop methodology PDF (HTML -> PDF via Playwright).
Run: python -I src/06_build_pdf.py   (needs: playwright + chromium; outputs report/SupeRR_R1_Methodology_Report.pdf)"""
import json, base64
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]; D = json.loads((ROOT / "site/data.json").read_text())
FIG = ROOT / "outputs/figures_pdf"; OUT = ROOT / "report"
img = lambda n: "data:image/png;base64," + base64.b64encode((FIG / n).read_bytes()).decode()
sgn = lambda x, d=2: ("+" if x >= 0 else "−") + f"{abs(x):.{d}f}"

def val(kind, side, tier, pred):
    for v in D["val"]:
        if v["kind"] == kind and v["side"] == side and v["tier"] == tier and v["pred"] == pred: return v["rho"]

bat_rows = "".join(f"<tr><td class=n>{p['rank']}</td><td><b>{p['name']}</b></td><td class=r>{sgn(p['rating'])}</td><td class=r>{p['p']:.0f} / {p['o']:.0f}</td><td class=r>{round(p['pt10']*100)}%</td></tr>" for p in D["bat"][:10])
bowl_rows = "".join(f"<tr><td class=n>{p['rank']}</td><td><b>{p['name']}</b></td><td class=r>{sgn(p['rating'])}</td><td class=r>{p['p']:.2f} / {p['o']:.2f}</td><td class=r>{p['out']:.1f}%</td><td class=r>{round(p['pt10']*100)}%</td></tr>" for p in D["bowl"][:10])
T = D["tiers"]
tier_rows = "".join(f"<tr><td><b>{k}</b></td><td class=r>{T[k]['balls']:,}</td><td class=r>{T[k]['rpb']:.2f}</td><td class=r>{T[k]['dot']:.1f}%</td><td class=r>{T[k]['bdy']:.1f}%</td><td class=r>{T[k]['wkt']:.1f}%</td></tr>" for k in ["Ordinary", "Elevated", "High", "Extreme"])
preds = [("level", "Pressure rating (our ranking)"), ("naive_overall_mean", "Raw overall run value (all balls)"), ("naive_pressure_mean", "Raw pressure-only average"), ("delta", "Change from own baseline"), ("cons", "Hit-rate consistency")]
def cell(side, tier, pred):
    a, t = val("random-split mean", side, tier, pred), val("temporal", side, tier, pred)
    return f"<td class=r>{sgn(a)} <span class=sm>({sgn(t)})</span></td>"
val_rows = "".join(f"<tr class='{'hl' if k=='level' else ''}'><td class=w>{lab}</td>{cell('bat','Elev+',k)}{cell('bat','High+',k)}{cell('bowl','Elev+',k)}{cell('bowl','High+',k)}</tr>" for k, lab in preds)
S = {}
for r in D["sens"]: S.setdefault(r["setting"], {})[r["side"]] = r["top10_overlap"]
lab = {"alpha x3": "Ridge strength x3", "alpha /3": "Ridge strength / 3", "no pressure re-weight (k=0)": "No pressure up-weighting", "slope f=1": "Looser pressure slope", "0.8/0/0.2": "Add 20% consistency to the score", "0.5/0.25/0.25": "Add delta and consistency (50/25/25)"}
sens_rows = "".join(f"<tr><td class=w>{lab[k]}</td><td class=r>{S[k]['bat']}/10</td><td class=r>{S[k]['bowl']}/10</td></tr>" for k in lab)
crit = [
  ("Required run rate and runs needed", "Used", "Inputs to the win-probability model in chases; drives leverage."),
  ("Wickets in hand", "Used", "Model input; also sets the expected cost of a wicket."),
  ("Balls remaining and stage of innings", "Used", "Model input; phase (powerplay, middle, death) sets the outcome mix and the expected runs."),
  ("How close the contest is", "Used", "Leverage is largest when win probability is near 50% late in the innings; decided states (below 3% or above 97%) are excluded."),
  ("Chase versus first innings", "Used", "First-innings pressure comes from a weaker model (AUC 0.68), so those balls count at half weight."),
  ("Which side is behind", "Used", "Directional weights: batters feel more pressure when behind, bowlers when the batting side is ahead."),
  ("IPL versus T20I", "Used", "Competition type is a model input; ratings are fitted on both together and linked through shared opponents."),
  ("Opposition quality", "Used", "Crossed batter and bowler effects: each is credited relative to who they faced."),
  ("Role (batting slot, pace versus spin)", "Used", "Role effects in the rating model, baseline and under pressure."),
  ("Lower-level events", "Used", "Emerging Teams Asia Cup and Rising Stars count at half weight."),
  ("Small samples", "Used", "Ridge shrinkage, minimum 40 (36 for bowlers) effective pressure balls and 8 matches, bootstrap bands."),
  ("Rain-affected matches, ties, super overs", "Excluded", "34 revised-target matches, tied and no-result games, and super overs removed."),
  ("Run-outs and retirements", "Excluded", "Not credited to the bowler; batters not credited for byes, leg-byes or wides."),
  ("Knockout and final stakes", "Not used", "Only 31 matches, too few to estimate; a possible extension."),
  ("Venue, pitch, dew, partnerships", "Not used", "Not available in the data or not modelled; listed as a limitation."),
]
crit_rows = "".join(f"<tr><td class=w>{a}</td><td><span class='pill {b.replace(' ','')}'>{b}</span></td><td class=w>{c}</td></tr>" for a, b, c in crit)
css = """
@page { size: A4; margin: 15mm 15mm 17mm; }
:root { --ink:#0D1230; --ink2:#4A5072; --ink3:#7A80A0; --line:#D9DCEB; --acc:#D81B73; --soft:#F4F5FA; --good:#0E8F5A; --warn:#B86E00; --bad:#C4302B; }
* { box-sizing: border-box; }
body { font-family: "Inter", "DejaVu Sans", sans-serif; font-size: 9.6pt; line-height: 1.5; color: var(--ink); margin: 0; }
h1, h2, h3 { font-family: "Inter Display", "Inter", sans-serif; letter-spacing: -0.015em; margin: 0; line-height: 1.1; }
h1 { font-size: 28pt; font-weight: 800; }
h2 { font-size: 17pt; font-weight: 800; margin-bottom: 8px; }
h3 { font-size: 11pt; font-weight: 700; margin: 14px 0 5px; }
p { margin: 0 0 7px; } .muted { color: var(--ink2); }
.mono, td.r, td.n, .num { font-family: "DejaVu Sans Mono", monospace; font-size: 8.4pt; }
.eyebrow { font-family: "DejaVu Sans Mono", monospace; font-size: 7.4pt; letter-spacing: .14em; text-transform: uppercase; color: var(--ink3); }
.cover { background: var(--ink); color: #fff; margin: -15mm -15mm 0; padding: 16mm 15mm 9mm; }
.cover .eyebrow { color: #A8AED0; } .cover h1 em { font-style: normal; color: #FF6FB0; }
.cover p { color: #C9CDE6; max-width: 150mm; font-size: 11pt; margin-top: 10px; }
.strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; margin: 16px 0 0; border-top: 1px solid #3A4070; }
.strip div { padding: 10px 10px 0 0; } .strip b { font-family: "Inter Display"; font-size: 19pt; display: block; line-height: 1; } .strip span { font-size: 8pt; color: #A8AED0; }
.box { background: var(--soft); border-left: 3px solid var(--acc); padding: 9px 12px; margin: 10px 0; border-radius: 0 4px 4px 0; }
.box b { color: var(--ink); }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.6pt; }
th { font-family: "DejaVu Sans Mono", monospace; font-size: 6.9pt; letter-spacing: .06em; text-transform: uppercase; color: var(--ink3); text-align: left; padding: 5px 6px; border-bottom: 1.5px solid var(--ink); font-weight: 500; }
td { padding: 4px 6px; white-space: nowrap; border-bottom: 1px solid var(--line); vertical-align: top; }
td.r, th.r { text-align: right; } td.w { white-space: normal; } td.n { color: var(--ink3); width: 22px; } th.n { width: 22px; }
tr.hl td { background: #FCE8F1; font-weight: 600; }
.sm { color: var(--ink3); font-size: 7.4pt; font-weight: 400; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.cols3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.card { border: 1px solid var(--line); border-radius: 6px; padding: 9px 11px; break-inside: avoid; }
.card .st { font-family: "DejaVu Sans Mono", monospace; font-size: 6.8pt; letter-spacing: .1em; text-transform: uppercase; }
.keep { color: var(--good); } .drop { color: var(--bad); } .note { color: var(--warn); }
.card h3 { margin: 3px 0 4px; font-size: 10pt; } .card p { font-size: 8.6pt; color: var(--ink2); margin: 0; }
.formula { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt; background: var(--soft); padding: 8px 10px; border-radius: 4px; margin: 6px 0 8px; }
img.fig { width: 100%; display: block; margin: 6px 0 2px; } .cap { font-size: 7.8pt; color: var(--ink3); margin-bottom: 8px; }
.pill { font-family: "DejaVu Sans Mono", monospace; font-size: 6.8pt; padding: 1px 6px; border-radius: 9px; border: 1px solid var(--line); white-space: nowrap; }
.pill.Used { color: var(--good); border-color: #9AD3B8; } .pill.Excluded { color: var(--warn); border-color: #E7C48E; } .pill.Notused { color: var(--ink3); }
.pb { break-before: page; } .nb { break-inside: avoid; }
ul { margin: 4px 0 8px; padding-left: 16px; } li { margin-bottom: 3px; }
ol { margin: 4px 0 8px; padding-left: 18px; } ol li { margin-bottom: 4px; }
"""
html = f"""<!doctype html><html><head><meta charset=utf-8><style>{css}</style></head><body>
<section class=cover>
  <div class=eyebrow>SupeRR Selector Season 2 &middot; Round 1 &middot; Methodology report</div>
  <h1 style="margin-top:12px">Who delivers when<br>the match is <em>on the line?</em></h1>
  <p>A data-driven system that rates IPL and T20I batters and bowlers on what they do in the highest-pressure situations, compared with what an average player does in the same situation.</p>
  <div class=strip><div><b>{D['meta']['matches']}</b><span>clean matches ({D['meta']['ipl']} IPL, {D['meta']['t20i']} T20I)</span></div><div><b>{round(D['meta']['balls']/1000)}k</b><span>deliveries scored</span></div><div><b>{len(D['bat'])+len(D['bowl'])}</b><span>players eligible to rank</span></div><div><b>0.89</b><span>win-probability AUC in chases</span></div></div>
</section>

<h3 style="margin-top:16px">Summary</h3>
<p>We define pressure as how much the next ball can swing the match, estimated with a win-probability model. Each delivery is then scored in runs above what an average player produces in the same match state, with wickets costed. A ridge model with crossed batter and bowler effects turns those scores into a pressure rating per player, controlling for opponent, role and small samples. We tested every candidate score on matches it had not seen and kept only what predicted held-out pressure results. A 100-run match bootstrap gives each player a probability of staying in the top 10.</p>
<div class=box><b>Main finding.</b> Baseline skill measured on pressure-weighted balls, adjusted for opponent and role, was the only signal that predicted future pressure results. Player-specific "clutch" measures (change from own baseline, hit-rate consistency) did not, so they do not drive the ranking.</div>
<div class=cols>
<div><h3>Top 10 batters</h3><table><thead><tr><th class=n>#</th><th>Player</th><th class=r>Rating</th><th class=r>SR press/ord</th><th class=r>Top-10</th></tr></thead><tbody>{bat_rows}</tbody></table></div>
<div><h3>Top 10 bowlers</h3><table><thead><tr><th class=n>#</th><th>Player</th><th class=r>Rating</th><th class=r>Econ press/ord</th><th class=r>Wkt%</th><th class=r>Top-10</th></tr></thead><tbody>{bowl_rows}</tbody></table></div>
</div>
<p class="cap">Rating = estimated runs per over better than an average player in the same match state, after adjusting for opponent, role and sample size. Top-10 = share of 100 bootstrap runs in which the player stays in the top 10. Strike rate and economy are raw, for context only.</p>

<section class=pb>
<h2>1. How we defined pressure</h2>
<p>Pressure is how much the next ball can swing the match. We estimate it with a win-probability model instead of hand-picking weights, because required rate, wickets in hand and balls left interact: 12 an over is routine with eight wickets in hand and desperate with three.</p>
<ol>
<li><b>Win-probability model.</b> A gradient-boosted classifier predicts whether the batting side wins, from innings, balls left, wickets lost, runs scored, runs required and required rate (chases), current run rate and competition type. It is trained in 5 folds grouped by match, so no delivery is scored by a model that saw its match. Out-of-fold AUC is 0.89 in chases and 0.68 in first innings (log-loss 0.538 against 0.693 for chance).</li>
<li><b>Leverage.</b> For every ball we apply each possible outcome (dot, 1 to 6, wicket), weighted by how often it happens in that phase, and average the absolute change in win probability. That expected swing is the ball's leverage.</li>
<li><b>Live states and tiers.</b> Balls with win probability outside 3% to 97% are labelled Decided and excluded. Among live balls, leverage is ranked within each innings and cut into four tiers: <b>Ordinary</b> (below the 60th percentile), <b>Elevated</b> (60th to 90th), <b>High</b> (90th to 97th) and <b>Extreme</b> (97th and above).</li>
<li><b>First innings.</b> Pressure there comes from a weaker model, so first-innings balls count at half weight.</li>
<li><b>Direction.</b> Batter pressure is scaled by 0.6 + 0.8 &times; (1 &minus; win probability) and bowler pressure by 0.6 + 0.8 &times; win probability, so a 50-50 game weighs both equally and a side that is behind feels more of it.</li>
</ol>
<h3>Does the definition separate pressure from ordinary play?</h3>
<p>Yes. As pressure rises, batters play more dot balls (34.2% in Ordinary, 38.1% in High) and lose wickets more often (5.4% in Ordinary, 7.3% in Extreme). Scoring speed barely changes, so a raw strike rate cannot show who copes with pressure. The cost of pressure appears as wickets and dots.</p>
<table><thead><tr><th>Tier</th><th class=r>Balls</th><th class=r>Runs / ball</th><th class=r>Dot balls</th><th class=r>Boundaries</th><th class=r>Wickets</th></tr></thead><tbody>{tier_rows}</tbody></table>
<div class=cols><div><img class=fig src="{img('fig1_pressure_tiers.png')}"><div class=cap>Outcome rates by pressure tier, league-wide.</div></div>
<div><img class=fig src="{img('fig4_pressure_map.png')}"><div class=cap>Pressure peaks late in close chases.</div></div></div>
</section>

<section class=pb>
<h2>2. What we took into account</h2>
<p>The problem statement lists required rate, wickets in hand, balls remaining, stage of innings and closeness, and invites other signals. This table shows every factor we considered and what we did with it.</p>
<table><thead><tr><th style="width:34%">Factor</th><th style="width:12%">Status</th><th>How it is handled</th></tr></thead><tbody>{crit_rows}</tbody></table>
</section>

<section class=pb>
<h2>3. How we measured performance</h2>
<h3 style="margin-top:0">Run value above what the match state predicts</h3>
<p>Every ball gets a value V in runs, from the batting side's view:</p>
<div class=formula>V = (runs &minus; expected runs) &minus; C &times; (wicket &minus; expected wicket chance)</div>
<p>Expected runs and wicket chance come from gradient-boosted models of the match state only (no player identity), fitted out-of-fold. C is the expected runs lost when a wicket falls in that state (4.6 runs on average), from a rest-of-innings run model. A four off a ball where the average is 1.4 runs is worth about +2.6. A wicket in the powerplay costs about 6.6 runs on average, in the death overs about 1.9.</p>
<div class=box><b>Why not win-probability credit?</b> Win probability only defines pressure. We tested it as a score and dropped it: batter credit drifted by &minus;1.0 points per 100 balls in the Extreme tier, because the same noisy model both selects and scores those balls, and it barely persisted across seasons (Spearman &rho; about 0.1 for batters, against 0.19 for run value and 0.30 for bowlers).</div>
<h3>Pressure-adjusted rating model</h3>
<p>A ridge regression with crossed batter and bowler effects (the idea behind adjusted plus-minus) explains V on every live ball:</p>
<ul>
<li><b>Each player has a baseline</b> (usual level) <b>and a pressure slope</b> (change as pressure intensity rises). The bowler side is sign-flipped, so positive always means good for that player.</li>
<li><b>Opponent controlled:</b> a batter is credited relative to the bowlers he faced and a bowler relative to the batters he bowled to.</li>
<li><b>Role controlled:</b> batting slot (1-2, 3-4, 5-6, 7+) and pace versus spin, as a baseline and a pressure shift.</li>
<li><b>Weights:</b> lower-level events count 0.5; balls are up-weighted by 1 + 3 &times; pressure intensity so pressure balls drive the estimate.</li>
<li><b>Shrinkage:</b> ridge strength 1,000, with the pressure slope penalised four times as hard as the baseline. The typical player has only 15 to 25 balls in the top tiers, so this is the main protection against small samples.</li>
</ul>
<div class=formula>rating = baseline + shrunk pressure slope  (reported in runs per over)</div>
<p>Delta (the slope alone) and a consistency score (shrunk share of pressure matches with positive net run value) are calculated for every player as diagnostics. The final ranking uses the rating alone, for the reason shown in section 5.</p>
<h3>Fair comparisons</h3>
<ul>
<li><b>Data:</b> 867 matches have ball-by-ball data. We removed 34 rain-affected matches, tied and no-result games and super overs, leaving {D['meta']['matches']} matches and about {round(D['meta']['balls']/1000)},000 deliveries.</li>
<li><b>Eligibility:</b> at least 40 effective pressure balls (36 for bowlers; first-innings balls count half) across at least 8 matches, giving {len(D['bat'])} batters and {len(D['bowl'])} bowlers from 888 and 647 players.</li>
<li><b>IPL and T20I:</b> competition type is a model input and the rating model is fitted on both together. Players who appear in both link the two pools.</li>
<li><b>Recency:</b> every listed player has played since mid-2025, which matters for an IPL 2027 auction.</li>
</ul>
</section>

<section class=pb>
<h2>4. Final rankings and evidence</h2>
<h3 style="margin-top:0">Batters</h3>
<table><thead><tr><th class=n>#</th><th>Player</th><th class=r>Rating</th><th class=r>SR press / ord</th><th class=r>Boundary %</th><th class=r>Dismissal %</th><th class=r>Naive rank</th><th class=r>Balls (m)</th><th class=r>Top-10</th></tr></thead><tbody>
{''.join(f"<tr><td class=n>{p['rank']}</td><td><b>{p['name']}</b></td><td class=r>{sgn(p['rating'])}</td><td class=r>{p['p']:.0f} / {p['o']:.0f}</td><td class=r>{p['bdy']:.1f}</td><td class=r>{p['out']:.1f}</td><td class=r>#{p['naive']}</td><td class=r>{p['pb']} ({p['pm']})</td><td class=r>{round(p['pt10']*100)}%</td></tr>" for p in D['bat'][:10])}</tbody></table>
<h3>Bowlers</h3>
<table><thead><tr><th class=n>#</th><th>Player</th><th class=r>Rating</th><th class=r>Econ press / ord</th><th class=r>Dot %</th><th class=r>Wicket %</th><th class=r>Naive rank</th><th class=r>Balls (m)</th><th class=r>Top-10</th></tr></thead><tbody>
{''.join(f"<tr><td class=n>{p['rank']}</td><td><b>{p['name']}</b></td><td class=r>{sgn(p['rating'])}</td><td class=r>{p['p']:.2f} / {p['o']:.2f}</td><td class=r>{p['dot']:.1f}</td><td class=r>{p['out']:.1f}</td><td class=r>#{p['naive']}</td><td class=r>{p['pb']} ({p['pm']})</td><td class=r>{round(p['pt10']*100)}%</td></tr>" for p in D['bowl'][:10])}</tbody></table>
<p class=cap>Naive rank = position among the same eligible players by raw pressure strike rate (batters) or raw pressure economy (bowlers). Only 5 of the 10 batters and 4 of the 10 bowlers appear in a naive top 10, so raw pressure stats would have given a different list. Bumrah's raw pressure economy (7.64) is no better than his ordinary one (7.63): the rating credits his run suppression against what an average bowler concedes in the same states.</p>
<img class=fig src="{img('fig2_top10_uncertainty.png')}" style="width:84%"><div class=cap>Rating (z-score) with the 10th to 90th percentile range over 100 bootstrap resamples of matches. Places 8 to 10 in both lists are close calls.</div>
</section>

<section class=pb>
<h2>5. How we reached the conclusion</h2>
<p>We fitted every candidate on half the matches and checked how well it ranks players on pressure balls in the other half. We repeated this on 10 random 50/50 splits by match and once chronologically (fit on 2022 to 2024, test on 2025 to 2026). Values are Spearman rank correlations: random-split average, with the chronological test in brackets. Tier "Elev+" combines Elevated, High and Extreme balls; "High+" is High and Extreme only. About 68 batters and 90 bowlers per test in Elev+, fewer in High+.</p>
<table><thead><tr><th>Score</th><th class=r>Batters Elev+</th><th class=r>Batters High+</th><th class=r>Bowlers Elev+</th><th class=r>Bowlers High+</th></tr></thead><tbody>{val_rows}</tbody></table>
<img class=fig src="{img('fig3_predictive_validity.png')}" style="width:70%"><div class=cap>Held-out rank correlation by score, Elevated + High + Extreme tiers (random-split average).</div>
<div class=cols3>
<div class=card><div class="st keep">Kept</div><h3>Baseline skill on pressure-weighted balls</h3><p>The only signal that predicted held-out pressure results. Adjusted for opponent and role and shrunk for small samples.</p></div>
<div class=card><div class="st drop">Dropped</div><h3>Pressure slope and hit-rate</h3><p>Neither predicted held-out results. A composite with 40% weight on them lowered accuracy in every variant, so the final score is the rating alone.</p></div>
<div class=card><div class="st drop">Dropped</div><h3>Win-probability credit as the score</h3><p>It drifted in the Extreme tier and barely persisted across seasons, so run value scores players and win probability only defines pressure.</p></div>
</div>
<h3>What this means</h3>
<ul>
<li>Raw pressure-only averages are the worst guide for batters (rho {sgn(val('random-split mean','bat','Elev+','naive_pressure_mean'))}). With 15 to 25 pressure balls per player they are mostly noise.</li>
<li>Our reading, consistent with the wider sports-analytics literature on clutch performance, is that the best predictor of who delivers under pressure is underlying skill, measured on the balls that matter and adjusted for opponent and role.</li>
<li>The rating is no more accurate than a raw all-ball average on this test ({sgn(val('random-split mean','bat','Elev+','level'))} against {sgn(val('random-split mean','bat','Elev+','naive_overall_mean'))} for batters, within noise). Its advantages are opponent and role adjustment, shrinkage, and weight on pressure balls.</li>
</ul>
<h3>Sensitivity of the top 10</h3>
<p>Names unchanged out of 10 when a single setting changes:</p>
<table style="width:64%"><thead><tr><th>Change</th><th class=r>Batters</th><th class=r>Bowlers</th></tr></thead><tbody>{sens_rows}</tbody></table>
</section>

<section class=pb>
<h2>6. Reliability and limitations</h2>
<ul>
<li><b>Pressure-specific skill is hard to detect.</b> Only baseline skill predicted future pressure results. We rank on that and make no claim of a measurable clutch gene.</li>
<li><b>Batters in the top tiers are the least certain.</b> Held-out correlation for batters in the High and Extreme tiers is close to zero (+0.02), against +0.12 for bowlers. Read batter order within the top 10 as a tier, not a precise ranking. In the chronological test the rating's correlation is {sgn(val('temporal','bowl','Elev+','level'))} for bowlers and {sgn(val('temporal','bat','Elev+','level'))} for batters across all pressure tiers.</li>
<li><b>Thin samples at the edge.</b> Several players have fewer than 80 pressure balls. Shrinkage and the bootstrap account for this, but a few more matches could move them. Several players outside the lists, such as Samson, Parag and Chakravarthy, are statistically close to the edge names.</li>
<li><b>First-innings pressure is approximate.</b> The first-innings win-probability model is weaker (AUC 0.68), so those balls count at half weight and the tiers are mostly a chase story.</li>
<li><b>Missing context.</b> Pitch, dew, partnerships and batting-order decisions are not modelled. Players who appear in only one competition are linked to others through shared opponents, which is weaker for T20I-only players.</li>
<li><b>Settings were tuned on this data.</b> Ridge strength and pressure weights were tuned on held-out splits of this dataset. The sensitivity analysis shows the effect is modest (8 or 9 of 10 names stable), but the final lists are not independently validated.</li>
</ul>
<h2 style="margin-top:18px">7. Reproducing the results</h2>
<p>The pipeline is a set of Python scripts run in order on the six CSVs in the data pack (pandas, scikit-learn, SciPy): <span class=mono>01_pressure_index.py</span> (states, win-probability model, leverage, tiers, run values), <span class=mono>02_ratings.py</span> (ridge ratings and bootstrap), <span class=mono>03_validate.py</span> (held-out validation and sensitivity), <span class=mono>04_report_assets.py</span> (tables and figures). Random seeds are fixed. An interactive version of these results is available as a companion website.</p>
<p class=cap>Data window: {D['meta']['first']} to {D['meta']['last']}. IPL 2022 to 2026 and T20Is from 2022.</p>
</section>
</body></html>"""
(OUT / "report.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); pg.goto("file://" + str(OUT / "report.html")); pg.wait_for_timeout(500)
    pg.pdf(path=str(OUT / "SupeRR_R1_Methodology_Report.pdf"), format="A4", print_background=True, display_header_footer=True,
           header_template="<span></span>", margin=dict(top="15mm", bottom="17mm", left="15mm", right="15mm"),
           footer_template='<div style="width:100%;font-family:Inter,sans-serif;font-size:7pt;color:#7A80A0;padding:0 15mm;display:flex;justify-content:space-between"><span>SupeRR Selector S2 &middot; Round 1 &middot; Methodology report</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
    b.close()
print("pdf built")
