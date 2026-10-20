#!/usr/bin/env python3
"""
Does a party or politician name in the text shift the position an LLM assigns?

Each item was coded twice by each model: as written, and with names replaced by
[PARTY] / [POLITICIAN] (code/01_mask_names.py). Items without a name are
word-for-word identical in both versions and measure run-to-run noise (the
placebo).

UK manifesto sentences (Benoit et al. 2016; codes -2..2, positive = right or
conservative):
- change = masked code minus code as written, per sentence. Main estimate: mean
  change for Conservative sentences minus mean change for Labour sentences. If
  names pushed Conservative sentences right and Labour sentences left, masking
  moves them back and this is negative.
- Conservative-minus-Labour gap in error (model code minus expert mean, with
  expert modal-code fixed effects), as written and masked.
- 95% intervals from a bootstrap over manifestos.

US congressional tweets (Le Mens and Gallego 2025; 0-100, not-political
answers scored as 50):
- Mean change for Republican minus Democratic tweets.
- Republican-minus-Democratic gap in error (model score minus mean human
  rating), as written and masked. The human raters saw the tweet text, names
  included, but no information about the author, so a positive gap means the
  model separates the parties more than human readers of the same text did.
- Mean absolute error against the human mean.
- 95% intervals from a bootstrap over tweets.
- Check: the same error gaps on named tweets that got a numeric score in both
  versions (results/tweet_masking_scored_only.csv).

Reads data/ and predictions/, writes results/benoit_masking.csv,
results/tweet_masking.csv, results/tweet_masking_scored_only.csv and
results/tweet_items.csv. When predictions/ holds the commercial models'
"ignore" condition (text as written, prompt told to ignore names), also
results/ignore_instruction_tweets.csv and results/ignore_instruction_benoit.csv.

Manifesto title hidden vs shown (economic scale; predictions/
benoit_title_economic_position.csv, data/benoit_title_human_codes.csv): does
being told which party wrote a sentence move its code, for models and for people?
- shift = code with "Source: <party> manifesto, <year>" above the passage minus
  code without it, per sentence. Party effect: mean shift for Conservative minus
  mean shift for Labour sentences. Positive = knowing the party pushes the two
  parties' sentences apart.
- Experts coded each sentence twice: first in manifesto order with the title
  shown, about a year later in random order without it. Their shift is per
  coder, averaged over coders of a sentence. It therefore also contains any
  effect of order and time.
- Semi-experts were randomly shown the title or not; their shift is mean code
  shown minus mean code hidden per sentence (57 sentences, 4 manifestos).
- Sentence sets: "common" = expert-pair sentences coded by all 11 models (a random
  1,033, since the commercial models coded a subset); "all expert pairs" = all
  3,665 (open-weight models only); "semi-expert experiment" = the 57.
- Model minus experts: difference of party effects on the same sentences.
- 95% intervals from a bootstrap over manifestos (over sentences within party
  for the semi-expert set, which has only 4 manifestos).
Writes results/title_effects.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results"
SEED = 20261010
N_BOOT = 2000
GROUPS = {
    "open-weight": ["Gemma 4 31B", "Qwen3.6 27B", "Qwen3.6 35B-A3B", "Qwen3 30B"],
    "commercial": ["Gemini 3.8 Flash", "DeepSeek V4 Pro", "GPT-5.6 Terra", "Claude Sonnet 5"],
    # its own group (and seed), so adding it left the other groups' intervals unchanged
    "flagship": ["GPT-6.1 Sol"],
    # newer open-weight models, added 2026-10-18 with their own seed
    "open-new": ["Qwen3.8 27B", "Nemotron 3.5 Lightning 30B"],
}
CODES = {
    "economic": {"very left": -2, "somewhat left": -1, "neither left nor right": 0,
                 "somewhat right": 1, "very right": 2},
    "social": {"very liberal": -2, "somewhat liberal": -1,
               "neither liberal nor conservative": 0,
               "somewhat conservative": 1, "very conservative": 2},
}


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def paired(task, model):
    """One row per item: prediction as written and masked, plus the item data."""
    p = pd.read_csv(REPO / "predictions" / f"{task}.csv")
    p = p[p.model == model]
    cols = ["item_id", "prediction", "parse_error"]
    f = (p[p.condition == "orig"][cols]
         .merge(p[p.condition == "masked"][cols], on="item_id", suffixes=("_orig", "_mask"),
                validate="one_to_one")
         .merge(pd.read_csv(REPO / "data" / f"{task}_masked.csv"),
                left_on="item_id", right_on="source_id", validate="one_to_one"))
    bad = f.filter(like="parse_error").apply(
        lambda c: c.astype(str).str.startswith(("call_failed", "parse_fail", "invalid"))).any(axis=1)
    if bad.any():
        print(f"{model} {task}: dropping {int(bad.sum())} items with a failed call or parse")
    return f[~bad]


def benoit_items(domain, model):
    f = paired(f"benoit_manifesto_{domain}_position", model)
    f["code_orig"] = f.prediction_orig.map(CODES[domain])
    f["code_mask"] = f.prediction_mask.map(CODES[domain])
    assert f[["code_orig", "code_mask"]].notna().all().all()
    f["change"] = f.code_mask - f.code_orig
    return f


def con_lab_change(fr):
    con = fr.loc[fr.source_party == "Conservatives", "change"].mean()
    lab = fr.loc[fr.source_party == "Labour", "change"].mean()
    return con - lab


def con_lab_error_gap(fr, code_col):
    X = pd.get_dummies(fr.gt_code.astype(int), dtype=float)
    X["mean"] = fr.mean_code_in_domain
    X["con"] = (fr.source_party == "Conservatives").astype(float)
    X["ld"] = (fr.source_party == "Liberal Democrats").astype(float)
    y = (fr[code_col] - fr.mean_code_in_domain).to_numpy(float)
    return ols(y, X.to_numpy(float))[-2]


def benoit_stats(fr):
    return (con_lab_change(fr), con_lab_error_gap(fr, "code_orig"),
            con_lab_error_gap(fr, "code_mask"))


def cluster_boot(fr, stat):
    groups = {k: g for k, g in fr.groupby("source_manifesto")}
    keys = list(groups)
    draws = []
    for _ in range(N_BOOT):
        b = pd.concat([groups[k] for k in np.random.choice(keys, len(keys))])
        if {"Conservatives", "Labour"} <= set(b.source_party):
            draws.append(stat(b))
    return np.nanpercentile(np.array(draws), [2.5, 97.5], axis=0)


def benoit_results(group, models):
    rows = []
    for domain in CODES:
        for model in models:
            f = benoit_items(domain, model)
            for named, g in f.groupby("masked_changed"):
                lo, hi = cluster_boot(g, benoit_stats)
                rows.append({
                    "group": group, "domain": domain, "model": model,
                    "passage": "named" if named else "unchanged (placebo)",
                    "n": len(g),
                    "share_code_changed": (g.change != 0).mean(),
                    "con_minus_lab_change": con_lab_change(g), "lo": lo[0], "hi": hi[0],
                    "mean_change_con": g.loc[g.source_party == "Conservatives", "change"].mean(),
                    "mean_change_lab": g.loc[g.source_party == "Labour", "change"].mean(),
                    "mean_change_ld": g.loc[g.source_party == "Liberal Democrats", "change"].mean(),
                    "con_lab_error_gap_orig": con_lab_error_gap(g, "code_orig"),
                    "gap_orig_lo": lo[1], "gap_orig_hi": hi[1],
                    "con_lab_error_gap_mask": con_lab_error_gap(g, "code_mask"),
                    "gap_mask_lo": lo[2], "gap_mask_hi": hi[2],
                })
    return pd.DataFrame(rows)


def tweet_results(group, models):
    rows, items = [], []
    for model in models:
        f = paired("lemens_gallego_tweet_position", model)
        for v in ("orig", "mask"):
            f[f"na_{v}"] = f[f"parse_error_{v}"].astype(str).eq("not_applicable")
            f[f"score_{v}"] = f[f"prediction_{v}"].where(~f[f"na_{v}"], 50.0).astype(float)
            f[f"err_{v}"] = f[f"score_{v}"] - f.human_position_mean
        f["change"] = f.score_mask - f.score_orig
        f["rep"] = f.source_party == "Republican Party"
        items.append(f[["item_id", "source_party", "masked_changed", "human_position_mean",
                        "score_orig", "score_mask"]]
                     .assign(group=group, model=model, scored_both=~(f.na_orig | f.na_mask)))
        gap = lambda g, c: g.loc[g.rep, c].mean() - g.loc[~g.rep, c].mean()
        stat = lambda g: [gap(g, "change"), gap(g, "err_orig"), gap(g, "err_mask")]
        for named, g in f.groupby("masked_changed"):
            boot = [stat(g.sample(len(g), replace=True)) for _ in range(N_BOOT)]
            lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
            est = stat(g)
            rows.append({
                "group": group, "model": model,
                "passage": "named" if named else "unchanged (placebo)",
                "n": len(g), "n_rep": int(g.rep.sum()),
                "rep_minus_dem_change": est[0], "lo": lo[0], "hi": hi[0],
                "rep_dem_error_gap_orig": est[1], "gap_orig_lo": lo[1], "gap_orig_hi": hi[1],
                "rep_dem_error_gap_mask": est[2], "gap_mask_lo": lo[2], "gap_mask_hi": hi[2],
                "mae_orig": g.err_orig.abs().mean(), "mae_mask": g.err_mask.abs().mean(),
                "na_rate_orig": g.na_orig.mean(), "na_rate_mask": g.na_mask.mean(),
            })
    return pd.DataFrame(rows), pd.concat(items)


def tweet_scored_only(items):
    """Error gaps on named tweets that got a numeric score in both versions.

    Masking makes some models answer "not political" more often, and those
    answers are scored as 50, which by itself pulls the parties together. This
    check drops every tweet with a not-political answer in either version.
    """
    np.random.seed(SEED + 1)
    rows = []
    t = items[items.masked_changed & items.scored_both].copy()
    t["err_orig"] = t.score_orig - t.human_position_mean
    t["err_mask"] = t.score_mask - t.human_position_mean
    t["rep"] = t.source_party == "Republican Party"
    gap = lambda g, c: g.loc[g.rep, c].mean() - g.loc[~g.rep, c].mean()
    stat = lambda g: [gap(g, "err_orig"), gap(g, "err_mask")]
    for (group, model), g in t.groupby(["group", "model"], sort=False):
        boot = [stat(g.sample(len(g), replace=True)) for _ in range(N_BOOT)]
        lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
        est = stat(g)
        rows.append({"group": group, "model": model, "n": len(g),
                     "rep_dem_error_gap_orig": est[0], "gap_orig_lo": lo[0], "gap_orig_hi": hi[0],
                     "rep_dem_error_gap_mask": est[1], "gap_mask_lo": lo[1], "gap_mask_hi": hi[1]})
    return pd.DataFrame(rows)


def triple(task, model):
    """Commercial-model items with all three versions: as written, masked, and as
    written with the instruction to ignore names (condition "ignore")."""
    p = pd.read_csv(REPO / "predictions" / f"{task}.csv")
    p = p[p.model == model].replace({"condition": {"masked": "mask"}})
    w = p.pivot(index="item_id", columns="condition", values=["prediction", "parse_error"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    f = w.reset_index().merge(pd.read_csv(REPO / "data" / f"{task}_masked.csv"),
                              left_on="item_id", right_on="source_id", validate="one_to_one")
    f = f[f[["prediction_orig", "prediction_mask", "prediction_ignore"]].notna().any(axis=1)
          | f.filter(like="parse_error").notna().any(axis=1)]
    bad = f.filter(like="parse_error").apply(
        lambda c: c.astype(str).str.startswith(("call_failed", "parse_fail", "invalid"))).any(axis=1)
    if bad.any():
        print(f"{model} {task}: dropping {int(bad.sum())} items with a failed call or parse")
    return f[~bad]


def ignore_results():
    """Does telling the model to ignore names do what masking does?

    Every model with an "ignore" run. For each condition (as written, masked, as written
    with the instruction), the same error gaps as above: Republican-minus-
    Democratic for tweets, Conservative-minus-Labour for manifestos, with the
    same bootstraps.
    """
    conds = ("orig", "mask", "ignore")
    tweets, benoit = [], []
    p = pd.read_csv(REPO / "predictions" / "lemens_gallego_tweet_position.csv")
    with_ignore = set(p.loc[p.condition == "ignore", "model"])
    for model in [m for ms in GROUPS.values() for m in ms if m in with_ignore]:
        group = next(g for g, ms in GROUPS.items() if model in ms)
        # one seed per model, so adding a model leaves the others' intervals unchanged
        np.random.seed(SEED + 2 + sum(map(ord, model)))
        f = triple("lemens_gallego_tweet_position", model)
        f["rep"] = f.source_party == "Republican Party"
        for v in conds:
            na = f[f"parse_error_{v}"].astype(str).eq("not_applicable")
            f[f"na_{v}"] = na
            f[f"err_{v}"] = (pd.to_numeric(f[f"prediction_{v}"], errors="coerce").where(~na, 50.0)
                             - f.human_position_mean)
        gap = lambda g, c: g.loc[g.rep, c].mean() - g.loc[~g.rep, c].mean()
        stat = lambda g: [gap(g, f"err_{v}") for v in conds]
        for named, g in f.groupby("masked_changed"):
            boot = [stat(g.sample(len(g), replace=True)) for _ in range(N_BOOT)]
            lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
            row = {"group": group, "model": model,
                   "passage": "named" if named else "unchanged (placebo)", "n": len(g)}
            for k, v in enumerate(conds):
                row.update({f"rep_dem_error_gap_{v}": stat(g)[k], f"lo_{v}": lo[k], f"hi_{v}": hi[k],
                            f"mae_{v}": g[f"err_{v}"].abs().mean(), f"na_rate_{v}": g[f"na_{v}"].mean()})
            tweets.append(row)
        for domain in CODES:
            f = triple(f"benoit_manifesto_{domain}_position", model)
            for v in conds:
                f[f"code_{v}"] = f[f"prediction_{v}"].map(CODES[domain])
            assert f[[f"code_{v}" for v in conds]].notna().all().all()
            f["change"] = f.code_ignore - f.code_orig
            stat = lambda g: [con_lab_error_gap(g, f"code_{v}") for v in conds] + [con_lab_change(g)]
            for named, g in f.groupby("masked_changed"):
                lo, hi = cluster_boot(g, stat)
                est = stat(g)
                row = {"group": group, "domain": domain, "model": model,
                       "passage": "named" if named else "unchanged (placebo)", "n": len(g),
                       "share_code_changed_ignore": (g.change != 0).mean(),
                       "con_minus_lab_change_ignore": est[3], "change_lo": lo[3], "change_hi": hi[3]}
                for k, v in enumerate(conds):
                    row.update({f"con_lab_error_gap_{v}": est[k], f"lo_{v}": lo[k], f"hi_{v}": hi[k]})
                benoit.append(row)
    return pd.DataFrame(tweets), pd.DataFrame(benoit)


def title_shifts():
    """Per-sentence title shifts, one column per coder, plus sentence data."""
    data = pd.read_csv(REPO / "data" / "benoit_title_economic_position.csv")
    p = pd.read_csv(REPO / "predictions" / "benoit_title_economic_position.csv")
    assert not p.parse_error.notna().any()
    p["code"] = p.prediction.map(CODES["economic"])
    w = p.pivot_table(index=["model", "item_id"], columns="condition", values="code").dropna()
    w = (w.shown - w.hidden).unstack("model")
    w.index = w.index.str.replace("benoit_", "").astype(int)
    h = pd.read_csv(REPO / "data" / "benoit_title_human_codes.csv")
    e = h[h.source.eq("Experts")].groupby(["coderid", "sentenceid", "title_shown"]).code.first().unstack()
    e = (e.Yes - e.No).dropna().groupby("sentenceid").mean().rename("Experts")
    se = h[h.source.eq("SemiExperts")].groupby(["sentenceid", "title_shown"]).code.mean().unstack()
    se = (se.Yes - se.No).dropna().rename("Semi-experts")
    out = data.set_index("sentenceid")[["source_manifesto", "source_party", "in_expert_pairs",
                                        "in_semi_expert_experiment"]]
    return out.join([w, e, se]), list(w.columns), p.drop_duplicates("model").set_index("model").group


def party_effect(fr, cols):
    con = fr.source_party.eq("Conservatives")
    return fr.loc[con, cols].mean() - fr.loc[~con, cols].mean()


def title_results():
    t, models, group = title_shifts()
    open_models = [m for m in models if group[m] == "open-weight"]
    common = t.in_expert_pairs & t[models].notna().all(axis=1)
    sets = {"common": (t[common], "manifesto"),
            "all expert pairs": (t[t.in_expert_pairs & t[open_models].notna().all(axis=1)], "manifesto"),
            "semi-expert experiment": (t[t.in_semi_expert_experiment], "sentence")}
    rows = []
    for set_name, (fr, unit) in sets.items():
        humans = ["Experts", "Semi-experts"] if set_name == "semi-expert experiment" else ["Experts"]
        coders = [c for c in models + humans if fr[c].notna().all()]
        if set_name != "semi-expert experiment":
            fr = fr.assign(**{f"diff:{m}": fr[m] - fr.Experts for m in coders if m != "Experts"})
        cols = [c for c in fr.columns if c in coders or c.startswith("diff:")]
        np.random.seed(SEED)
        draws = []
        for _ in range(N_BOOT):
            if unit == "manifesto":
                keys = fr.source_manifesto.unique()
                b = pd.concat([fr[fr.source_manifesto == k] for k in np.random.choice(keys, len(keys))])
                if b.source_party.nunique() < 2:
                    continue
            else:
                b = pd.concat([g.sample(len(g), replace=True) for _, g in fr.groupby("source_party")])
            draws.append(party_effect(b, cols))
        lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
        est = party_effect(fr, cols)
        ci = dict(zip(cols, zip(lo, hi)))
        for c in coders:
            d = f"diff:{c}"
            rows.append({
                "sentences": set_name, "n": len(fr), "coder": c,
                "group": group.get(c, "human"),
                "party_effect": est[c], "lo": ci[c][0], "hi": ci[c][1],
                "minus_experts": est.get(d, np.nan),
                "minus_experts_lo": ci[d][0] if d in ci else np.nan,
                "minus_experts_hi": ci[d][1] if d in ci else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    benoit, tweets, items = [], [], []
    present = set(pd.read_csv(REPO / "predictions" / "lemens_gallego_tweet_position.csv").model)
    for group, models in GROUPS.items():
        models = [m for m in models if m in present]
        if not models:
            continue
        # one seed per group, so each group's intervals do not depend on the other
        np.random.seed(SEED)
        benoit.append(benoit_results(group, models))
        t, i = tweet_results(group, models)
        tweets.append(t)
        items.append(i)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    items = pd.concat(items)
    for name, frame in {"benoit_masking": pd.concat(benoit), "tweet_masking": pd.concat(tweets),
                        "tweet_items": items,
                        "tweet_masking_scored_only": tweet_scored_only(items)}.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
        if name != "tweet_items":
            print(f"\n== {name}")
            print(frame.round(3).to_string(index=False))
    if (pd.read_csv(REPO / "predictions" / "lemens_gallego_tweet_position.csv").condition == "ignore").any():
        t, b = ignore_results()
        for name, frame in {"ignore_instruction_tweets": t, "ignore_instruction_benoit": b}.items():
            frame.to_csv(OUT / f"{name}.csv", index=False)
            print(f"\n== {name}")
            print(frame.round(3).to_string(index=False))
    if (REPO / "predictions" / "benoit_title_economic_position.csv").exists():
        frame = title_results()
        frame.to_csv(OUT / "title_effects.csv", index=False)
        print("\n== title_effects")
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
