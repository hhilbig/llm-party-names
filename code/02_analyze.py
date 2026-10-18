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
results/tweet_items.csv.
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


def main():
    OUT.mkdir(exist_ok=True)
    benoit, tweets, items = [], [], []
    for group, models in GROUPS.items():
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


if __name__ == "__main__":
    main()
