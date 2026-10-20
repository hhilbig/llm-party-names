#!/usr/bin/env python3
"""
Collect the model predictions from the polsci-open-bench run folders into one
long file per corpus in predictions/.

The runs themselves live in polsci-open-bench
(https://github.com/hhilbig/polsci-open-bench):
- open-weight models, text as written: output/sidecar/task_ext_20261002
- open-weight models, names masked: output/sidecar/task_ext_20261010_masked
  (config experiments/task_ext_20261010_masked.yaml)
- commercial models, both versions: output/sidecar/task_ext_20261011_masked_api
  (code/run_commercial_models.py in this repository)
- text as written plus an instruction to ignore names (condition "ignore"):
  output/sidecar/task_ext_20261018_ignore (open-weight, config
  experiments/task_ext_20261018_ignore.yaml) and
  output/sidecar/task_ext_20261018_ignore_api (commercial)
- manifesto title hidden vs shown (economic scale): output/sidecar/task_ext_20261019_title
  (open-weight, config experiments/task_ext_20261019_title.yaml, all 3,682
  sentences) and output/sidecar/task_ext_20261019_title_api/full (commercial,
  1,050 of them). Sentences from data/benoit_title_economic_position.csv, built
  by code/build_party_title_tasks.py in polsci-open-bench.

With the Benoit et al. coding file as second argument (coding_all_long_2014-03-14.csv
from github.com/kbenoit/CSTA-APSR, "Data - Created/coding_all_long.zip"), also
writes the human economic codes of those sentences, with whether the coder saw
the manifesto title, to data/benoit_title_human_codes.csv.

Usage: python3 code/00_export_predictions.py /path/to/polsci-open-bench [coding_all_long.csv]
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BENCH = Path(sys.argv[1]).resolve()
SIDE = BENCH / "output" / "sidecar"

RUNS = {
    "open-weight": {
        "orig": SIDE / "task_ext_20261002",
        "masked": SIDE / "task_ext_20261010_masked",
        "ignore": SIDE / "task_ext_20261018_ignore",
        "ignore_suffix": "_ignore",
        "models": {
            "gemma4_31b_it_qat_w4a16": "Gemma 4 31B",
            "qwen3_6_27b_fp8": "Qwen3.6 27B",
            "qwen3_6_35b_a3b_fp8": "Qwen3.6 35B-A3B",
            "qwen3_30b_a3b_instruct_2507_fp8": "Qwen3 30B",
        },
    },
    "commercial": {
        "orig": SIDE / "task_ext_20261011_masked_api" / "full" / "orig",
        "masked": SIDE / "task_ext_20261011_masked_api" / "full" / "masked",
        "ignore": SIDE / "task_ext_20261018_ignore_api" / "full" / "ignore",
        "models": {
            "gemini_3_8_flash": "Gemini 3.8 Flash",
            "deepseek_v4_pro": "DeepSeek V4 Pro",
            "gpt_5_6_terra": "GPT-5.6 Terra",
            "claude_sonnet_5": "Claude Sonnet 5",
        },
    },
    # added 2026-10-18 as a current flagship; same folders and items as the other
    # commercial models, reasoning at minimal effort because it cannot be disabled
    "flagship": {
        "orig": SIDE / "task_ext_20261011_masked_api" / "full" / "orig",
        "masked": SIDE / "task_ext_20261011_masked_api" / "full" / "masked",
        "ignore": SIDE / "task_ext_20261018_ignore_api" / "full" / "ignore",
        "models": {"gpt_6_1_sol": "GPT-6.1 Sol"},
    },
    # added 2026-10-18: newer open-weight models, all three versions in one job each
    # (config experiments/task_ext_20261018_newopen.yaml)
    "open-new": {
        "orig": SIDE / "task_ext_20261018_newopen",
        "masked": SIDE / "task_ext_20261018_newopen",
        "ignore": SIDE / "task_ext_20261018_newopen",
        "ignore_suffix": "_ignore",
        "models": {"qwen3_8_27b_fp8": "Qwen3.8 27B",
                   "nemotron3_5_lightning_fp8": "Nemotron 3.5 Lightning 30B"},
    },
}
TITLE_RUNS = {
    "open-weight": (SIDE / "task_ext_20261019_title",
                    {**RUNS["open-weight"]["models"], **RUNS["open-new"]["models"]}),
    "commercial": (SIDE / "task_ext_20261019_title_api" / "full",
                   {**RUNS["commercial"]["models"], **RUNS["flagship"]["models"]}),
}
TITLE_TASK = "benoit_title_economic_position"
TASKS = {
    "benoit_manifesto_economic_position": "pred_economic_position",
    "benoit_manifesto_social_position": "pred_social_position",
    "lemens_gallego_tweet_position": "pred_score",
}


def main():
    (REPO / "predictions").mkdir(exist_ok=True)
    for task, col in TASKS.items():
        frames = []
        for group, run in RUNS.items():
            for key, label in run["models"].items():
                for condition in [c for c in ("orig", "masked", "ignore") if c in run and run[c].exists()]:
                    name = {"masked": f"{task}_masked",
                            "ignore": task + run.get("ignore_suffix", "")}.get(condition, task)
                    path = run[condition] / key / "tasks" / f"{name}.csv"
                    if not path.exists():
                        print(f"missing, skipped: {path}")
                        continue
                    f = pd.read_csv(path)
                    frames.append(pd.DataFrame({
                        "group": run.get("group", group), "model": label, "model_id": f.model_id,
                        "condition": condition, "item_id": f.item_id,
                        "prediction": f[col], "parse_error": f.parse_error,
                        "raw_output": f.raw_content_preview,
                    }))
        out = pd.concat(frames, ignore_index=True)
        key = ["group", "model", "condition", "item_id"]
        assert not out.duplicated(key).any()
        out.to_csv(REPO / "predictions" / f"{task}.csv", index=False)
        print(task, len(out), "rows;", out.groupby(["group", "condition"]).size().to_dict())

    export_title()


def export_title():
    frames = []
    for group, (run, models) in TITLE_RUNS.items():
        for key, label in models.items():
            for condition in ["hidden", "shown"]:
                f = pd.read_csv(run / key / "tasks" / f"{TITLE_TASK}_{condition}.csv")
                frames.append(pd.DataFrame({
                    "group": group, "model": label, "model_id": f.model_id, "condition": condition,
                    "item_id": f.item_id, "prediction": f.pred_economic_position,
                    "parse_error": f.parse_error, "raw_output": f.raw_content_preview,
                }))
    out = pd.concat(frames, ignore_index=True)
    assert not out.duplicated(["model", "condition", "item_id"]).any()
    out.to_csv(REPO / "predictions" / f"{TITLE_TASK}.csv", index=False)
    print(TITLE_TASK, len(out), "rows;", out.groupby(["group", "condition"]).size().to_dict())

    data = pd.read_csv(BENCH / "data" / f"{TITLE_TASK}.csv")
    data.to_csv(REPO / "data" / f"{TITLE_TASK}.csv", index=False)
    if len(sys.argv) > 2:
        c = pd.read_csv(sys.argv[2], low_memory=False,
                        usecols=["source", "coderid", "sentenceid", "scale", "title_shown", "code"])
        c = c[c.scale.eq("Economic") & c.code.isin([-2, -1, 0, 1, 2]) & c.sentenceid.isin(data.sentenceid)
              & c.source.isin(["Experts", "SemiExperts"])]
        c.drop(columns="scale").to_csv(REPO / "data" / "benoit_title_human_codes.csv", index=False)
        print("human codes:", len(c), "rows;", c.groupby(["source", "title_shown"]).size().to_dict())


if __name__ == "__main__":
    main()
