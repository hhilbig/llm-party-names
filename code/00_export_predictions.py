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

Usage: python3 code/00_export_predictions.py /path/to/polsci-open-bench
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
        "models": {
            "gemini_3_8_flash": "Gemini 3.8 Flash",
            "deepseek_v4_pro": "DeepSeek V4 Pro",
            "gpt_5_6_terra": "GPT-5.6 Terra",
            "claude_sonnet_5": "Claude Sonnet 5",
        },
    },
}
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
                for condition in ("orig", "masked"):
                    name = task if condition == "orig" else f"{task}_masked"
                    f = pd.read_csv(run[condition] / key / "tasks" / f"{name}.csv")
                    frames.append(pd.DataFrame({
                        "group": group, "model": label, "model_id": f.model_id,
                        "condition": condition, "item_id": f.item_id,
                        "prediction": f[col], "parse_error": f.parse_error,
                        "raw_output": f.raw_content_preview,
                    }))
        out = pd.concat(frames, ignore_index=True)
        key = ["group", "model", "condition", "item_id"]
        assert not out.duplicated(key).any()
        out.to_csv(REPO / "predictions" / f"{task}.csv", index=False)
        print(task, len(out), "rows;", out.groupby(["group", "condition"]).size().to_dict())


if __name__ == "__main__":
    main()
