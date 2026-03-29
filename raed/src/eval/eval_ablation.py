from __future__ import annotations

import argparse
import json
from pathlib import Path

from raed.src.eval.common import save_table_csv
from raed.src.utils.logging import dump_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Stage B/B2 ablation metrics")
    parser.add_argument("--decoder_metrics", type=str, default="raed/outputs/eval_restoration/restoration_metrics.json")
    parser.add_argument("--option1_metrics", type=str, default="raed/outputs/eval_diffusion_refinement/option1_metrics.json")
    parser.add_argument("--option2_metrics", type=str, default="raed/outputs/eval_diffusion_refinement/option2_metrics.json")
    parser.add_argument("--out_dir", type=str, default="raed/outputs/eval_ablation")
    return parser.parse_args()


def _read_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    rows = []

    decoder = _read_json(args.decoder_metrics)
    rows.append({"method": "decoder_only", **{k: v for k, v in decoder.items() if isinstance(v, (int, float))}})

    option1 = _read_json(args.option1_metrics)
    rows.append({"method": "diffusion_option1", **{k: v for k, v in option1.items() if isinstance(v, (int, float))}})

    option2 = _read_json(args.option2_metrics)
    rows.append({"method": "diffusion_option2", **{k: v for k, v in option2.items() if isinstance(v, (int, float))}})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table_csv(str(out_dir / "ablation_metrics.csv"), rows)
    dump_metrics_json(out_dir / "ablation_metrics.json", {"rows": rows})
    print({"ablation_csv": str(out_dir / "ablation_metrics.csv"), "rows": rows})


if __name__ == "__main__":
    main()
