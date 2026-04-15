#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_score(path: Path) -> float | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("score", "mean_chosen_score", "global_mean_of_prompt_means"):
        if key in obj and isinstance(obj[key], (int, float)):
            return float(obj[key])
    return None


def main() -> None:
    p = argparse.ArgumentParser("Aggregate summary.json and eval json into CSV")
    p.add_argument("--input_root", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    args = p.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for js in sorted(input_root.rglob("*.json")):
        sc = _read_score(js)
        if sc is None:
            continue
        rel = js.relative_to(input_root)
        parts = rel.parts
        method = parts[0] if len(parts) > 1 else "unknown"
        rows.append(
            {
                "method": method,
                "path": str(rel),
                "score": sc,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "path", "score"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
