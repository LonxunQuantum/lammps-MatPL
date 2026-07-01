#!/usr/bin/env python3
"""Compare GPUMD inference outputs with MatPL inference outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GPUMD force_train.out/bec_train.out against MatPL "
            "inference_force.txt/inference_bec.txt."
        )
    )
    parser.add_argument(
        "--matpl-dir",
        type=Path,
        default=Path("test_result_nep4_ewald"),
        help="MatPL result directory containing inference_force.txt and inference_bec.txt",
    )
    parser.add_argument("--gpumd-dir", type=Path, default=Path("../GPUMD"), help="GPUMD result directory")
    parser.add_argument("--threshold", type=float, default=1.0e-4, help="PASS threshold for max abs error")
    parser.add_argument("--output", type=Path, default=Path("compare_gpumd_matpl_summary.csv"), help="CSV summary path")
    parser.add_argument("--json-output", type=Path, default=Path("compare_gpumd_matpl_summary.json"), help="JSON detail path")
    return parser.parse_args()


def load_numeric_columns(path: Path, ncols: int, usecols: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [float(x) for x in line.split()]
        if len(fields) < ncols:
            raise ValueError(f"{path}:{lineno} has {len(fields)} columns, expected at least {ncols}")
        rows.append(fields[:usecols])
    return rows


def max_abs_error(actual: list[list[float]], reference: list[list[float]]) -> tuple[float, float, dict[str, float | int]]:
    if len(actual) != len(reference):
        raise ValueError(f"row count mismatch: actual={len(actual)}, reference={len(reference)}")

    max_err = -1.0
    sum_err = 0.0
    nvalue = 0
    max_loc: dict[str, float | int] = {}

    for row_idx, (actual_row, ref_row) in enumerate(zip(actual, reference), 1):
        if len(actual_row) != len(ref_row):
            raise ValueError(f"column count mismatch at row {row_idx}: actual={len(actual_row)}, reference={len(ref_row)}")
        for col_idx, (actual_value, ref_value) in enumerate(zip(actual_row, ref_row), 1):
            err = abs(actual_value - ref_value)
            if not math.isfinite(err):
                raise ValueError(f"non-finite error at row {row_idx}, col {col_idx}")
            sum_err += err
            nvalue += 1
            if err > max_err:
                max_err = err
                max_loc = {
                    "row": row_idx,
                    "column": col_idx,
                    "gpumd": actual_value,
                    "matpl": ref_value,
                    "abs_error": err,
                }

    return max_err, sum_err / nvalue if nvalue else 0.0, max_loc


def main() -> int:
    args = parse_args()

    gpumd_force = load_numeric_columns(args.gpumd_dir / "force_train.out", ncols=3, usecols=3)
    gpumd_bec = load_numeric_columns(args.gpumd_dir / "bec_train.out", ncols=9, usecols=9)
    matpl_force = load_numeric_columns(args.matpl_dir / "inference_force.txt", ncols=3, usecols=3)
    matpl_bec = load_numeric_columns(args.matpl_dir / "inference_bec.txt", ncols=9, usecols=9)

    force_max, force_mean, force_loc = max_abs_error(gpumd_force, matpl_force)
    bec_max, bec_mean, bec_loc = max_abs_error(gpumd_bec, matpl_bec)

    rows = [
        {
            "quantity": "force",
            "rows": len(gpumd_force),
            "columns": 3,
            "max_abs_error": force_max,
            "mean_abs_error": force_mean,
            "pass": force_max < args.threshold,
        },
        {
            "quantity": "bec",
            "rows": len(gpumd_bec),
            "columns": 9,
            "max_abs_error": bec_max,
            "mean_abs_error": bec_mean,
            "pass": bec_max < args.threshold,
        },
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["quantity", "rows", "columns", "max_abs_error", "mean_abs_error", "pass"])
        writer.writeheader()
        writer.writerows(rows)

    detail = {
        "threshold": args.threshold,
        "matpl_dir": str(args.matpl_dir),
        "gpumd_dir": str(args.gpumd_dir),
        "force": {**rows[0], "max_at": force_loc},
        "bec": {**rows[1], "max_at": bec_loc},
    }
    args.json_output.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")

    print(f"threshold: {args.threshold:.1e}")
    print("quantity rows columns max_abs_error mean_abs_error status")
    for row in rows:
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"{row['quantity']} {row['rows']} {row['columns']} "
            f"{row['max_abs_error']:.12e} {row['mean_abs_error']:.12e} {status}"
        )
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")

    return 0 if all(row["pass"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
