#!/usr/bin/env python3
"""Compare MatPL inference outputs with saved reference outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CASES = [
    "test_result_nep4_ewald",
    "test_result_nep4_pppm",
    "test_result_nep5_ewald",
    "test_result_nep5_pppm",
]

QUANTITIES = [
    ("total_energy", "inference_total_energy.txt", 1),
    ("force", "inference_force.txt", 3),
    ("bec", "inference_bec.txt", 9),
]


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare MatPL result directories against MatPL-infer/ref-result.")
    parser.add_argument("--matpl-root", type=Path, default=base, help="MatPL-infer directory containing generated results")
    parser.add_argument("--reference-root", type=Path, default=base / "ref-result", help="directory containing reference results")
    parser.add_argument("--threshold", type=float, default=1.0e-5, help="PASS threshold for max abs error")
    parser.add_argument("--output", type=Path, default=Path("../validation_results/matpl_ref_summary.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("../validation_results/matpl_ref_summary.json"))
    return parser.parse_args()


def load_matrix(path: Path, ncols: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(x) for x in line.split()]
        if len(values) < ncols:
            raise ValueError(f"{path}:{lineno} has {len(values)} columns, expected at least {ncols}")
        rows.append(values[:ncols])
    return rows


def max_abs_error(actual: list[list[float]], reference: list[list[float]]) -> tuple[float, float, dict[str, float | int]]:
    if len(actual) != len(reference):
        raise ValueError(f"row count mismatch: actual={len(actual)}, reference={len(reference)}")
    max_err = -1.0
    sum_err = 0.0
    nvalue = 0
    max_at: dict[str, float | int] = {}
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
                max_at = {
                    "row": row_idx,
                    "column": col_idx,
                    "actual": actual_value,
                    "reference": ref_value,
                    "abs_error": err,
                }
    return max_err, sum_err / nvalue if nvalue else 0.0, max_at


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    details = {
        "threshold": args.threshold,
        "matpl_root": str(args.matpl_root),
        "reference_root": str(args.reference_root),
        "results": [],
    }

    for case in CASES:
        for quantity, filename, ncols in QUANTITIES:
            actual_path = args.matpl_root / case / filename
            reference_path = args.reference_root / case / filename
            actual = load_matrix(actual_path, ncols)
            reference = load_matrix(reference_path, ncols)
            max_err, mean_err, max_at = max_abs_error(actual, reference)
            passed = max_err < args.threshold
            row = {
                "case": case,
                "quantity": quantity,
                "rows": len(actual),
                "columns": ncols,
                "max_abs_error": max_err,
                "mean_abs_error": mean_err,
                "pass": passed,
            }
            rows.append(row)
            details["results"].append({**row, "max_at": max_at})

    with args.output.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["case", "quantity", "rows", "columns", "max_abs_error", "mean_abs_error", "pass"],
        )
        writer.writeheader()
        writer.writerows(rows)

    args.json_output.write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")

    print(f"threshold: {args.threshold:.1e}")
    print("case quantity rows columns max_abs_error mean_abs_error status")
    for row in rows:
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"{row['case']} {row['quantity']} {row['rows']} {row['columns']} "
            f"{row['max_abs_error']:.12e} {row['mean_abs_error']:.12e} {status}"
        )
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")

    return 0 if all(row["pass"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
