#!/usr/bin/env python3
"""Run one-structure qNEP LAMMPS inference and compare with MatPL-infer output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


STRUCTURE_ORDER = ["part000", "part020", "part040", "part060", "part080"]
DEFAULT_TYPE_MAP = "8 1 6 16 7 9 3"
DEFAULT_LMP_COMMAND = "mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos"
DEFAULT_DUMP_COLUMNS = (
    "id type x y z fx fy fz c_eatom "
    "c_satom[1] c_satom[2] c_satom[3] c_satom[4] c_satom[5] c_satom[6] "
    "c_cbec[1] c_cbec[2] c_cbec[3] c_cbec[4] c_cbec[5] c_cbec[6] "
    "c_cbec[7] c_cbec[8] c_cbec[9]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a run-0 LAMMPS input for selected electrolyte structures, run qNEP, "
            "and compare force/BEC against MatPL-infer/ref-result/test_result_<reference-model>_{ewald,pppm}."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="electrolyte test root containing configs, MatPL-infer, and lmps-infer",
    )
    parser.add_argument(
        "--structure",
        choices=STRUCTURE_ORDER + ["all"],
        default="all",
        help="structure to run/compare",
    )
    parser.add_argument(
        "--kspace",
        choices=["ewald", "pppm", "both"],
        default="both",
        help="kspace mode to run/compare",
    )
    parser.add_argument("--potential", default="../nep5.txt", help="NEP potential file visible from lmps-infer")
    parser.add_argument("--reference-model", default="nep5", help="MatPL reference result prefix, e.g. nep4 or nep5")
    parser.add_argument("--reference-root", type=Path, default=None, help="directory containing test_result_<model>_<kspace> references")
    parser.add_argument("--type-map", default=DEFAULT_TYPE_MAP, help="pair_coeff element/type map")
    parser.add_argument("--threshold", type=float, default=1.0e-5, help="PASS threshold for max abs error")
    parser.add_argument("--lmp-command", default=DEFAULT_LMP_COMMAND, help="LAMMPS launch command")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="only compare existing dumps; do not launch LAMMPS",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue remaining cases if one LAMMPS run or comparison fails",
    )
    parser.add_argument("--work-dir", type=Path, default=None, help="LAMMPS working directory")
    parser.add_argument("--input-dir", type=Path, default=None, help="generated LAMMPS input directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="dump output directory")
    parser.add_argument("--log-dir", type=Path, default=None, help="LAMMPS log/stdout directory")
    parser.add_argument("--result-dir", type=Path, default=None, help="comparison report directory")
    return parser.parse_args()


def atom_count(path: Path) -> int:
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "atoms":
            return int(fields[0])
    raise ValueError(f"cannot find atom count in {path}")


def load_matrix(path: Path, ncols: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(x) for x in line.split()]
        if len(values) != ncols:
            raise ValueError(f"{path}:{lineno} has {len(values)} columns, expected {ncols}")
        rows.append(values)
    return rows


def split_reference(rows: list[list[float]], counts: dict[str, int]) -> dict[str, list[list[float]]]:
    chunks: dict[str, list[list[float]]] = {}
    offset = 0
    for name in STRUCTURE_ORDER:
        n = counts[name]
        chunks[name] = rows[offset : offset + n]
        offset += n
    if offset != len(rows):
        raise ValueError(f"reference row count mismatch: expected {offset}, got {len(rows)}")
    return chunks


def load_xyz_frames(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    lines = path.read_text().splitlines()
    frames: dict[str, list[tuple[float, float, float]]] = {}
    idx = 0
    for name in STRUCTURE_ORDER:
        n = int(lines[idx].strip())
        comment = lines[idx + 1]
        match = re.search(r'Lattice="([^"]+)"', comment)
        if not match:
            raise ValueError(f"cannot find Lattice in frame {name} of {path}")
        lattice = [float(x) for x in match.group(1).split()]
        lengths = (lattice[0], lattice[4], lattice[8])
        idx += 2
        coords = []
        for _ in range(n):
            fields = lines[idx].split()
            coords.append(tuple(float(fields[i + 1]) % lengths[i] for i in range(3)))
            idx += 1
        frames[name] = coords
    if idx != len(lines):
        raise ValueError(f"{path} has trailing lines after expected frames")
    return frames


def coord_key(coord: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(round(x, 6) for x in coord)


def write_lammps_input(
    path: Path,
    config_path: Path,
    kspace: str,
    potential: str,
    type_map: str,
    dump_path: Path,
) -> None:
    dump_parent = dump_path.parent
    text = f"""package kokkos neigh half comm device
newton on

variable        THERMO_FREQ     equal 1
variable        DUMP_FREQ       equal 1

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin
neigh_modify    delay 0 every 1 check yes

box             tilt large
read_data       {config_path}
change_box      all triclinic

pair_style      matpl/nep/kk {potential} kspace {kspace}
pair_coeff      * * {type_map}

compute         eatom all pe/atom
compute         satom all stress/atom NULL virial
compute         virsum all reduce sum c_satom[1] c_satom[2] c_satom[3] c_satom[4] c_satom[5] c_satom[6]

variable        vir_xx equal -c_virsum[1]
variable        vir_yy equal -c_virsum[2]
variable        vir_zz equal -c_virsum[3]
variable        vir_xy equal -c_virsum[4]
variable        vir_xz equal -c_virsum[5]
variable        vir_yz equal -c_virsum[6]

compute         pvir all pressure NULL virial

thermo_style    custom step temp pe ke etotal press vol v_vir_xx v_vir_yy v_vir_zz v_vir_xy v_vir_xz v_vir_yz c_pvir[1] c_pvir[2] c_pvir[3] c_pvir[4] c_pvir[5] c_pvir[6]
thermo          ${{THERMO_FREQ}}
thermo_modify   format float %20.12g flush yes

compute         cbec all qnep/bec/atom

dump            mydump all custom ${{DUMP_FREQ}} {dump_path} {DEFAULT_DUMP_COLUMNS}
dump_modify     mydump sort id
dump_modify     mydump format float %20.12g

run             0
"""
    dump_parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def load_dump(path: Path, xyz_coords: list[tuple[float, float, float]]) -> tuple[list[list[float]], list[list[float]]]:
    lines = path.read_text().splitlines()
    header_idx = None
    columns: list[str] | None = None
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            header_idx = i
            columns = line.split()[2:]
            break
    if header_idx is None or columns is None:
        raise ValueError(f"cannot find ATOMS section in {path}")

    col_index = {name: idx for idx, name in enumerate(columns)}
    needed = ["id", "x", "y", "z", "fx", "fy", "fz"] + [f"c_cbec[{i}]" for i in range(1, 10)]
    missing = [name for name in needed if name not in col_index]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    by_coord: dict[tuple[float, float, float], tuple[list[float], list[float]]] = {}
    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.startswith("ITEM:"):
            break
        fields = line.split()
        coord = tuple(float(fields[col_index[name]]) for name in ("x", "y", "z"))
        force = [float(fields[col_index[name]]) for name in ("fx", "fy", "fz")]
        bec = [float(fields[col_index[f"c_cbec[{i}]"]]) for i in range(1, 10)]
        key = coord_key(coord)
        if key in by_coord:
            raise ValueError(f"duplicate coordinate key {key} in {path}")
        by_coord[key] = (force, bec)

    forces: list[list[float]] = []
    becs: list[list[float]] = []
    for coord in xyz_coords:
        item = by_coord.get(coord_key(coord))
        if item is None:
            raise ValueError(f"{path} missing xyz coordinate {coord}")
        forces.append(item[0])
        becs.append(item[1])
    return forces, becs


def max_abs_error(actual: list[list[float]], reference: list[list[float]]) -> tuple[float, float, tuple[int, int, float, float]]:
    if len(actual) != len(reference):
        raise ValueError(f"row count mismatch: actual {len(actual)}, reference {len(reference)}")
    max_err = -1.0
    max_loc = (0, 0, 0.0, 0.0)
    sum_err = 0.0
    nval = 0
    for atom_idx, (a_row, r_row) in enumerate(zip(actual, reference), 1):
        if len(a_row) != len(r_row):
            raise ValueError(f"column count mismatch at atom {atom_idx}")
        for comp_idx, (actual_value, reference_value) in enumerate(zip(a_row, r_row), 1):
            err = abs(actual_value - reference_value)
            if not math.isfinite(err):
                raise ValueError(f"non-finite error at atom {atom_idx}, component {comp_idx}")
            sum_err += err
            nval += 1
            if err > max_err:
                max_err = err
                max_loc = (atom_idx, comp_idx, actual_value, reference_value)
    return max_err, sum_err / nval if nval else 0.0, max_loc


def run_lammps(command: str, input_path: Path, log_path: Path, stdout_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = shlex.split(command) + ["-log", str(log_path), "-in", str(input_path)]
    with stdout_path.open("w") as stdout:
        result = subprocess.run(cmd, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LAMMPS failed for {input_path}; see {stdout_path}")


def compare_case(
    root: Path,
    structure: str,
    kspace: str,
    dump_path: Path,
    counts: dict[str, int],
    xyz_frames: dict[str, list[tuple[float, float, float]]],
    threshold: float,
    reference_root: Path,
    reference_model: str,
) -> dict[str, object]:
    ref_dir = reference_root / f"test_result_{reference_model}_{kspace}"
    ref_force = split_reference(load_matrix(ref_dir / "inference_force.txt", 3), counts)
    ref_bec = split_reference(load_matrix(ref_dir / "inference_bec.txt", 9), counts)
    actual_force, actual_bec = load_dump(dump_path, xyz_frames[structure])

    force_max, force_mean, force_loc = max_abs_error(actual_force, ref_force[structure])
    bec_max, bec_mean, bec_loc = max_abs_error(actual_bec, ref_bec[structure])
    return {
        "structure": structure,
        "kspace": kspace,
        "force_max_abs": force_max,
        "force_mean_abs": force_mean,
        "force_pass": force_max < threshold,
        "force_max_at": {
            "atom_index_in_matpl_order": force_loc[0],
            "component": force_loc[1],
            "lammps": force_loc[2],
            "matpl": force_loc[3],
        },
        "bec_max_abs": bec_max,
        "bec_mean_abs": bec_mean,
        "bec_pass": bec_max < threshold,
        "bec_max_at": {
            "atom_index_in_matpl_order": bec_loc[0],
            "component": bec_loc[1],
            "lammps": bec_loc[2],
            "matpl": bec_loc[3],
        },
    }


def print_table(results: list[dict[str, object]], threshold: float) -> None:
    print(f"threshold: {threshold:.1e}")
    print("structure kspace force_max_abs force_mean_abs force_status bec_max_abs bec_mean_abs bec_status")
    for row in results:
        print(
            f"{row['structure']} {row['kspace']} "
            f"{row['force_max_abs']:.12e} {row['force_mean_abs']:.12e} "
            f"{'PASS' if row['force_pass'] else 'FAIL'} "
            f"{row['bec_max_abs']:.12e} {row['bec_mean_abs']:.12e} "
            f"{'PASS' if row['bec_pass'] else 'FAIL'}"
        )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    work_dir = (args.work_dir or Path(__file__).resolve().parent).resolve()
    reference_root = (args.reference_root or root / "MatPL-infer" / "ref-result").resolve()
    input_dir = (args.input_dir or work_dir / "auto_inputs").resolve()
    output_dir = (args.output_dir or work_dir / "auto_outputs").resolve()
    log_dir = (args.log_dir or work_dir / "auto_logs").resolve()
    result_dir = (args.result_dir or work_dir / "compare_results").resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    structures = STRUCTURE_ORDER if args.structure == "all" else [args.structure]
    kspaces = ["ewald", "pppm"] if args.kspace == "both" else [args.kspace]

    counts = {name: atom_count(root / "configs" / f"{name}.lmp") for name in STRUCTURE_ORDER}
    xyz_frames = load_xyz_frames(root / "MatPL-infer" / "train.xyz")

    results: list[dict[str, object]] = []
    failures = 0
    for structure in structures:
        for kspace in kspaces:
            case = f"{structure}_{kspace}"
            input_path = input_dir / f"{case}.lmp"
            dump_path = output_dir / kspace / f"{structure}.peratom.dump"
            log_path = log_dir / f"{case}.log"
            stdout_path = log_dir / f"{case}.out"
            config_path = (root / "configs" / f"{structure}.lmp").resolve()
            try:
                write_lammps_input(input_path, config_path, kspace, args.potential, args.type_map, dump_path)
                if not args.skip_run:
                    print(f"[run] {case}", flush=True)
                    run_lammps(args.lmp_command, input_path, log_path, stdout_path, work_dir)
                print(f"[compare] {case}", flush=True)
                results.append(
                    compare_case(
                        root,
                        structure,
                        kspace,
                        dump_path,
                        counts,
                        xyz_frames,
                        args.threshold,
                        reference_root,
                        args.reference_model,
                    )
                )
            except Exception as exc:
                failures += 1
                print(f"[error] {case}: {exc}", file=sys.stderr, flush=True)
                if not args.keep_going:
                    raise

    csv_path = result_dir / "compare_summary.csv"
    json_path = result_dir / "compare_summary.json"
    if results:
        with csv_path.open("w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "structure",
                    "kspace",
                    "force_max_abs",
                    "force_mean_abs",
                    "force_pass",
                    "bec_max_abs",
                    "bec_mean_abs",
                    "bec_pass",
                ],
            )
            writer.writeheader()
            for row in results:
                writer.writerow({key: row[key] for key in writer.fieldnames})
        json_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print_table(results, args.threshold)
        print(f"\nwrote {csv_path}")
        print(f"wrote {json_path}")

    accuracy_failures = sum(1 for row in results if not row["force_pass"] or not row["bec_pass"])
    if failures or accuracy_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
