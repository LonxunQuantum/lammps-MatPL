#!/usr/bin/env python3
"""
自动运行全部测试算例，并在结束后生成对比报告。
"""

from pathlib import Path
import argparse
import subprocess
import sys
from test_config import SYSTEMS, ENSEMBLES, LAMMPS_CMD

def collect_cases(base_dir: Path):
    cases = []
    for system in SYSTEMS:
        for ensemble in ENSEMBLES:
            direct_dir = base_dir / system / ensemble
            rerun_dir = base_dir / system / "rerun" / ensemble
            cases.append((f"{system}/{ensemble}", direct_dir))
            cases.append((f"{system}/rerun/{ensemble}", rerun_dir))
    return cases

def run_case(label: str, case_dir: Path, dry_run: bool = False):
    input_file = case_dir / "lmp.in"
    screen_log = case_dir / "screen.log"

    if not case_dir.exists():
        print(f"[跳过] 目录不存在: {label} -> {case_dir}")
        return None

    if not input_file.exists():
        print(f"[跳过] 缺少 lmp.in: {label} -> {input_file}")
        return None

    print(f"[运行] {label}")
    print(f"       cwd = {case_dir}")
    print(f"       cmd = {' '.join(LAMMPS_CMD)}")

    if dry_run:
        return 0

    with open(screen_log, "w", encoding="utf-8") as f:
        f.write(f"# cwd: {case_dir}\n")
        f.write(f"# cmd: {' '.join(LAMMPS_CMD)}\n\n")
        result = subprocess.run(
            LAMMPS_CMD,
            cwd=case_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if result.returncode == 0:
        print(f"[完成] {label}")
    else:
        print(f"[失败] {label} (returncode={result.returncode})")

    return result.returncode

def run_compare(base_dir: Path, dry_run: bool = False):
    compare_script = base_dir / "compare_results.py"
    if not compare_script.exists():
        print(f"[跳过] 未找到 compare_results.py: {compare_script}")
        return None

    cmd = [sys.executable, str(compare_script)]
    print(f"[对比] {' '.join(cmd)}")

    if dry_run:
        return 0

    result = subprocess.run(
        cmd,
        cwd=base_dir,
        check=False,
    )

    if result.returncode == 0:
        print("[完成] 已生成 test_report.txt")
    else:
        print(f"[失败] compare_results.py 执行失败 (returncode={result.returncode})")

    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="运行全部 LAMMPS 测试算例")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将执行的命令，不实际运行"
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="运行结束后不执行 compare_results.py"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="某个算例失败后立即停止"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cases = collect_cases(base_dir)

    print(f"测试根目录: {base_dir}")
    print(f"算例数量: {len(cases)}")

    success = 0
    failed = 0
    skipped = 0

    for label, case_dir in cases:
        ret = run_case(label, case_dir, dry_run=args.dry_run)
        if ret is None:
            skipped += 1
            continue
        if ret == 0:
            success += 1
        else:
            failed += 1
            if args.stop_on_error:
                print("[停止] 检测到失败，终止后续算例")
                break

    print("\n运行统计:")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  跳过: {skipped}")

    if not args.no_compare:
        print("")
        run_compare(base_dir, dry_run=args.dry_run)

if __name__ == "__main__":
    main()