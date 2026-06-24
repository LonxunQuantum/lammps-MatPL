#!/usr/bin/env python3
"""
清理测试输出并备份测试报告：
1) 若存在 test_report.txt,则备份到 backups/test_report_时间戳.txt
2) 清理 HfO2 / water 在 NPT、NVT(含 rerun)目录下的历史输出文件
"""

from pathlib import Path
from datetime import datetime
import argparse
from test_config import SYSTEMS, ENSEMBLES

def backup_report(base_dir: Path, dry_run: bool = False):
    report = base_dir / "test_report.txt"
    if not report.exists():
        print("[跳过] 未找到 test_report.txt")
        return

    backup_dir = base_dir / "backups"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"test_report_{ts}.txt"

    if dry_run:
        print(f"[DRY-RUN] 备份: {report} -> {backup_path}")
        return

    backup_dir.mkdir(parents=True, exist_ok=True)
    report.rename(backup_path)
    print(f"[完成] 已备份报告: {backup_path}")

def collect_targets(base_dir: Path):
    targets = []

    for system in SYSTEMS:
        for ensemble in ENSEMBLES:
            direct_dir = base_dir / system / ensemble
            rerun_dir = base_dir / system / "rerun" / ensemble

            for run_dir in [direct_dir, rerun_dir]:
                targets.append(run_dir / "log.lammps")
                targets.append(run_dir / "lammps.dump")

                for extra in ["log.cite", "screen", "screen.log", "stdout.txt", "stderr.txt"]:
                    targets.append(run_dir / extra)

    return targets

def clean_outputs(base_dir: Path, dry_run: bool = False):
    targets = collect_targets(base_dir)
    removed = 0
    missing = 0

    for p in targets:
        if p.exists():
            if dry_run:
                print(f"[DRY-RUN] 删除: {p}")
            else:
                p.unlink()
                print(f"[删除] {p}")
            removed += 1
        else:
            missing += 1

    print(f"\n统计: 删除 {removed} 个文件, 未找到 {missing} 个目标文件")

def main():
    parser = argparse.ArgumentParser(description="清理测试输出并备份 test_report.txt")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的操作，不实际修改文件"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    print(f"工作目录: {base_dir}")

    backup_report(base_dir, dry_run=args.dry_run)
    clean_outputs(base_dir, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
