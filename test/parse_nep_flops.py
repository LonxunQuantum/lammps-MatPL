#!/usr/bin/env python3
"""
parse_nep_flops.py - 从 LAMMPS log 中抽取 NEP kernel FLOP/耗时统计。

用法：
    python3 parse_nep_flops.py <logfile> [--csv]

每个 MD 步产生 5 段报告块，脚本将：
  - 找出所有完整的 5-段组合（一步一组）
  - 计算每段的平均耗时、平均 TFLOPs、平均 TFLOP/s
  - 输出各段汇总、合计行、以及整体 TFLOP/s
  - 若指定 --csv，同时输出 CSV 格式

5 个阶段对应的 log 标头：
  [NEP_FLOP_STATS_REPORT_2B_DESCRIPTORS]
  [NEP_FLOP_STATS_REPORT_3B_DESCRIPTORS]
  [NEP_FLOP_STATS_REPORT_ANN_TC_FUSED]
  [NEP_FLOP_STATS_REPORT_2B_BACKWARD]
  [NEP_FLOP_STATS_REPORT_3B_BACKWARD]
"""

import re
import sys
import argparse
from collections import defaultdict

# ─── 常量 ───────────────────────────────────────────────────────────────────

BLOCK_KEYS = [
    "2B_DESCRIPTORS",
    "3B_DESCRIPTORS",
    "ANN_TC_FUSED",
    "2B_BACKWARD",
    "3B_BACKWARD",
]

STAGE_ID = {
    "2B_DESCRIPTORS": "2B_DESC",
    "3B_DESCRIPTORS": "3B_DESC",
    "ANN_TC_FUSED":   "ANN_TC",
    "2B_BACKWARD":    "2B_BWD",
    "3B_BACKWARD":    "3B_BWD",
}

STAGE_LABEL = {
    "2B_DESC": "2B Descriptors",
    "3B_DESC": "3B Descriptors",
    "ANN_TC":  "ANN TC (all types)",
    "2B_BWD":  "2B Backward",
    "3B_BWD":  "3B Backward",
}

STAGE_ORDER = ["2B_DESC", "3B_DESC", "ANN_TC", "2B_BWD", "3B_BWD"]

# ─── 解析 ────────────────────────────────────────────────────────────────────

def split_blocks(text):
    """将 log 文本分割成 (block_key, block_text) 列表。"""
    header_re = re.compile(
        r'\[NEP_FLOP_STATS_REPORT_(2B_DESCRIPTORS|3B_DESCRIPTORS|ANN_TC_FUSED|2B_BACKWARD|3B_BACKWARD)\]'
    )
    positions = [(m.start(), m.group(1)) for m in header_re.finditer(text)]
    blocks = []
    for i, (start, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        blocks.append((key, text[start:end]))
    return blocks


def group_into_steps(blocks):
    """将连续的 5 段合并为一个 step；不完整的跳过。"""
    steps = []
    i = 0
    while i <= len(blocks) - 5:
        window = blocks[i:i + 5]
        keys = [b[0] for b in window]
        if keys == BLOCK_KEYS:
            steps.append({b[0]: b[1] for b in window})
            i += 5
        else:
            i += 1
    return steps


def extract_float(pattern, text, default=None):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


def parse_step(step_blocks):
    """从一个 step 的 5 段文本中提取 (time_ms, tflops)，返回 dict。"""
    result = {}

    # ── 2B DESC ──────────────────────────────────────────────────────────
    b = step_blocks["2B_DESCRIPTORS"]
    result["2B_DESC"] = (
        extract_float(r'Kernel Time:\s+([\d.]+)\s+ms', b),
        extract_float(r'Total Workload:\s+([\d.]+)\s+TFLOPs', b),
    )

    # ── 3B DESC ──────────────────────────────────────────────────────────
    b = step_blocks["3B_DESCRIPTORS"]
    result["3B_DESC"] = (
        extract_float(r'Kernel Time:\s+([\d.]+)\s+ms', b),
        extract_float(r'Total Workload:\s+([\d.]+)\s+TFLOPs', b),
    )

    # ── ANN TC（合并所有 type）────────────────────────────────────────────
    b = step_blocks["ANN_TC_FUSED"]
    # ANN TC header 用小写 "Kernel time"，且无 /step 后缀
    ann_times = [float(x) for x in re.findall(r'Kernel time:\s+([\d.]+)\s+ms', b)]
    ann_flops = [float(x) for x in re.findall(r'Math FLOPs\s+\(useful\):\s+([\d.]+)\s+TFLOPs', b)]
    if ann_times and ann_flops:
        result["ANN_TC"] = (sum(ann_times), sum(ann_flops))
    else:
        result["ANN_TC"] = (None, None)

    # ── 2B BWD ───────────────────────────────────────────────────────────
    b = step_blocks["2B_BACKWARD"]
    result["2B_BWD"] = (
        extract_float(r'Kernel Time:\s+([\d.]+)\s+ms', b),
        extract_float(r'Total Workload:\s+([\d.]+)\s+TFLOPs', b),
    )

    # ── 3B BWD ───────────────────────────────────────────────────────────
    b = step_blocks["3B_BACKWARD"]
    result["3B_BWD"] = (
        extract_float(r'Kernel Time:\s+([\d.]+)\s+ms', b),
        extract_float(r'Total Workload:\s+([\d.]+)\s+TFLOPs', b),
    )

    return result


# ─── 报告 ────────────────────────────────────────────────────────────────────

def compute_stats(all_step_data):
    """计算每个阶段在所有 step 上的平均值。"""
    stats = {}
    for stage in STAGE_ORDER:
        samples = [(d[stage][0], d[stage][1]) for d in all_step_data
                   if d[stage][0] is not None and d[stage][1] is not None]
        if not samples:
            stats[stage] = None
            continue
        n = len(samples)
        avg_t = sum(s[0] for s in samples) / n
        avg_f = sum(s[1] for s in samples) / n
        perf  = avg_f / (avg_t * 1e-3) if avg_t > 0 else 0.0
        stats[stage] = {
            "avg_time_ms": avg_t,
            "avg_tflops":  avg_f,
            "avg_tflops_s": perf,
            "n_samples": n,
        }
    return stats


def print_table(stats, n_steps):
    W = 76
    print("=" * W)
    print(f"  NEP FLOP Statistics  (averaged over {n_steps} MD step(s))")
    print("=" * W)
    header = f"  {'Stage':<22} {'Time (ms)':>10} {'TFLOPs':>12} {'TFLOP/s':>10}"
    print(header)
    print("-" * W)

    total_t  = 0.0
    total_f  = 0.0

    for stage in STAGE_ORDER:
        s = stats[stage]
        label = STAGE_LABEL[stage]
        if s is None:
            print(f"  {label:<22} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
            continue
        t  = s["avg_time_ms"]
        f  = s["avg_tflops"]
        p  = s["avg_tflops_s"]
        print(f"  {label:<22} {t:>10.3f} {f:>12.6f} {p:>10.3f}")
        total_t += t
        total_f += f

    print("-" * W)
    total_p = total_f / (total_t * 1e-3) if total_t > 0 else 0.0
    print(f"  {'TOTAL':<22} {total_t:>10.3f} {total_f:>12.6f} {total_p:>10.3f}")
    print("=" * W)
    print(f"  Unit: Time=ms/step  |  TFLOPs=per MD step  |  TFLOP/s=avg throughput")
    print()


def print_csv(stats, n_steps):
    print("# NEP FLOP Statistics CSV")
    print(f"# Averaged over {n_steps} MD step(s)")
    print("stage,avg_time_ms,avg_tflops,avg_tflops_s")
    total_t = 0.0
    total_f = 0.0
    for stage in STAGE_ORDER:
        s = stats[stage]
        if s is None:
            print(f"{STAGE_LABEL[stage]},N/A,N/A,N/A")
            continue
        print(f"{STAGE_LABEL[stage]},{s['avg_time_ms']:.3f},{s['avg_tflops']:.6f},{s['avg_tflops_s']:.3f}")
        total_t += s["avg_time_ms"]
        total_f += s["avg_tflops"]
    total_p = total_f / (total_t * 1e-3) if total_t > 0 else 0.0
    print(f"TOTAL,{total_t:.3f},{total_f:.6f},{total_p:.3f}")


def print_per_step(all_step_data):
    """逐步打印耗时和 TFLOPs（多步时附加）。"""
    print(f"  {'Step':>4}  " + "  ".join(f"{STAGE_LABEL[s][:8]:>8}" for s in STAGE_ORDER) + "   Total(ms)")
    for i, step_data in enumerate(all_step_data):
        times = [step_data[s][0] for s in STAGE_ORDER]
        valid = [t for t in times if t is not None]
        total = sum(valid)
        row = f"  {i+1:>4}  " + "  ".join(
            f"{t:>8.3f}" if t is not None else f"{'N/A':>8}"
            for t in times
        )
        print(row + f"   {total:>8.3f}")
    print()


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从 LAMMPS log 中抽取 NEP FLOP/耗时统计。"
    )
    parser.add_argument("logfile", help="LAMMPS log 文件路径")
    parser.add_argument("--csv", action="store_true", help="同时输出 CSV 格式")
    parser.add_argument("--per-step", action="store_true",
                        help="打印逐步耗时明细（多步时有用）")
    args = parser.parse_args()

    with open(args.logfile, "r", errors="replace") as f:
        text = f.read()

    blocks = split_blocks(text)
    if not blocks:
        print("未找到任何 NEP_FLOP_STATS_REPORT 块。", file=sys.stderr)
        sys.exit(1)

    steps = group_into_steps(blocks)
    if not steps:
        print("未找到完整的 5-段报告组（2B_DESC→3B_DESC→ANN→2B_BWD→3B_BWD）。",
              file=sys.stderr)
        print(f"找到的块类型顺序：{[b[0] for b in blocks[:20]]}", file=sys.stderr)
        sys.exit(1)

    print(f"[ 找到 {len(steps)} 个完整 MD 步的 FLOP 报告 ]\n")

    all_step_data = [parse_step(s) for s in steps]

    if args.per_step and len(all_step_data) > 1:
        print("各步耗时（ms）：")
        print_per_step(all_step_data)

    stats = compute_stats(all_step_data)
    print_table(stats, len(all_step_data))

    if args.csv:
        print_csv(stats, len(all_step_data))


if __name__ == "__main__":
    main()
