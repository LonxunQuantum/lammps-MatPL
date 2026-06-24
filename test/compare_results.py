#!/usr/bin/env python3
"""
LAMMPS 结果对比脚本（含阈值判断及最终统计）
对比六个体系 (HfO2, water, Cu, W, WMoTaV, graphene) 在 NPT 和 NVT 系综下的：
  - log.lammps 中的热力学量（包括每原子势能和维里张量）
  - dump 文件（统一命名为 lammps.dump）中的原子受力分量 (fx, fy, fz)
基准结果位于 ref_results 目录，待测结果位于各系综目录及 rerun 目录。
输出测试报告 test_report.txt，并依据阈值给出 PASS/FAIL 结论，末尾附总结。
"""

import os
import numpy as np
import pandas as pd
from test_config import SYSTEMS, ENSEMBLES

# ==================== 阈值定义 ====================
# 基于典型运行结果设定的合理阈值，超过此值视为失败
THRESHOLDS = {
    'Temp':           1e-3,   # 温度差异
    'v_pe_peratom':   1e-5,   # 每原子势能
    # 维里张量各分量
    'v_W_xx': 2e-2, 'v_W_yy': 2e-2, 'v_W_zz': 2e-2,
    'v_W_xy': 2e-2, 'v_W_xz': 2e-2, 'v_W_yz': 2e-2,
    # 受力分量（按 dump 中的最大绝对值）
    'fx': 1e-2, 'fy': 1e-2, 'fz': 1e-2,
}

# ==================== 文件解析函数（增强版） ====================

def parse_log(filepath):
    """
    解析 LAMMPS log 文件，提取 thermo 数据。
    假设 log 文件中第一个以 'Step' 开头的行为列名行，
    后续连续数字行为数据，直到遇到空行或非数字行。
    返回 pandas DataFrame。
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 定位列名行
    header_line = None
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Step'):
            header_line = line.strip()
            header_idx = i
            break
    if header_line is None:
        raise ValueError(f"未找到 'Step' 行: {filepath}")

    columns = header_line.split()
    # 收集数据行
    data_lines = []
    for line in lines[header_idx+1:]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped[0].isdigit() or stripped[0] == '-':
            data_lines.append(stripped)
        else:
            break

    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) == len(columns):
            data.append([float(x) for x in parts])
        else:
            continue
    return pd.DataFrame(data, columns=columns)

def parse_dump(filepath):
    """
    解析 LAMMPS dump 文件，返回列表，每个元素为 (step, atoms_df)。
    自动跳过空行，并对错误位置进行提示。
    atoms_df 包含原子数据，按 id 排序。
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    steps = []
    i = 0
    nlines = len(lines)
    while i < nlines:
        # 跳过空行
        while i < nlines and lines[i].strip() == '':
            i += 1
        if i >= nlines:
            break
        line = lines[i].strip()
        if line.startswith('ITEM: TIMESTEP'):
            i += 1
            # 跳过空行后读取步数
            while i < nlines and lines[i].strip() == '':
                i += 1
            if i >= nlines:
                raise ValueError(f"文件结束，未找到 TIMESTEP 数值")
            step = int(lines[i].strip())
            i += 1

            # 定位 NUMBER OF ATOMS
            while i < nlines and not lines[i].strip().startswith('ITEM: NUMBER OF ATOMS'):
                i += 1
            if i >= nlines:
                raise ValueError(f"在步 {step} 未找到 'ITEM: NUMBER OF ATOMS'")
            i += 1
            while i < nlines and lines[i].strip() == '':
                i += 1
            if i >= nlines:
                raise ValueError(f"在步 {step} 未找到原子数数值")
            natoms = int(lines[i].strip())
            i += 1

            # 定位 BOX BOUNDS
            while i < nlines and not lines[i].strip().startswith('ITEM: BOX BOUNDS'):
                i += 1
            if i >= nlines:
                raise ValueError(f"在步 {step} 未找到 'ITEM: BOX BOUNDS'")
            i += 1
            for _ in range(3):
                while i < nlines and lines[i].strip() == '':
                    i += 1
                if i >= nlines:
                    raise ValueError(f"在步 {step} 盒子坐标行不足")
                i += 1

            # 定位 ATOMS
            while i < nlines and not lines[i].strip().startswith('ITEM: ATOMS'):
                i += 1
            if i >= nlines:
                raise ValueError(f"在步 {step} 未找到 'ITEM: ATOMS'")
            atom_cols = lines[i].replace('ITEM: ATOMS', '').strip().split()
            i += 1

            atom_data = []
            for j in range(natoms):
                while i < nlines and lines[i].strip() == '':
                    i += 1
                if i >= nlines:
                    raise ValueError(f"在步 {step} 原子数据不足，预期 {natoms}，实际 {j}")
                parts = lines[i].strip().split()
                if len(parts) != len(atom_cols):
                    raise ValueError(
                        f"在步 {step} 原子数据列数不匹配：预期 {len(atom_cols)}，实际 {len(parts)}\n"
                        f"行内容: {lines[i].strip()}"
                    )
                atom_data.append([float(x) for x in parts])
                i += 1

            df = pd.DataFrame(atom_data, columns=atom_cols)
            df = df.sort_values(by='id').reset_index(drop=True)
            steps.append((step, df))
        else:
            i += 1
    return steps

# ==================== 对比函数 ====================

def compare_logs(df_ref, df_test, label):
    """对比两个 log DataFrame,返回各列的差异统计。"""
    merged = pd.merge(df_ref, df_test, on='Step', suffixes=('_ref', '_test'))
    if len(merged) == 0:
        raise ValueError("没有共同的步数可比较")
    stats = {}
    for col in df_ref.columns:
        if col == 'Step':
            continue
        ref_col = col + '_ref'
        test_col = col + '_test'
        diff = merged[test_col] - merged[ref_col]
        abs_diff = diff.abs()
        stats[col] = {
            'max_abs_diff': abs_diff.max(),
            'mean_abs_diff': abs_diff.mean(),
            'rmse': np.sqrt((diff**2).mean())
        }
    return stats

def compare_dumps(dump_ref, dump_test, label):
    """对比两个 dump 文件，返回每个力的统计摘要。"""
    if len(dump_ref) != len(dump_test):
        raise ValueError(f"时间步数不一致: ref {len(dump_ref)}, test {len(dump_test)}")
    steps_stats = []
    for (step_ref, df_ref), (step_test, df_test) in zip(dump_ref, dump_test):
        if step_ref != step_test:
            raise ValueError(f"步数不匹配: ref {step_ref}, test {step_test}")
        if len(df_ref) != len(df_test):
            raise ValueError(f"原子数不匹配 at step {step_ref}: ref {len(df_ref)}, test {len(df_test)}")
        fx_ref = df_ref['fx'].values
        fy_ref = df_ref['fy'].values
        fz_ref = df_ref['fz'].values
        fx_test = df_test['fx'].values
        fy_test = df_test['fy'].values
        fz_test = df_test['fz'].values
        diff_fx = fx_test - fx_ref
        diff_fy = fy_test - fy_ref
        diff_fz = fz_test - fz_ref
        step_stats = {
            'step': step_ref,
            'fx_max_abs': np.abs(diff_fx).max(),
            'fx_mean_abs': np.abs(diff_fx).mean(),
            'fx_rmse': np.sqrt((diff_fx**2).mean()),
            'fy_max_abs': np.abs(diff_fy).max(),
            'fy_mean_abs': np.abs(diff_fy).mean(),
            'fy_rmse': np.sqrt((diff_fy**2).mean()),
            'fz_max_abs': np.abs(diff_fz).max(),
            'fz_mean_abs': np.abs(diff_fz).mean(),
            'fz_rmse': np.sqrt((diff_fz**2).mean()),
        }
        steps_stats.append(step_stats)
    summary = {}
    keys = ['fx_max_abs', 'fx_mean_abs', 'fx_rmse',
            'fy_max_abs', 'fy_mean_abs', 'fy_rmse',
            'fz_max_abs', 'fz_mean_abs', 'fz_rmse']
    for key in keys:
        values = [s[key] for s in steps_stats]
        summary[key + '_avg'] = np.mean(values)
        summary[key + '_max'] = np.max(values)
        summary[key + '_min'] = np.min(values)
    return summary, steps_stats

def format_log_stats(log_stats):
    """将log统计结果按物理意义分组，生成格式化的报告行"""
    lines = []
    groups = {
        '温度': ['Temp'],
        '每原子势能': ['v_pe_peratom'],
        '每原子维里张量': ['v_W_xx', 'v_W_yy', 'v_W_zz', 'v_W_xy', 'v_W_xz', 'v_W_yz'],
        '其他热力学量': []
    }
    categorized = set(sum(groups.values(), []))
    groups['其他热力学量'] = [col for col in log_stats.keys() if col not in categorized]
    for group_name, cols in groups.items():
        if not cols:
            continue
        lines.append(f"    [{group_name}]")
        for col in cols:
            if col not in log_stats:
                continue
            stats = log_stats[col]
            lines.append(
                f"      {col:12s} : max_abs_diff = {stats['max_abs_diff']:12.6e}, "
                f"mean_abs_diff = {stats['mean_abs_diff']:12.6e}, rmse = {stats['rmse']:12.6e}"
            )
    return lines

# ==================== 阈值检查函数 ====================

def check_thresholds(log_stats, dump_summary):
    """根据预定义阈值判断测试是否通过。返回 (passed: bool, failures: list of str)"""
    failures = []
    for col, threshold in THRESHOLDS.items():
        if col in log_stats:
            max_abs = log_stats[col]['max_abs_diff']
            if max_abs >= threshold:
                failures.append(f"{col}: max_abs_diff = {max_abs:.6e} >= threshold {threshold:.1e}")
    for comp in ['fx', 'fy', 'fz']:
        key = comp + '_max_abs_max'
        if key in dump_summary and comp in THRESHOLDS:
            max_abs = dump_summary[key]
            if max_abs >= THRESHOLDS[comp]:
                failures.append(f"{comp} force: max_max_abs = {max_abs:.6e} >= threshold {THRESHOLDS[comp]:.1e}")
    return len(failures) == 0, failures

# ==================== 主程序 ====================

def main():
    base_dir = '.'
    report_lines = []
    report_lines.append("LAMMPS 测试结果对比报告（含阈值判断）")
    report_lines.append("=" * 60)

    # 统计变量
    dir_total = 0
    dir_pass = 0
    dir_fail = 0
    rerun_total = 0
    rerun_pass = 0
    rerun_fail = 0
    failed_details = []   # 存储 (类型, 路径, 原因)

    for system in SYSTEMS:
        for ensemble in ENSEMBLES:
            report_lines.append(f"\n>>> 体系: {system}  系综: {ensemble} <<<")

            ref_dir = os.path.join(base_dir, system, 'ref_results', ensemble)
            direct_dir = os.path.join(base_dir, system, ensemble)
            rerun_dir = os.path.join(base_dir, system, 'rerun', ensemble)

            ref_log = os.path.join(ref_dir, 'log.lammps')
            ref_dump = os.path.join(ref_dir, 'lammps.dump')

            if not os.path.exists(ref_log):
                report_lines.append(f"  [缺失] 基准 log 文件: {ref_log}")
                continue
            if not os.path.exists(ref_dump):
                report_lines.append(f"  [缺失] 基准 dump 文件: {ref_dump}")
                continue

            try:
                df_log_ref = parse_log(ref_log)
                dump_ref = parse_dump(ref_dump)
            except Exception as e:
                report_lines.append(f"  [错误] 解析基准文件失败: {e}")
                continue

            # ---------- 直接运行 ----------
            direct_log = os.path.join(direct_dir, 'log.lammps')
            direct_dump = os.path.join(direct_dir, 'lammps.dump')
            if os.path.exists(direct_log) and os.path.exists(direct_dump):
                dir_total += 1
                test_path = f"{system}/{ensemble}/direct"
                report_lines.append("  --- 直接运行 (direct) 与基准对比 ---")
                try:
                    df_log_direct = parse_log(direct_log)
                    dump_direct = parse_dump(direct_dump)

                    log_stats = compare_logs(df_log_ref, df_log_direct, "direct")
                    report_lines.append("    [log 差异]")
                    report_lines.extend(format_log_stats(log_stats))

                    dump_summary, _ = compare_dumps(dump_ref, dump_direct, "direct")
                    report_lines.append("    [dump 受力差异]")
                    for comp in ['fx', 'fy', 'fz']:
                        report_lines.append(
                            f"      {comp:2s} : avg_max_abs = {dump_summary[comp+'_max_abs_avg']:12.6e}, "
                            f"max_max_abs = {dump_summary[comp+'_max_abs_max']:12.6e}"
                        )

                    passed, fails = check_thresholds(log_stats, dump_summary)
                    if passed:
                        report_lines.append("    [结论] PASS")
                        dir_pass += 1
                    else:
                        report_lines.append("    [结论] FAIL")
                        for f in fails:
                            report_lines.append("      - " + f)
                        dir_fail += 1
                        failed_details.append(("direct", test_path, "; ".join(fails)))
                except Exception as e:
                    report_lines.append(f"    [错误] 直接运行对比失败: {e}")
                    dir_fail += 1
                    failed_details.append(("direct", test_path, f"异常: {e}"))
            else:
                report_lines.append("  [未进行] 直接运行文件缺失")

            # ---------- rerun ----------
            rerun_log = os.path.join(rerun_dir, 'log.lammps')
            rerun_dump = os.path.join(rerun_dir, 'lammps.dump')
            if os.path.exists(rerun_log) and os.path.exists(rerun_dump):
                rerun_total += 1
                test_path = f"{system}/{ensemble}/rerun"
                report_lines.append("  --- rerun 与基准对比 ---")
                try:
                    df_log_rerun = parse_log(rerun_log)
                    dump_rerun = parse_dump(rerun_dump)

                    log_stats = compare_logs(df_log_ref, df_log_rerun, "rerun")
                    report_lines.append("    [log 差异]")
                    report_lines.extend(format_log_stats(log_stats))

                    dump_summary, _ = compare_dumps(dump_ref, dump_rerun, "rerun")
                    report_lines.append("    [dump 受力差异]")
                    for comp in ['fx', 'fy', 'fz']:
                        report_lines.append(
                            f"      {comp:2s} : avg_max_abs = {dump_summary[comp+'_max_abs_avg']:12.6e}, "
                            f"max_max_abs = {dump_summary[comp+'_max_abs_max']:12.6e}"
                        )

                    passed, fails = check_thresholds(log_stats, dump_summary)
                    if passed:
                        report_lines.append("    [结论] PASS")
                        rerun_pass += 1
                    else:
                        report_lines.append("    [结论] FAIL")
                        for f in fails:
                            report_lines.append("      - " + f)
                        rerun_fail += 1
                        failed_details.append(("rerun", test_path, "; ".join(fails)))
                except Exception as e:
                    report_lines.append(f"    [错误] rerun 对比失败: {e}")
                    rerun_fail += 1
                    failed_details.append(("rerun", test_path, f"异常: {e}"))
            else:
                report_lines.append("  [未进行] rerun 文件缺失")

    # ==================== 最终总结 ====================
    report_lines.append("\n" + "=" * 60)
    report_lines.append("                    测 试 总 结")
    report_lines.append("=" * 60)
    report_lines.append(f"直接运行对比: 总测试数 {dir_total}, 通过 {dir_pass}, 失败 {dir_fail}")
    report_lines.append(f"Rerun 对比:    总测试数 {rerun_total}, 通过 {rerun_pass}, 失败 {rerun_fail}")

    total = dir_total + rerun_total
    total_pass = dir_pass + rerun_pass
    total_fail = dir_fail + rerun_fail
    report_lines.append(f"合计:           总测试数 {total}, 通过 {total_pass}, 失败 {total_fail}")

    if failed_details:
        report_lines.append("\n失败详情:")
        for typ, path, reason in failed_details:
            report_lines.append(f"  [{typ}] {path}  原因: {reason}")
    else:
        report_lines.append("\n所有测试均通过阈值检查。")

    report = '\n'.join(report_lines)
    print(report)
    with open('test_report.txt', 'w') as f:
        f.write(report)
    print("\n报告已保存至 test_report.txt")

if __name__ == "__main__":
    main()
