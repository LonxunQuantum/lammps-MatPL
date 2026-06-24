# LAMMPS-MatPL 测试工具说明

本目录用于批量运行和对比 LAMMPS 测试算例，当前支持以下体系与系综：

- 体系：`HfO2`、`water`、`C`、`Cu`、`W`、`WMoTaV`
- 系综：`NPT`、`NVT`

支持两类测试结果：

- **直接运行结果**：`<system>/<ensemble>/`
- **rerun 结果**：`<system>/rerun/<ensemble>/`

---

## 目录中的主要文件

- `test_config.py`  
  公共配置文件，定义：
  - `SYSTEMS`
  - `ENSEMBLES`
  - `LAMMPS_CMD`

- `run_all_tests.py`  
  批量运行全部测试算例，并可在结束后自动执行结果对比。

- `compare_results.py`  
  对比待测结果与基准结果，生成 `test_report.txt`。

- `clean_test_outputs.py`  
  清理历史输出文件，并将已有的 `test_report.txt` 按时间戳备份。

---

## 目录结构

目录结构示例如下：

```text
test/
├── Readme.md
├── test_config.py
├── run_all_tests.py
├── compare_results.py
├── clean_test_outputs.py
├── test_report.txt
├── backups/
├── HfO2/
│   ├── NPT/
│   │   ├── lmp.in
│   │   ├── log.lammps
│   │   ├── lammps.dump
│   │   └── screen.log
│   ├── NVT/
│   ├── ref_results/
│   │   ├── NPT/
│   │   │   ├── log.lammps
│   │   │   └── lammps.dump
│   │   └── NVT/
│   │       ├── log.lammps
│   │       └── lammps.dump
│   └── rerun/
│       ├── NPT/
│       └── NVT/
└── water/
    ├── NPT/
    ├── NVT/
    ├── ref_results/
    └── rerun/
```

说明：

- 基准结果放在：`<system>/ref_results/<ensemble>/`
- 直接运行结果放在：`<system>/<ensemble>/`
- rerun 结果放在：`<system>/rerun/<ensemble>/`
- log 文件名固定为：`log.lammps`
- dump 文件名固定为：`lammps.dump`

---

## 环境要求

- Linux
- Python 3.8+
- Python 依赖：
  - `numpy`
  - `pandas`

安装示例：

```bash
pip install numpy pandas
```

此外需要系统中可直接调用 LAMMPS 命令。

---

## LAMMPS 运行命令

当前测试使用如下命令运行 LAMMPS：

```bash
mpirun -np 1 --bind-to numa lmp_mpi -k on g 1 -sf kk -pk kokkos gpu/aware off -in lmp.in
```

该命令已配置在 `test_config.py` 的 `LAMMPS_CMD` 中，`run_all_tests.py` 会直接复用。

---

## 使用方法

### 1. 清理旧结果

运行前可先清理历史输出，并备份已有测试报告：

```bash
python3 clean_test_outputs.py
```

仅预览将执行的操作：

```bash
python3 clean_test_outputs.py --dry-run
```

功能说明：

- 若存在 `test_report.txt`，会备份到：
  - `backups/test_report_时间戳.txt`
- 会清理各测试目录下的历史输出，例如：
  - `log.lammps`
  - `lammps.dump`
  - `screen.log`
  - 以及部分常见日志文件

---

### 2. 批量运行全部测试

```bash
python3 run_all_tests.py
```

仅预览运行命令：

```bash
python3 run_all_tests.py --dry-run
```

运行结束后默认会自动执行结果对比，并生成 `test_report.txt`。

可选参数：

- `--dry-run`  
  仅打印将执行的命令，不实际运行

- `--no-compare`  
  运行完算例后不执行 `compare_results.py`

- `--stop-on-error`  
  某个算例失败后立即停止

示例：

```bash
python3 run_all_tests.py --stop-on-error
```

---

### 3. 单独执行结果对比

如果已经有输出文件，也可以直接只做对比：

```bash
python3 compare_results.py
```

执行后会：

- 在终端打印摘要
- 生成 `test_report.txt`

---

## 对比内容

`compare_results.py` 会将待测结果与基准结果进行对比，包含两部分：

### 1. `log.lammps` 热力学量

按 `Step` 对齐后，比较：

- `Temp`
- `pe_peratom`
- `W_xx`
- `W_yy`
- `W_zz`
- `W_xy`
- `W_xz`
- `W_yz`

输出统计量包括：

- `max_abs_diff`
- `mean_abs_diff`
- `rmse`

---

### 2. `dump` 文件中的原子受力

比较每个时间步中原子的：

- `fx`
- `fy`
- `fz`

输出统计量包括：

- 每步统计
- 汇总统计

常见摘要形式为：

- `avg`
- `max`
- `min`

---

## 输出文件说明

### `screen.log`

`run_all_tests.py` 运行每个算例时，会将标准输出和错误输出写入对应目录下的 `screen.log`。

例如：

- `water/NPT/screen.log`
- `HfO2/rerun/NVT/screen.log`

---

### `test_report.txt`

`compare_results.py` 生成的总测试报告，位于 `test/` 目录下。

若再次清理，旧报告会被备份到 `backups/` 目录。

---

## 修改测试范围

测试体系和系综统一由 `test_config.py` 管理。

例如：

```python
SYSTEMS = ["HfO2", "water", "C", "Cu", "W", "WMoTaV"]
ENSEMBLES = ["NPT", "NVT"]
```

如果后续需要新增体系或系综，只需修改该文件，其余脚本会自动复用。

---

## 常见问题

### 1. 为什么建议把公共参数放进 `test_config.py`？

因为以下脚本都需要共享配置：

- `run_all_tests.py`
- `clean_test_outputs.py`
- `compare_results.py`

统一管理可以避免重复维护和配置不一致。

---

### 2. 如果某个目录下没有 `lmp.in` 会怎样？

`run_all_tests.py` 会跳过该算例，并在终端打印提示，不会导致整个脚本直接崩溃。

---

### 3. 如果基准结果缺失会怎样？

`compare_results.py` 会在报告中标明缺失或未进行，具体以脚本当前实现为准。

---

### 4. dump 文件必须包含哪些列？

至少需要包含：

- `id`
- `fx`
- `fy`
- `fz`

否则无法进行受力对比。

---

## 推荐执行流程

推荐按以下顺序操作：

```bash
python3 clean_test_outputs.py
python3 run_all_tests.py
```

如果只想重新生成报告：

```bash
python3 compare_results.py
```

---

## 维护建议

后续如果继续扩展测试工具，建议统一放入 `test_config.py` 的内容包括：

- 体系列表
- 系综列表
- LAMMPS 运行命令
- 清理文件模式
- 报告文件名
- 备份目录名

这样更便于维护。
