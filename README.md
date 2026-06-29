# MatPL NEP/KK — LAMMPS Kokkos Interface for NEP 

> **版本：MatPL-pro-2026.6** (2026-06-29)

MatPL NEP/KK 为 LAMMPS 提供 GPU 加速的 NEP (Neuroevolution Potential) 推理能力，通过 Kokkos 框架实现。本仓库包含 LAMMPS 接口代码和评估 License 文件。预编译的 `libnep_gpu.so` 动态库可从 GitHub Release 下载，或联系 matpl@pwmat.com 获取。

## 特性 (Features)

- 支持 **MatPL-2025.3+** 和 **GPUMD** 训练的 `nep.txt` 势函数文件
- **NEP4, NEP5, NEP4_ZBL, NEP5_ZBL, typewise-ZBL** 变体
- **混合精度推理** (FP32/FP16)
- 多模型偏差计算，用于主动学习工作流
- **多节点多 GPU 并行** (MPI + Kokkos)
- 逐原子能量输出，维里 (6 分量 + 9 分量质心维里用于热流)
- **MatPL 热流**：直接 GPU 计算 + 时间平均 fix 输出
- 运行时 License 验证

## 目录结构

```
.
├── nep_gpu/                              # 预编译 GPU 推理库 + License
│   ├── cuda-11.8/                        # 放置 libnep_gpu.so（从 GitHub Release 下载）
│   ├── cuda-12.8/                        # 放置 libnep_gpu.so（从 GitHub Release 下载）
│   └── summer_holiday.lic                # 评估 License（16 卡，2026-08-31 前有效）
├── KOKKOS/                               # 开源 Kokkos 包装层
│   ├── pair_nep_kokkos.cpp/h             # PairNEPKokkos 实现
│   ├── compute_matpl_heatflux_kokkos.cpp/h  # MatPL 热流 compute
│   └── fix_matpl_heatflux_ave_kokkos.cpp/h  # MatPL 热流 fix（时间平均）
├── MATPLDP/                              # DP 模型（可选）
├── MATPLD3/                              # D3 色散校正接口（可选，需 CUDA）
├── nep_gpu_loader.h                      # 开源 dlopen 加载器
├── pair_nep.cpp / pair_nep.h             # CPU fallback pair style
├── nep_cpu.cpp / nep_cpu.h               # CPU NEP 实现
├── kknep-patch.sh                        # GPU 版 LAMMPS 补丁脚本
├── kknep-patch-cpu.sh                    # CPU-only LAMMPS 补丁脚本
├── matpl-patch.sh                        # MatPL 补丁脚本
├── MatPLPackages.cmake                   # CMake 包定义
├── examples/                             # 示例输入文件
│   ├── H2O/                              # 水分子多节点多 GPU
│   ├── HfO2/                             # HfO2 多节点多 GPU
│   └── Heat_flux/                        # 热流计算
├── test/                                 # 测试套件
└── README.md                             # 本文件
```

## 预编译 libnep_gpu.so — 下载

预编译的 `libnep_gpu.so` 文件随 GitHub Release 发布。请根据 GPU 硬件选择对应版本下载，放入 `nep_gpu/` 对应子目录中。

> **下载链接**：[GitHub Releases](https://github.com/LonxunQuantum/lammps-MatPL/releases)
> 在最新 Release 的 Assets 中下载所需 .so 文件。如无法访问 GitHub，请联系 matpl@pwmat.com 获取。

下载后将 .so 文件放入对应目录：

| Release Asset | 放入位置 | CUDA | SM 目标 | 适用 GPU |
|-------------|----------|------|---------|----------|
| `cuda11.8/libnep_gpu.so` | `nep_gpu/cuda11.8/` | 11.8 | `35,37,60,61,70,75,80,86,89,90` | K40/K80, P100, V100, GTX 1080, TITAN V, T4, RTX 2080, A100, A6000, RTX 3090, RTX 4090, H100 |
| `cuda-12.8/libnep_gpu.so` | `nep_gpu/cuda-12.8/` | 12.8 | `60,61,70,75,80,86,89,90,100,101,120` | P100, V100, GTX 1080, TITAN V, T4, RTX 2080, A100, A6000, RTX 3090, RTX 4090, H100, B100/B200 (sm_100), RTX 5080/5090 (sm_120) |

> **注意**：编译时的 CUDA 版本必须与 .so 链接的 CUDA 运行时版本匹配（cuda-11.8 .so 需要 CUDA 11.x 运行时，cuda-12.8 .so 需要 CUDA 12.x 运行时）。

### GPU 速查表

| GPU 型号 | SM | 推荐目录 | Kokkos CMake 标志 |
|----------|-----|----------|-------------------|
| K40, K80 | 3.5 / 3.7 | cuda11.8 | `Kokkos_ARCH_KEPLER35=ON` / `KEPLER37=ON` |
| P100 | 6.0 | cuda-11.8 或 cuda-12.8 | `Kokkos_ARCH_PASCAL60=ON` |
| GTX 1080, GTX 1080 Ti | 6.1 | cuda-11.8 或 cuda-12.8 | `Kokkos_ARCH_PASCAL61=ON` |
| V100, TITAN V | 7.0 | cuda-11.8 或 cuda-12.8 | `Kokkos_ARCH_VOLTA70=ON` |
| T4, RTX 2080, RTX 2080 Ti | 7.5 | cuda-11.8 或 cuda-12.8 | `Kokkos_ARCH_TURING75=ON` |
| A100 | 8.0 | cuda-11.8 或 cuda-12.8 | `Kokkos_ARCH_AMPERE80=ON` |
| A6000, RTX 3090, RTX 3080 Ti | 8.6 | cuda-11.8 或 cuda-12.8 | `Kokkos_ARCH_AMPERE86=ON` |
| RTX 4090 | 8.9 | cuda-11.8 或 cuda-12.8 | `Kokkos_ARCH_ADA89=ON` |
| H100 | 9.0 | cuda-11.8 或 cuda-12.8 | `Kokkos_ARCH_HOPPER90=ON` |
| B100, B200 | 10.0 | cuda-12.8 | `Kokkos_ARCH_BLACKWELL100=ON` |
| RTX 5080, RTX 5090 | 12.0 | cuda-12.8 | `Kokkos_ARCH_BLACKWELL120=ON` |

## 评估 License

`nep_gpu/summer_holiday.lic` 为评估 License，参数如下：

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大并行 GPU 数 | 16 | 支持单节点或多节点共 16 卡 |
| 过期时间 | 2026-08-31 | UTC 时间 |
| 硬件绑定 | 无限制 | 不绑定特定 GPU UUID |
| 试用天数 | 无限制 | 完整功能，非试用 License |

> **License 过期后续期**：请联系 matpl@pwmat.com 申请新的 License 文件。

## 环境要求

| 组件 | 推荐版本 | 说明 |
|------|---------|------|
| LAMMPS | stable_29Aug2024, stable_2Aug2023, 或 stable_22Jul2025 | 通过 `kknep-patch.sh` 补丁 |
| CUDA Toolkit | 11.6+（匹配 .so 的 CUDA 版本） | 编译环境需与运行时的 libcudart 版本一致 |
| GCC | 8.3+ | 需 C++17 支持 |
| OpenMPI | 4.1.x | 多节点并行 |
| CMake | 3.18+ | 构建系统 |

---

# 用户手册

以下介绍如何编译带 NEP/KK 支持的 LAMMPS 并运行模拟。

## 第一步：为 LAMMPS 源码打补丁

```bash
bash kknep-patch.sh /path/to/lammps
```

此脚本会：
1. 备份原始 `CMakeLists.txt`
2. 添加 CUDA 语言支持到项目
3. 插入 `PKG_NEP_KK` 包定义（仅开源 Kokkos 包装层）
4. 复制接口文件到 LAMMPS：
   - `pair_nep.cpp`, `pair_nep.h` → `src/`
   - `nep_cpu.cpp`, `nep_cpu.h` → `src/`
   - `nep_gpu_loader.h` → `src/`
   - `KOKKOS/pair_nep_kokkos.cpp`, `pair_nep_kokkos.h` → `src/KOKKOS/`
   - `KOKKOS/compute_matpl_heatflux_kokkos.*` → `src/KOKKOS/`
   - `KOKKOS/fix_matpl_heatflux_ave_kokkos.*` → `src/KOKKOS/`
   - `MATPLDP/`, `MATPLD3/` (可选 DP/D3 接口)

支持的 LAMMPS 版本：**2023、2024 和 2025 发行版**。

## 第二步：加载编译环境

根据目标 GPU 和选择的 .so 版本，加载匹配的 CUDA 环境：

```bash
# === 使用 cuda-12.8 .so（支持 Ampere/Hopper/Blackwell）===
# 需要 CUDA 12.x 运行时，匹配 libcudart.so.12
module --ignore-cache load openmpi/4.1.6 cuda/12.8-share gcc/11.2.1 cmake/3.31.6

# === 使用 cuda-11.8 .so（支持 Ampere/Hopper）===
# 需要 CUDA 11.x 运行时，匹配 libcudart.so.11.0
module --ignore-cache load openmpi/4.1.6 cuda/11.8-share gcc/8.3.1 cmake/3.31.6
```

> **重要**：编译时的 CUDA 主版本必须与 .so 链接的 libcudart 主版本一致：
> - `cuda-12.8/` 和 `cuda-11.8/` .so 分别链接 `libcudart.so.12` 和 `libcudart.so.11.0`
> - CUDA 12.x 环境与 `cuda-11.8` .so 不兼容（反之亦然）

> **注意**：以上为龙讯 mcloud 编译环境加载方式供参考。请根据本地集群调整模块名。

## 第三步：下载并放置 libnep_gpu.so

1. 从 [GitHub Releases](https://github.com/LonxunQuantum/lammps-MatPL/releases) 下载对应 GPU 的 `libnep_gpu.so`
   （如无法访问 GitHub，请联系 matpl@pwmat.com）

2. 将下载的 `libnep_gpu.so` 放入 `nep_gpu/` 对应子目录（参见上方表格）

3. 设置环境变量指向 .so 文件：

```bash
# 查看你的 GPU 型号和计算能力
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 根据 GPU 速查表选择对应的 .so 路径：
export NEP_GPU_LIB_PATH=/path/to/nep_gpu/cuda-12.8/libnep_gpu.so   # 例如 RTX 4090
export NEP_GPU_LIB_PATH=/path/to/nep_gpu/cuda-11.8/libnep_gpu.so   # 例如 A100
```

## 第四步：编译 LAMMPS

```bash
cd /path/to/lammps/build

cmake -C ../cmake/presets/basic.cmake \
    -DPKG_MESONT=no \
    -DPKG_JPEG=no \
    -DPKG_KOKKOS=yes \
    -DPKG_NEP_KK=yes \
    -DKokkos_ENABLE_CUDA=yes \
    -DKokkos_ENABLE_OPENMP=yes \
    -DKokkos_ENABLE_CUDA_LAMBDA=yes \
    -DFFT_KOKKOS=CUFFT \
    -DKokkos_ARCH_ADA89=ON \
    -DTEST_TIME=ON \
    ../cmake

cmake --build . -- -j8
```

> **重要**：`Kokkos_ARCH_*` 标志需根据实际 GPU 调整。参见上方 GPU 速查表中的 "Kokkos CMake 标志" 列。RTX 4090 使用 `ADA89`（不是 `AMPERE89`）。

## 第五步：运行时设置

### 5.1 放置 libnep_gpu.so

```bash
# 方式 A：直接指向 .so 文件（推荐）
export NEP_GPU_LIB_PATH=/path/to/nep_gpu/cuda-12.8/libnep_gpu.so

# 方式 B：将目录加入 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/nep_gpu/cuda-12.8:$LD_LIBRARY_PATH
```

### 5.2 放置 License 文件

```bash
# 方式 A：指向 License 文件（推荐）
export NEP_LICENSE_PATH=/path/to/nep_gpu/summer_holiday.lic

# 方式 B：放在默认位置
mkdir -p ~/.nep_license
cp nep_gpu/summer_holiday.lic ~/.nep_license/license.json
```

### 5.3 准备输入文件

- LAMMPS 输入脚本（`.in`）
- 原子结构文件（如 `conf.lmp`，LAMMPS data 格式）
- NEP 势函数文件（`nep.txt`）— 由 MatPL 或 GPUMD 训练

## 第六步：运行模拟

```bash
# 单 GPU
mpirun -np 1 --bind-to numa lmp -k on g 1 -sf kk -pk kokkos -in input.in

# 单节点多 GPU（4 卡）
mpirun -np 4 --bind-to numa lmp -k on g 4 -sf kk -pk kokkos -in input.in

# 多节点（2 节点，每节点 4 卡）
mpirun -np 8 --bind-to numa --map-by ppr:4:node lmp -k on g 4 -sf kk -pk kokkos -in input.in
```

---

# LAMMPS 输入脚本参考

## Pair Style

```lammps
pair_style   matpl/nep/kk  nep.txt
pair_coeff   * * Hf O
```

`pair_coeff * *` 后的元素名对应 data 文件中的原子类型，映射到 NEP 势函数的元素顺序。

## 多模型偏差计算（主动学习）

```lammps
pair_style   matpl/nep/kk  nep1.txt nep2.txt nep3.txt out_freq 10 out_file explr.error
pair_coeff   * * Hf O
```

参数：
- 多个 `nep*.txt` 文件启用偏差计算
- `out_freq N` — 每 N 步输出偏差（默认 `1`）
- `out_file name` — 偏差输出文件名（默认 `explr.error`）

## Kokkos 必要设置

```lammps
package kokkos neigh half comm device
newton on
```

## 完整示例（NPT）

```lammps
package kokkos neigh half comm device
newton on

units           metal
boundary        p p p
atom_style      atomic
neighbor        1.0 bin
neigh_modify    delay 10

read_data       structure.lmp
replicate       10 10 10

mass   1    178.49
mass   2    15.999

pair_style   matpl/nep/kk  nep.txt
pair_coeff   * * Hf O

velocity        all create 300.0 12345
fix             1 all npt/kk temp 300 300 0.1 iso 0.0 0.0 0.5

thermo_style    custom step temp pe ke etotal press vol
thermo          10
timestep        0.002

run             1000
```

---

# MatPL 热流

Kokkos NEP pair style 提供直接的 MatPL 热流路径，无需传统 `ke/atom + pe/atom + centroid/stress/atom + heat/flux` 后处理。

```txt
pair_style   matpl/nep/kk  nep.txt
pair_coeff   * * C

compute      flux all matpl/heatflux/kk
fix          fluxout all matpl/heatflux/ave/kk 10 100 1000 flux file compute_HeatFlux.out
```

六个向量分量顺序：`Jx, Jy, Jz, Jconv,x, Jconv,y, Jconv,z`。Virial 贡献可恢复为 `(1-4, 2-5, 3-6)`。

完整示例和验证脚本参见 [examples/Heat_flux/](examples/Heat_flux/)。

---

# CPU-Only 回退

对于无 GPU 系统，可使用 CPU pair style（无需 License）：

```bash
bash kknep-patch-cpu.sh /path/to/lammps
```

```lammps
pair_style   matpl/nep  nep.txt
pair_coeff   * * Hf O
```

---

# 故障排除

| 错误信息 | 原因 | 解决方法 |
|---------|------|----------|
| `Failed to load NEP GPU library 'libnep_gpu.so'` | 找不到库 | 设置 `NEP_GPU_LIB_PATH` 指向 .so 文件 |
| `Symbol not found: nep_*` | 库版本不匹配 | 使用匹配的 `libnep_gpu.so` 版本 |
| `NEP GPU license check failed` | License 缺失或无效 | 设置 `NEP_LICENSE_PATH=/path/to/summer_holiday.lic` |
| `License file not found` | License 路径错误 | 验证文件路径和权限 |
| `License signature invalid` | 文件损坏或被篡改 | 重新获取 License 文件 |
| `License has expired` | 已过过期日期 (UTC) | 当前版本 2026-08-31 到期，联系 matpl@pwmat.com 续期 |
| `GPU does not match license binding` | 使用的 GPU 未注册在 License 中 | 用 `nvidia-smi --query-gpu=uuid` 核对 GPU UUID；或使用无硬件绑定的评估 License |
| `GPU count exceeds license limit` | GPU 数超过许可 | 设置 `CUDA_VISIBLE_DEVICES` 限制可见 GPU（当前最多 16 卡） |
| `Trial period has expired` | 试用天数用尽 | 联系 matpl@pwmat.com 获取正式 License |
| `License permanently disabled (tamper detected)` | 检测到 3 次以上时钟回拨或文件篡改 | 需联系 matpl@pwmat.com 获取新 License 并重置 |
| `CMake 3.11.0 is too old` | 模块缓存过期 | `module --ignore-cache load cmake/3.31.6` |
| ZBL 力不正确 | Ghost atom ZBL 力在 `neigh full` 下未反向通信 | 使用 `package kokkos neigh half comm device` + `newton on` |
| nvcc 使用系统 GCC 4.8.5 | CUDA 主机编译器未设置 | cmake 中设置 `-DCMAKE_CUDA_HOST_COMPILER=$(which g++)` |

---

# 环境变量参考

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `NEP_GPU_LIB_PATH` | `libnep_gpu.so` 完整路径 | `libnep_gpu.so`（依赖 `LD_LIBRARY_PATH`） |
| `NEP_LICENSE_PATH` | License 文件完整路径 | `~/.nep_license/license.json` |
| `NEP_FORCE_PLAN_LEGACY` | 强制使用 legacy 路径 | `0` |
| `NEP_FP_STORE_F16_ENABLE` | 启用 FP16 存储 | `0` |
| `NEP_SUM_FXYZ_STORE_F16_ENABLE` | 启用 sum_fxyz FP16 存储 | `0` |
| `NEP_Q_STORE_ANN_GEMM_FP16_ENABLE` | 启用 FP16 描述子到 ANN 传输 | `0` |
| `NEP_3B_TILE` | 3-body tile 大小 (1,4,5,8) | `5` |

---

# 示例

- [examples/H2O/](examples/H2O/) — 水分子多节点多 GPU 模拟
- [examples/HfO2/](examples/HfO2/) — HfO2 多节点多 GPU 模拟
- [examples/Heat_flux/](examples/Heat_flux/) — 石墨烯热流计算（含对比验证脚本）

---

# 测试

```bash
cd test/
export NEP_GPU_LIB_PATH=/path/to/your/libnep_gpu.so
export NEP_LICENSE_PATH=/path/to/nep_gpu/summer_holiday.lic

python3 run_all_tests.py
```

测试覆盖：HfO2、water、C、Cu、W、WMoTaV、GaN_typewise_zbl、U (typewise_zbl_cutoff)，涵盖 NPT/NVT 系综及 direct/rerun 模式。辅助工具：`python3 clean_test_outputs.py` 清理旧输出，`python3 compare_results.py` 仅生成对比报告。

详见 [test/Readme.md](test/Readme.md)。

---

# 许可说明

| 组件 | 许可 |
|------|------|
| `nep_gpu_loader.h`, `pair_nep.cpp/h`, `nep_cpu.cpp/h`, `KOKKOS/*`, `kknep-patch.sh`, `kknep-patch-cpu.sh`, `examples/*`, `test/*` | GPL v2+ |
| `nep_gpu/*/libnep_gpu.so` | 仅限评估使用 |
| `nep_gpu/summer_holiday.lic` | 评估 License，2026-08-31 前有效 |

评估 License 仅供非商业用途试用。商业使用或 License 续期请联系：**matpl@pwmat.com**。

---

# 引用

- 使用本 NEP-Kokkos LAMMPS 接口请引用：
  - Pengfei Suo, Xingxing Wu, Hongzhen Tian, et al. Towards Scalable and Efficient Machine-Learning Force Fields: The MatPL package and Its Advancements on Neuroevolution Potentials. ChemRxiv. 06 May 2026. DOI: https://doi.org/10.26434/chemrxiv.15001665/v3

- 使用 `NEP` 类请引用：
  - Ke Xu, et al. [GPUMD 4.0: A high-performance molecular dynamics package for versatile materials simulations with machine-learned potentials](https://doi.org/10.1002/mgea.70028), MGE Advances **3**, e70028 (2025).

- 使用 LAMMPS 的 `NEP` 类接口，也建议适当引用 LAMMPS。
