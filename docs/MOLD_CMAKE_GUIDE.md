# MatPL NULL + Mold 安装与 CMake 编译操作手册 V2

更新：2026-11-01（Asia/Shanghai）。适用于 Linux / Mcloud，推荐 LAMMPS `stable_22Jul2025_update4`。所有编译、模型推理和模拟均通过 Slurm 计算节点执行；Git、文件安装和日志检查可在登录节点执行。

主流程生成支持 **MatPL NEP CPU、NEP Kokkos/GPU 和 Mold `square/well`** 的 LAMMPS。DP 使用第 8 步的独立构建。当前安装脚本会为 CMake 启用 CUDA 语言，即使选择 DP 或 CPU 推理构建，也要有可用的 CUDA 编译工具链。

## 0. GitHub 提交审计结论

已只读核对真实 GitHub 分支与本地 HEAD：

| 项目 | 核对结果 |
|---|---|
| GitHub | `git@github.com:LonxunQuantum/lammps-matpl.git` |
| 主分支 | `main`，`21fabf34eadc57a717f4c7fbce9bb80aad5fd991` |
| NULL 功能提交 | `fc33c65`，13 个生产代码/安装文件及 6 个说明/测试文件 |
| Mold 包装提交 | `9d29679b71bc9edcc62adb90bee6427dc52896cc` |
| 远程 pair_NULL 与本地 | 一致，核心实现文件与已测版本一致 |
| Mold 当前发布形态 | 根目录只有 `pair_NULL-mold-optional.patch`，尚未直接提交展开的扩展目录 |

核心提交和推送目标正确。Mold 的提交方式是“发布补丁包”，因此 clone 后必须先执行第 2 步展开补丁；仅运行 `kknep-patch.sh` 不会安装 Mold。这与“先展开补丁，再将新增目录提交 GitHub”的方案不同，但补丁内容完整且已实际验证可以展开、安装。

补丁 SHA256：

```text
3a85118652a431cdb1b64924e2f7e480f4387dc9bd68990948b182f0367a726f
```

补丁内保留了官方源文件原始 CRLF、行尾空格和字节。不要使用 `--whitespace=fix` 或编辑器格式化该补丁。新版文档更新同时为根目录原始补丁增加 `.gitattributes` 的 `-text -whitespace` 规则，保留其原始字节并避免将嵌套补丁误报为源码空白问题。

## 1. 获取接口仓库与新的 LAMMPS 源码

在同一个登录节点 Bash 终端按顺序执行。以下使用新的安装目录，不覆盖现有通过百万步验证的源码和程序：

```bash
task_root=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/pair_NULL
install_root="$task_root/install-matpl-mold-v2"
mkdir -p "$install_root"
cd "$install_root"

git clone --single-branch --branch pair_NULL \
  git@github.com:LonxunQuantum/lammps-matpl.git matpl-interface

git clone --depth 1 --branch stable_22Jul2025_update4 \
  git@github.com:lammps/lammps.git lammps

MATPL_INTERFACE="$install_root/matpl-interface"
LAMMPS_SRC="$install_root/lammps"
git --version
```

已有同名目录时先检查其内容，不要重复 clone 或覆盖。也可使用新解压的同版本官方 LAMMPS 源码，但需将其放在 `$install_root/lammps`，以与后面的作业脚本路径一致。

上述 SSH 获取方式沿用服务器已配置的 GitHub 密钥；网络允许时也可使用对应 HTTPS URL。本次检查中 raw.githubusercontent.com 的 HTTPS 请求被重置，GitHub SSH 查询正常。

## 2. 展开 GitHub 当前提供的 Mold 补丁包

在 MatPL 接口仓库执行，而不是在 LAMMPS 源码目录执行：

```bash
cd "$MATPL_INTERFACE"
git branch
git log --oneline -3
sha256sum pair_NULL-mold-optional.patch

if [ -f extensions/mold/upstream/manifest.json ]; then
  echo 'Mold 扩展目录已存在，继续核验文件。'
else
  git apply --check pair_NULL-mold-optional.patch
  git apply --binary --whitespace=nowarn pair_NULL-mold-optional.patch
fi

ls extensions/mold/upstream/
ls patches/
```

现在应有 `extensions/mold/upstream/` 中的 4 个 cpp/h 文件、LICENSE、manifest，以及 `extensions/mold/mold-flat-virial.patch` 和 `patches/lammps-22Jul2025-kokkos-bin-range.patch`。

核验官方文件字节：

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib, json
p = Path('extensions/mold/upstream')
m = json.loads((p / 'manifest.json').read_text())
assert m['commit'] == '6fc7218ad5be70e95bce905ac3819cf513ba4c50'
for name, expected in m['files'].items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected, name
print('PASS: official Mold sources match their manifest')
PY
```

本步骤用于本地安装，未自动提交/推送展开文件。若未来 GitHub 已直接包含扩展目录，条件分支会跳过重复展开，哈希检查仍会执行。

## 3. 安装 MatPL 接口到 LAMMPS

```bash
cd "$MATPL_INTERFACE"
bash kknep-patch.sh "$LAMMPS_SRC"

test -f "$LAMMPS_SRC/src/atom_type_mapping.h"
test -f "$LAMMPS_SRC/src/pair_nep.cpp"
test -f "$LAMMPS_SRC/src/KOKKOS/pair_nep_kokkos.cpp"
test -f "$LAMMPS_SRC/src/nep_gpu/force/nepkk.cu"
test -f "$LAMMPS_SRC/src/MATPLDP/pair_dp.cpp"
```

脚本复制 CPU/GPU/DP 接口文件，并修改 LAMMPS CMake 配置。新增 `atom_type_mapping.h` 必须随接口安装；不需要删除或压缩 LAMMPS 类型映射中的 NULL 位置。

## 4. 安装 Mold 并应用两项局部修复

以下命令用于第 1 步新准备、尚未安装 Mold 的源码树：

```bash
cd "$MATPL_INTERFACE"
cp extensions/mold/upstream/pair_square_well.{cpp,h} "$LAMMPS_SRC/src/"
cp extensions/mold/upstream/pair_lj_BG.{cpp,h} "$LAMMPS_SRC/src/"

patch --binary --dry-run -d "$LAMMPS_SRC" -p1 \
  < extensions/mold/mold-flat-virial.patch
patch --binary -d "$LAMMPS_SRC" -p1 \
  < extensions/mold/mold-flat-virial.patch

patch --dry-run -d "$LAMMPS_SRC" -p1 \
  < patches/lammps-22Jul2025-kokkos-bin-range.patch
patch -d "$LAMMPS_SRC" -p1 \
  < patches/lammps-22Jul2025-kokkos-bin-range.patch
```

第一项修复官方 `square/well` 平坦井区域的 `fpair` 未初始化问题，只补零力对应的 virial，不修改势函数公式；`--binary` 保留官方 CRLF。第二项为当前已测 LAMMPS 版本的 Kokkos 分箱范围保护。已经打过补丁的源码不要重复应用；若 dry-run 报已应用或不匹配，先检查版本和当前文件。

Mold 通过将源文件加入 LAMMPS `src/` 安装。CMake 重新配置后自动发现 cpp 和 PairStyle；无需先执行 make，也没有 `PKG_MOLD` 开关。

## 5. 用 Slurm + CMake 编译 NEP CPU/GPU + Mold

在安装根目录创建构建脚本：

```bash
cd "$install_root"
cat > build-nep-mold.sbatch <<'EOF'
#!/bin/bash
#SBATCH -p demo
#SBATCH -J matpl-mold-build
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o build-nep-mold-%j.out
#SBATCH -e build-nep-mold-%j.err
set -euo pipefail
module load gcc/9.3.0 cuda/11.8-share openmpi/4.1.6
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
install_root="$SLURM_SUBMIT_DIR"
LAMMPS_SRC="$install_root/lammps"
LAMMPS_BUILD="$install_root/build-nep-mold"
cmake_bin=/share/app/cmake/3.31.6/bin/cmake
test -f "$LAMMPS_SRC/src/atom_type_mapping.h"
test -f "$LAMMPS_SRC/src/pair_square_well.cpp"
"$cmake_bin" -S "$LAMMPS_SRC/cmake" -B "$LAMMPS_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_COMPILER="$LAMMPS_SRC/lib/kokkos/bin/nvcc_wrapper" \
  -DCMAKE_CUDA_COMPILER=/share/app/cuda/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_MPI=ON -DBUILD_OMP=ON \
  -DPKG_KOKKOS=ON -DPKG_NEP_KK=ON -DPREC_NEPINFER=OFF \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_AMPERE86=ON \
  -DFFT_KOKKOS=CUFFT -DPKG_KSPACE=ON -DPKG_MANYBODY=ON \
  -DPKG_MATPLDP=OFF -DPKG_MATPLD3=OFF
"$cmake_bin" --build "$LAMMPS_BUILD" --parallel "$SLURM_CPUS_PER_TASK"
"$LAMMPS_BUILD/lmp" -k on g 1 -h > "$LAMMPS_BUILD/styles.txt"
grep -F 'matpl/nep/kk' "$LAMMPS_BUILD/styles.txt"
grep -F 'square/well' "$LAMMPS_BUILD/styles.txt"
sha256sum "$LAMMPS_BUILD/lmp" > "$LAMMPS_BUILD/lmp.sha256"
bash "$install_root/matpl-interface/tests/null_mapping/run_unit.sh"
echo "PASS: CMake build, required styles and mapping unit tests. Binary: $LAMMPS_BUILD/lmp"
EOF
sbatch build-nep-mold.sbatch
```

必须从 `$install_root` 提交，因为脚本用 `$SLURM_SUBMIT_DIR` 定位 `lammps/` 和 `matpl-interface/`。GCC 9.3、CUDA 11.8、OpenMPI 4.1.6 和 Ampere86 配置与已通过 Mold 案例的配置一致，适用于 RTX 3080 Ti。更换机器需调整模块路径与 GPU 架构。

该构建启用 NEP_KK，默认 FP32 推理；`PREC_NEPINFER=ON` 为另一种 FP64 构建，不能把现有 FP32 验证结论直接当成其精度资格。该构建没有开启 DP 或 D3。

## 6. 检查构建结果

将下面的 `作业号` 替换为 sbatch 返回的数字：

```bash
sacct -j 作业号 --format=JobID,State,ExitCode,Elapsed
tail -30 "$install_root/build-nep-mold-作业号.out"
cat "$install_root/build-nep-mold/lmp.sha256"
grep -F 'matpl/nep/kk' "$install_root/build-nep-mold/styles.txt"
grep -F 'square/well' "$install_root/build-nep-mold/styles.txt"
```

要求 Slurm 为 `COMPLETED 0:0`，日志出现构建/样式/映射测试 PASS。新二进制路径为 `$install_root/build-nep-mold/lmp`。独立重编译的二进制哈希可能因构建路径或工具链变化而不同；以本次构建记录和实际验证为准。

`-h` 只能证明样式已注册，还需继续做实际模型/案例运行。

## 7. 可选：独立构建 MatPL DP

MatPL NULL 的 DP 功能也已在分支中；第 5 步的 NEP 构建没有开启它。以下使用本服务器已验证的 Torch 环境和独立 `build-dp/`，不替换 NEP + Mold 程序：

```bash
cd "$install_root"
cat > build-dp.sbatch <<'EOF'
#!/bin/bash
#SBATCH -p demo
#SBATCH -J matpl-dp-build
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH -o build-dp-%j.out
#SBATCH -e build-dp-%j.err
set -euo pipefail
module load gcc/9.3.0 cuda/11.6-share openmpi/4.1.6
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
dp_env=/share/app/anaconda3/envs/matpl-2026.3
dp_nvidia_libs=$(printf '%s:' "$dp_env"/lib/python3.11/site-packages/nvidia/*/lib)
export LD_LIBRARY_PATH="$dp_nvidia_libs$dp_env/lib:${LD_LIBRARY_PATH:-}"
install_root="$SLURM_SUBMIT_DIR"
cmake_bin=/share/app/cmake/3.31.6/bin/cmake
"$cmake_bin" -S "$install_root/lammps/cmake" -B "$install_root/build-dp" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_MPI=ON -DPKG_MANYBODY=ON \
  -DPKG_KOKKOS=OFF -DPKG_NEP_KK=OFF -DPKG_MATPLDP=ON \
  -DMKL_INCLUDE_DIR="$dp_env/include" \
  -DTorch_DIR="$dp_env/lib/python3.11/site-packages/torch/share/cmake/Torch"
"$cmake_bin" --build "$install_root/build-dp" --parallel "$SLURM_CPUS_PER_TASK"
"$install_root/build-dp/lmp" -h > "$install_root/build-dp/styles.txt"
grep -F 'matpl/dp' "$install_root/build-dp/styles.txt"
sha256sum "$install_root/build-dp/lmp" > "$install_root/build-dp/lmp.sha256"
EOF
sbatch build-dp.sbatch
```

这条 DP 配置使用 CUDA 11.6 编译工具、Torch `2.2.0+cu118` 的既有环境及 MKL include 路径；不同于 NEP Kokkos 的 CUDA 11.8 配置，不要混用构建目录。`matpl/dp` 通过 Torch 使用 CPU 或 CUDA，运行时不需要 Kokkos `-k on` 参数。它也不提供 centroid stress，逐原子 virial 需要模型保留可微计算图。

DP 输入示例：

```lammps
pair_style hybrid/overlay matpl/dp model.pt lj/cut 6.0
pair_coeff * * matpl/dp Hf O NULL
pair_coeff * * lj/cut 0.003 1.0
```

实际 DP 运行同样在 Slurm 中加载对应环境。这个模型示例需替换为自己的 Torch 模型；接口包含 DP 不代表 NEP 模型文件可以直接用于 DP。

## 8. 常见问题与验证范围

- 找不到 `extensions/mold/`：当前 GitHub 以根目录补丁包交付，先完成第 2 步。
- 找不到 `atom_type_mapping.h`：从新版接口仓库重新执行第 3 步，核对安装脚本来源。
- `Unrecognized pair style square/well`：确认四个 Mold 文件已复制到实际构建用的 `src/`，并重新运行 CMake 配置。
- patch 提示重复/反向：检查是否已安装，不要重复对已修复源码打补丁。
- Kokkos 单线程性能 WARNING：是性能建议，需结合退出码和完整运行日志判断结果。
- `compute ... pair square/well` 的 `c_1` 在当前 metal、未归一化设置下是总井势能；原案例 `v_well=c_1*count(all)` 会再乘 2100，不能把它当成同一能量列。
- 当前 NULL 不支持 charge-enabled/QNEP 模型。原阶段四仍有 DP GPU 约 2% 性能退化、三个性能单元 INCONCLUSIVE，以及既有精度/严格重复性失败；成功的 Mold 案例没有消除这些资格限制。
