# NULL 原子类型映射

`matpl/nep`、`matpl/nep/kk`、`matpl/dp` 支持完整的 LAMMPS atom type 映射。模型路径仍放在 `pair_style` 后；`pair_coeff * *` 后必须按 LAMMPS 类型顺序提供恰好 `atom->ntypes` 个映射。

```lammps
pair_style hybrid/overlay matpl/nep model.txt lj/cut 6.0
pair_coeff * * matpl/nep Si NULL
pair_coeff * * lj/cut 0.003 1.0
```

类型 2 不作为 NEP 中心或邻居；它的 NEP 力、能量、逐原子能量和 virial 均为零，其他子势仍可计算该类型。Kokkos 子势名为 `matpl/nep/kk`；DP 为 `matpl/dp`，模型文件使用相应的 Torch 模型。

`NULL` 大小写不敏感，可位于首部、中间、末尾或连续出现。非 NULL 元素必须存在于每个模型中；全 NULL、映射数量不符和未知元素会报错。重复 `pair_coeff` 会重建映射，模型各自的元素顺序可以不同。

`hybrid/overlay` 累加各子势的力、能量和 virial。`hybrid` 按分配的邻居表计算；NULL 过滤不替代 LAMMPS 的类型对分配。对 DP，如果某个有效类型的全部类型对被分配给其他子势，将缺少该类型的中心邻居表并明确报错；若该类型不参与 DP，应映射为 NULL。

多个模型按各自元素表独立建索引，只有第一个模型驱动轨迹，其余用于偏差。偏差仅统计有效拥有原子，分母为全局有效原子数，不包含其他子势写入的力。CPU/GPU NEP 在全局没有有效原子且请求多模型偏差时给出受控错误；DP 返回零贡献和零偏差。这与“全部类型映射为 NULL”的无效配置不同。

DP 保留已配置但当前没有原子的元素，以维持模型归一化；模型按元素分组的输出会按原始原子身份回填。DP 逐原子 virial 需要模型保留可微的 Etot/ImageDR 图，不满足时请求原子应力会明确报错。

CPU NEP 使用 double，沿用 `CENTROID_SAME`；GPU NEP 可同时请求普通六分量和原生九分量 centroid 应力，后者顺序为 `xx yy zz xy xz yz yx zx zy`。DP 使用 Float64，不提供 centroid stress。MPI 路径要求 Newton pair on；GPU NEP 沿用 Kokkos half neighbor 配置。

QNEP/charge-enabled 模型与 NULL 的组合暂不支持，会在初始化时明确报错。无 NULL 的 QNEP 路径维持原有支持范围。

## 安装与验证

在本仓库根目录执行原有安装入口：

```sh
bash kknep-patch.sh /absolute/path/to/lammps
```

安装脚本已包含新增的 `atom_type_mapping.h`。CPU、GPU、DP 仍沿用各自原有的 CMake 配置；GPU 启用 KOKKOS/CUDA、NEP_KK 和正确的设备架构，DP 使用 C++17、MATPLDP 和匹配的 Torch CMake 包。

在计算节点运行小型回归：

```sh
bash tests/null_mapping/run_unit.sh
```

该入口覆盖映射构造和 GPU 类型精简；完整物理/并行验证及限制见 [NULL_MAPPING_VALIDATION.md](NULL_MAPPING_VALIDATION.md)。

Mold `square/well` 是独立的第三方势函数，按其官方安装说明加入 LAMMPS 源码后再配置 CMake。NULL 核心接口不依赖 Mold；复现 MatPL + Mold 时使用 `hybrid/overlay matpl/nep/kk nep.txt square/well ...` 和 `pair_coeff * * matpl/nep/kk Si NULL`。已验证的 Mold 固定版本需要平坦井 virial 修复；该可选扩展及 LAMMPS 分箱保护补丁单独交付。
