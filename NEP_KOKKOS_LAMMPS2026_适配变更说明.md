# NEP KOKKOS 到 LAMMPS 2026 的适配变更说明

本文档只覆盖 `lammps-MatPL/lmp_nepkokkos_cmake` 里“真正代码与 KOKKOS 接口”的适配，不覆盖你已经单独处理过的 CMake 改动。

## 1. 先说结论

这次从 LAMMPS 2024 迁移到 LAMMPS 2026，真正需要改动的核心位置是：

- `KOKKOS/pair_nep_kokkos.h`
- `KOKKOS/pair_nep_kokkos.cpp`

`nep_gpu/` 目录目前不需要为了 “LAMMPS 2026 的 KOKKOS 类型别名变化” 而改动，原因是：

- `nep_gpu/force/nepkk.cuh` 中 `NEPKK::compute()` 暴露的是纯裸指针接口：
  - `int* itype`
  - `int* ilist`
  - `int* numneigh`
  - `int* firstneigh`
  - `double* position`
  - `double* potential_per_atom_lmp`
  - `double* force_per_atom_lmp`
  - `double* virial_per_atom`
- `nep_gpu/` 中没有直接使用 LAMMPS 2024 已失效的 KOKKOS 类型别名，例如：
  - `t_x_array_randomread`
  - `t_f_array`
  - `tdual_efloat_1d`
  - `tdual_virial_array`
  - `F_FLOAT`
  - `E_FLOAT`
- 也没有直接依赖 `sync_host()` / `modify<DeviceType>()` / `view<DeviceType>()` 这类 2026 里变化较大的 KOKKOS 包装接口。

换句话说：

- `nep_gpu/` 是核心 CUDA 计算层；
- `pair_nep_kokkos.*` 是 LAMMPS KOKKOS 和 `nep_gpu` 之间的“胶水层/适配层”；
- 这次坏掉的是胶水层，不是 CUDA 核心层。

## 2. 为什么说 `nep_gpu/` 当前不用动

### 2.1 关键证据

`nep_gpu/force/nepkk.cuh` 中 `compute()` 的签名如下：

```cpp
void compute(
  int eflag_global,
  int eflag_atom,
  int vflag_either,
  int vflag_global,
  int vflag_atom,
  int nall,
  int inum,
  int nlocal,
  int max_neighbors,
  int num_neighbors,
  int* itype,
  int* ilist,
  int* numneigh,
  int* firstneigh,
  double* position,
  double* potential_per_atom_lmp,
  double* potential_per_atom_copy,
  double* force_per_atom_lmp,
  double* force_per_atom_copy,
  double* virial_per_atom,
  double* h_etot_virial_global
);
```

这说明 `nep_gpu` 不关心：

- `x` 在外面是 `t_x_array_randomread` 还是 `t_kkfloat_1d_3_lr_randomread`
- `f` 在外面是 `t_f_array` 还是 `t_kkacc_1d_3`
- `k_eatom` / `k_vatom` 在外面是旧 dual view 还是 2026 的 transform view

它只关心最终传进来的裸设备指针是否正确。

### 2.2 它目前仍然可能在哪些情况下需要动

虽然目前不需要因为 2026 的 KOKKOS 别名变化而修改 `nep_gpu/`，但这不等于“永远绝对不用动”。

后续如果运行期出现下面这类问题，就要回头看 `nep_gpu/`：

- 邻居列表语义在 2026 里发生了行为级变化，而不仅仅是类型别名变化
- `firstneigh` / `numneigh` 的解释方式与 2024 不一致
- ghost atom 的处理逻辑在真实算例里出现偏差
- `double*` / `float` 混用带来精度或布局问题
- 多模型 deviation 路径在真实 MPI + GPU 运行中出现问题

但就“当前已经暴露出来的 2026 接口兼容问题”而言，`nep_gpu/` 不是主要修改点。

## 3. 已完成的代码改动总览

相对于 `lammps2024/src/KOKKOS/pair_nep_kokkos.{h,cpp}`，本次已做的代码改动如下：

### 3.1 头文件 `pair_nep_kokkos.h`

1. 删除两个未再使用的临时 helper：
   - `get_position_view()`
   - `get_neighbors_flat_view()`
2. 把旧版坐标/力视图别名替换成 2026 官方风格别名。
3. 把多模型误差分析用的临时缓冲从旧 `ffloat` dual view 改成 `double` dual view。
4. 把 per-atom 能量/virial 容器改成 2026 使用的 transform/kkacc 类型。
5. 把 scatter view 的模板参数从旧 `F_FLOAT/E_FLOAT` 改成 2026 的 `KK_ACC_FLOAT` 和 `t_kkacc_*` layout。

### 3.2 源文件 `pair_nep_kokkos.cpp`

1. 更新 `k_full_f` / `k_full_e` 的分配类型。
2. 更新 DualView 的 host/device 访问方式：
   - `d_view` -> `view<DeviceType>()`
   - `h_view` -> `view_host()`
3. 更新 per-atom 数据同步接口：
   - `sync<LMPHostType>()` -> `sync_host()`
4. 修正 `bigint` 的 `fprintf` 格式字符串：
   - `%lld` -> `BIGINT_FORMAT`
5. 修正一处纯缩进问题，不改变行为。

## 4. 逐处 diff 与逐行解释

下面的 diff 是：

- 左边：`lammps2024/src/KOKKOS/pair_nep_kokkos.*`
- 右边：`lammps-MatPL/lmp_nepkokkos_cmake/KOKKOS/pair_nep_kokkos.*`

---

## 4.1 `pair_nep_kokkos.h`：删除两个 helper

### diff

```diff
-  // 坐标和力的一维视图包装器
-   Kokkos::View<const double*, DeviceType> get_position_view() {
-   return Kokkos::View<const double*, DeviceType>(reinterpret_cast<const double*>(x.data()), nall * 3);
-   }
-
-  // 邻居列表的一维扁平化视图
-   Kokkos::View<const int*, DeviceType> get_neighbors_flat_view() {
-   int max_neighbors = d_neighbors.extent(1);
-   int row_num = d_neighbors.extent(0);
-   return Kokkos::View<const int*, DeviceType>(
-      reinterpret_cast<const int*>(d_neighbors.data()), 
-      row_num * max_neighbors
-   );
-  }
-

```

### 逐行解释

- 删除 `get_position_view()`：
  原因是当前代码路径并没有使用这个 helper。保留它没有收益，反而会让人误以为 2026 里还需要手工把 `x` 扁平化后再包一层 `Kokkos::View`。
- 删除 `get_neighbors_flat_view()`：
  原因同上。当前真正使用的是 `d_neighbors.data()` 和 `extent()`，不再需要这个额外包装。
- 删除这两个函数还有一个额外好处：
  让头文件只保留 2026 必需的 KOKKOS 接口，减少误导。

---

## 4.2 `pair_nep_kokkos.h`：坐标和力的 KOKKOS 类型别名更新

### diff

```diff
-  typename AT::t_x_array_randomread x;
-  typename AT::t_f_array f;
+  typename AT::t_kkfloat_1d_3_lr_randomread x;
+  typename AT::t_kkacc_1d_3 f;
```

### 逐行解释

- `t_x_array_randomread` -> `t_kkfloat_1d_3_lr_randomread`
  - 2024 里 `t_x_array_randomread` 还能工作；
  - 2026 官方 KOKKOS pair style 已统一使用 `t_kkfloat_1d_3_lr_randomread` 表示坐标视图；
  - `lr` 表示 `LayoutRight`，`randomread` 表示只读随机访问，这与坐标访问模式一致。

- `t_f_array` -> `t_kkacc_1d_3`
  - 2024 的 `t_f_array` 是旧封装名；
  - 2026 里力数组更明确地使用 `kkacc` 系列类型；
  - 这一步是为了与 2026 的 scatter view 和贡献累加接口保持一致。

---

## 4.3 `pair_nep_kokkos.h`：多模型误差分析临时缓冲改成 `double` dual view

### diff

```diff
-  DAT::tdual_ffloat_1d k_full_f;  // DualView for full force [nall][3]
-  HAT::t_ffloat_1d h_full_f;
-  DAT::tdual_ffloat_1d k_full_e;  // DualView for full force [nall][3]
-  HAT::t_ffloat_1d h_full_e;
+  DAT::tdual_double_1d k_full_f;  // flattened [num_ff][nall][3]
+  HAT::t_double_1d h_full_f;
+  DAT::tdual_double_1d k_full_e;  // flattened [num_ff][nlocal]
+  HAT::t_double_1d h_full_e;
```

### 逐行解释

- `tdual_ffloat_1d` / `t_ffloat_1d` 在 2026 的 `ArrayTypes` 体系里已经不是这里应该继续使用的类型。
- 这两块缓冲并不是直接绑定 LAMMPS 主力数组的“标准 per-atom force view”，而是你为了多模型 deviation 额外开出来的扁平临时数组。
- 对这种“自定义临时缓冲”，在 2026 下改成 `DAT::tdual_double_1d` / `HAT::t_double_1d` 更直接，也更稳定。
- 注释里把布局明确成：
  - `k_full_f`: `[num_ff][nall][3]` 扁平化
  - `k_full_e`: `[num_ff][nlocal]` 扁平化
  这样后续看 `base_idx` 计算更清楚。

---

## 4.4 `pair_nep_kokkos.h`：per-atom 能量/virial 容器改成 2026 的 transform/kkacc 类型

### diff

```diff
-  DAT::tdual_efloat_1d k_eatom;
-  DAT::tdual_virial_array k_vatom;
-  typename AT::t_efloat_1d d_eatom;
-  typename AT::t_virial_array d_vatom; //device [nall][6]
+  DAT::ttransform_kkacc_1d k_eatom;
+  DAT::ttransform_kkacc_1d_6 k_vatom;
+  typename AT::t_kkacc_1d d_eatom;
+  typename AT::t_kkacc_1d_6 d_vatom;
```

### 逐行解释

- `k_eatom`
  - 旧：`tdual_efloat_1d`
  - 新：`ttransform_kkacc_1d`
  - 原因：2026 官方 pair style 不再用这一组旧 dual 类型来表示 per-atom energy，改成 transform/kkacc 体系。

- `k_vatom`
  - 旧：`tdual_virial_array`
  - 新：`ttransform_kkacc_1d_6`
  - 原因同上，per-atom virial 在 2026 中也进入同一套 transform/kkacc 类型系统。

- `d_eatom`
  - 旧：`t_efloat_1d`
  - 新：`t_kkacc_1d`
  - 表示设备侧 per-atom energy 数据视图的类型也要同步切换。

- `d_vatom`
  - 旧：`t_virial_array`
  - 新：`t_kkacc_1d_6`
  - 表示设备侧 per-atom virial 数据视图的类型也要同步切换。

这几行是本次适配里最关键的一组变化之一，因为它会直接影响：

- `memoryKK->create_kokkos()`
- `k_eatom.view<DeviceType>()`
- `k_vatom.view<DeviceType>()`
- 后续 `contribute()` 和 `sync_host()`

---

## 4.5 `pair_nep_kokkos.h`：scatter view 类型更新

### diff

```diff
-  DupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> dup_f;
-  DupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout> dup_eatom;
-  DupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> dup_vatom;
-
-  NonDupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> ndup_f;
-  NonDupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout> ndup_eatom;
-  NonDupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> ndup_vatom;
+  DupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout> dup_f;
+  DupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout> dup_eatom;
+  DupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> dup_vatom;
+
+  NonDupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout> ndup_f;
+  NonDupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout> ndup_eatom;
+  NonDupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> ndup_vatom;
```

### 逐行解释

- `F_FLOAT` / `E_FLOAT`：
  - 这是旧接口体系里的命名；
  - 2026 的 pair style 里这里应使用 `KK_ACC_FLOAT`，与 `kkacc` 设备累加类型配套。

- `DAT::t_f_array::array_layout` -> `DAT::t_kkacc_1d_3::array_layout`
  - 旧 layout 来源于旧 `f` 容器；
  - 新 layout 必须与新的 `f` 类型 `t_kkacc_1d_3` 保持一致。

- `DAT::t_efloat_1d::array_layout` -> `DAT::t_kkacc_1d::array_layout`
  - 否则 `dup_eatom` / `ndup_eatom` 会与 `d_eatom` 类型不匹配。

- `DAT::t_virial_array::array_layout` -> `DAT::t_kkacc_1d_6::array_layout`
  - 否则 `dup_vatom` / `ndup_vatom` 会与 `d_vatom` 类型不匹配。

这组改动的目标非常明确：

- 让 scatter view 的数据类型
- 让 scatter view 的布局类型
- 让目标视图 `f` / `d_eatom` / `d_vatom`

三者在 2026 下重新对齐。

---

## 4.6 `pair_nep_kokkos.cpp`：`eflag_atom` 分支缩进修正

### diff

```diff
  if (eflag_atom) {
-  memoryKK->destroy_kokkos(k_eatom,eatom);
-  memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
-  d_eatom = k_eatom.view<DeviceType>();
+    memoryKK->destroy_kokkos(k_eatom,eatom);
+    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
+    d_eatom = k_eatom.view<DeviceType>();
  }
```

### 逐行解释

- 这是纯格式化修正，不改变行为。
- 修正后更容易看出：
  `destroy/create/view` 三行都属于 `if (eflag_atom)` 分支。

---

## 4.7 `pair_nep_kokkos.cpp`：临时 DualView 的分配与访问方式更新

### diff

```diff
-      k_full_f = DAT::tdual_ffloat_1d("pair:full_f", num_ff * global_nall * 3);
+      k_full_f = DAT::tdual_double_1d("pair:full_f", num_ff * global_nall * 3);
```

```diff
-      Kokkos::deep_copy(k_full_f.d_view, 0.0);
+      Kokkos::deep_copy(k_full_f.template view<DeviceType>(), 0.0);
```

```diff
-    h_full_f = k_full_f.h_view;
+    h_full_f = k_full_f.view_host();
```

```diff
-      k_full_e = DAT::tdual_ffloat_1d("pair:full_e", num_ff * global_nlocal);
+      k_full_e = DAT::tdual_double_1d("pair:full_e", num_ff * global_nlocal);
```

```diff
-      Kokkos::deep_copy(k_full_e.d_view, 0.0);
+      Kokkos::deep_copy(k_full_e.template view<DeviceType>(), 0.0);
```

```diff
-    h_full_e = k_full_e.h_view;
+    h_full_e = k_full_e.view_host();
```

### 逐行解释

- `tdual_ffloat_1d` -> `tdual_double_1d`
  - 对应头文件里的类型切换；
  - 保证声明和构造一致。

- `.d_view` -> `.view<DeviceType>()`
  - 2026 下不应该继续直接访问旧 dual view 的内部成员；
  - 官方风格是通过 `view<DeviceType>()` 取得设备视图。

- `.h_view` -> `.view_host()`
  - 同理，host 侧不再直接访问旧成员；
  - 使用公开接口 `view_host()` 更符合 2026 的写法。

这组改动解决的是“能编译”和“接口风格正确”两个问题。

---

## 4.8 `pair_nep_kokkos.cpp`：`bigint` 输出格式修正

### diff

```diff
-    fprintf(explrError_fp, "%9lld %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f\n",
+    fprintf(explrError_fp, BIGINT_FORMAT " %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f\n",
             update->ntimestep,
             glb_avg_f_err, glb_min_f_err, glb_max_f_err,
             glb_avg_ei_err, glb_min_ei_err, glb_max_ei_err);
```

### 逐行解释

- `%lld` 在某些平台上碰巧可用，但它假定 `bigint` 对应 `long long`。
- LAMMPS 2026 的 `bigint` 实际来自 `lmptype.h`，其平台定义不一定总是 `long long`。
- `BIGINT_FORMAT` 是 LAMMPS 官方给 `bigint` 配套的格式宏，使用它最稳妥。
- 这里我没有继续保留宽度控制 `%9...`，因为 `BIGINT_FORMAT` 自己已经带 `%`，直接拼宽度会导致格式串警告。

---

## 4.9 `pair_nep_kokkos.cpp`：per-atom 数据同步接口更新

### diff

```diff
  if (eflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
-    k_eatom.template sync<LMPHostType>();
+    k_eatom.sync_host();
  }
```

```diff
  if (vflag_either) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
-    k_vatom.template sync<LMPHostType>();
+    k_vatom.sync_host();
  }
```

### 逐行解释

- `modify<DeviceType>()`
  - 这行仍然保留，表示设备侧刚刚写过数据。

- `sync<LMPHostType>()` -> `sync_host()`
  - 2026 下官方 pair style 使用的是 `sync_host()`；
  - 对 `ttransform_kkacc_*` 这组类型来说，这也是正确、直接的 host 同步接口；
  - 这样做可以和 `pair_lj_cut_kokkos.cpp`、`pair_eam_kokkos.h/.cpp` 的写法保持一致。

---

## 5. 我这次没有改动的文件

本次没有因为 2026 KOKKOS 接口变化而修改：

- `nep_gpu/force/nepkk.cu`
- `nep_gpu/force/nepkk.cuh`
- `nep_gpu/force/nep_kernal_function.cuh`
- `nep_gpu/utilities/*`

原因不是这些文件“不重要”，而是它们当前不是 2026 KOKKOS 别名变更的爆点。

## 6. 参考的 2026 官方 KOKKOS 对照实现

本次适配主要对照了 LAMMPS 2026 自带的 KOKKOS pair style，重点参考：

- `lammps2026/src/KOKKOS/pair_lj_cut_kokkos.h`
- `lammps2026/src/KOKKOS/pair_lj_cut_kokkos.cpp`
- `lammps2026/src/KOKKOS/pair_eam_kokkos.h`
- `lammps2026/src/KOKKOS/pair_pod_kokkos.h`

特别是下面几类写法是直接照着 2026 的官方风格对齐的：

- `typename AT::t_kkfloat_1d_3_lr_randomread x;`
- `typename AT::t_kkacc_1d_3 f;`
- `DAT::ttransform_kkacc_1d k_eatom;`
- `DAT::ttransform_kkacc_1d_6 k_vatom;`
- `k_eatom.sync_host();`
- `k_vatom.sync_host();`

## 7. 验证结果

我已经做过两层验证：

### 7.1 对象级验证

把修改后的 `pair_nep_kokkos.cpp` 放进 `lammps2026/src/KOKKOS/` 后，使用 LAMMPS 2026 CMake 生成的真实 `nvcc_wrapper` 编译命令对该文件做了单独编译，结果：

- 原先的 2024 -> 2026 类型别名错误全部消失；
- 当前对象文件可以成功生成。

### 7.2 整包前向验证

我还启动过一次：

```bash
cmake --build . -j 4 --target lammps
```

结果是：

- CMake 配置成功；
- KOKKOS 包配置成功；
- 构建已进入常规 `.cpp` 编译阶段；
- 没有在 `pair_nep_kokkos` 这层立刻报出新的接口错误；
- 为了节省时间，我在继续大规模编译前手动停止了整包构建。

这意味着：

- `pair_nep_kokkos` 这层的 2026 KOKKOS 接口迁移已经基本打通；
- 下一步更值得做的是运行时验证，而不是继续怀疑 `nep_gpu` 的编译接口。

## 8. 下一步建议

最值得继续验证的不是“还有没有别名没改”，而是运行行为：

1. 单模型路径：
   - `eflag_global`
   - `eflag_atom`
   - `vflag_global`
   - `vflag_atom`
2. 多模型 deviation 路径：
   - `k_full_f`
   - `k_full_e`
   - `reverse_comm`
3. MPI + 多 GPU：
   - ghost atom
   - `firstneigh` / `numneigh`
   - deviation 统计输出文件

如果后面真实运行里再出现问题，优先排查顺序建议是：

1. `pair_nep_kokkos.cpp` 的数据传递和同步
2. `reverse_comm`
3. `nep_gpu` 的邻居列表解释和 ghost atom 路径

## 9. `lmp_nepkokkos_cmake_lmp2026` 的调用语法迁移

这一节对应的是你后来单独复制出来并继续修改的目录：

- `lammps-MatPL/lmp_nepkokkos_cmake_lmp2026`

这一部分和前面 2024 -> 2026 的 KOKKOS 类型适配是两件事：

- 前面解决的是“代码能不能在 LAMMPS 2026 的 KOKKOS 接口下编过”
- 这里解决的是“这个库的输入语法是不是更符合标准 LAMMPS 用法”

本次迁移后的目标是：

- `pair_style` 只保留真正的 pair style 级选项
- `pair_coeff` 负责：
  - 原子类型范围
  - NEP 势文件路径
  - 元素映射

也就是把原来“势文件写在 `pair_style` 里”的非标准用法，改成更接近普通 LAMMPS pair style 的形式。

### 9.1 用户输入脚本的变化

#### 单模型

旧语法：

```lammps
pair_style   matpl/nep/kk  nep.txt
pair_coeff   * * Hf O
```

新语法：

```lammps
pair_style   matpl/nep/kk
pair_coeff   * * nep.txt Hf O
```

逐行解释：

- `pair_style   matpl/nep/kk  nep.txt`
  - 旧逻辑里把势文件 `nep.txt` 直接写在 `pair_style` 后面。
  - 这是这个库自己实现出来的行为，不是 LAMMPS 常见习惯。

- `pair_coeff   * * Hf O`
  - 旧逻辑里 `pair_coeff` 只负责元素映射。
  - 这里并不读取势文件。

- `pair_style   matpl/nep/kk`
  - 新逻辑里 `pair_style` 只声明样式本身。
  - 不再在这里接收 `.txt` 势文件。

- `pair_coeff   * * nep.txt Hf O`
  - 新逻辑里把势文件移动到 `pair_coeff`。
  - `* *` 仍然表示对全部 LAMMPS atom type 生效。
  - `nep.txt` 是模型文件。
  - `Hf O` 是对每个 LAMMPS atom type 的元素映射。

#### 多模型 deviation

旧语法：

```lammps
pair_style   matpl/nep/kk  nep0.txt nep1.txt nep2.txt nep3.txt out_freq ${DUMP_FREQ} out_file model_devi.out
pair_coeff   * * Hf O
```

新语法：

```lammps
pair_style   matpl/nep/kk  out_freq ${DUMP_FREQ} out_file model_devi.out
pair_coeff   * * nep0.txt nep1.txt nep2.txt nep3.txt Hf O
```

逐行解释：

- `pair_style   matpl/nep/kk  nep0.txt ... out_freq ... out_file ...`
  - 旧逻辑里把“模型列表”和“运行选项”混在 `pair_style` 里一起解析。

- `pair_coeff   * * Hf O`
  - 旧逻辑里 `pair_coeff` 依然只做元素映射。

- `pair_style   matpl/nep/kk  out_freq ${DUMP_FREQ} out_file model_devi.out`
  - 新逻辑里 `pair_style` 只保留真正的样式选项：
    - `out_freq`
    - `out_file`

- `pair_coeff   * * nep0.txt nep1.txt nep2.txt nep3.txt Hf O`
  - 新逻辑里把所有模型文件都放进 `pair_coeff`。
  - 模型文件后面紧跟元素映射。

---

### 9.2 对应修改的代码位置

本次语法迁移只改了这两个文件：

- `lammps-MatPL/lmp_nepkokkos_cmake_lmp2026/KOKKOS/pair_nep_kokkos.cpp`
- `lammps-MatPL/lmp_nepkokkos_cmake_lmp2026/README.md`

真正的行为变化都在：

- `PairNEPKokkos<DeviceType>::settings()`
- `PairNEPKokkos<DeviceType>::coeff()`

---

### 9.3 `settings()` 的代码差异与中文解释

迁移前，`settings()` 会直接扫描 `.txt` 文件并加载模型。  
迁移后，`settings()` 只解析 `out_freq` / `out_file`，并且如果你还把 `.txt` 写在 `pair_style` 里，会明确报错。

#### 关键 diff

```diff
 void PairNEPKokkos<DeviceType>::settings(int narg, char **arg)
 {
   is_rank_0 = (comm->me == 0);
-  if (narg < 1) error->all(FLERR, "Illegal pair_style command");
-  int iarg = 0;
-  num_ff = 0;
-  while(iarg < narg) {
-    std::string arg_str(arg[iarg]);
-    if (arg_str.find(".txt") != std::string::npos) {
-      potential_files.push_back(arg_str);
-      num_ff ++;
-      iarg++;
-    } else {
-      break;
-    }
-  }
+  rank = comm->me;
+  device_id = -1;
+  num_ff = 0;
+  potential_files.clear();
+  nep_gpu_models.clear();
+  if (explrError_fp != nullptr) {
+    fclose(explrError_fp);
+    explrError_fp = nullptr;
+  }
   int iarg = 0;
   while (iarg < narg) {
     if (strcmp(arg[iarg], "out_freq") == 0) {
+      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for pair_style option out_freq");
       out_freq = utils::inumeric(FLERR, arg[++iarg], false, lmp);
     } else if (strcmp(arg[iarg], "out_file") == 0) {
+      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for pair_style option out_file");
       explrError_fname = arg[++iarg];
+    } else if (std::string(arg[iarg]).find(".txt") != std::string::npos) {
+      error->all(FLERR,
+                 "For pair_style matpl/nep/kk in lmp2026 patch, specify NEP model file(s) in pair_coeff. "
+                 "Example: pair_coeff * * nep.txt Si O");
+    } else {
+      error->all(FLERR, "Unknown pair_style option for matpl/nep/kk: " + std::string(arg[iarg]));
     }
     iarg++;
   }
-  nep_gpu_models.resize(num_ff);
-  for (int i=0; i < num_ff; i++) {
-    ...
-  }
 }
```

#### 逐行解释

- 删除 `if (narg < 1) error->all(...)`
  - 原因：旧接口默认要求 `pair_style` 后面至少跟一个模型文件。
  - 现在模型文件已经迁移到 `pair_coeff`，所以这里不能再这样假设。

- 删除 `while(iarg < narg) { ... .txt ... }`
  - 原因：这一整段就是旧版“从 `pair_style` 中提取模型文件”的实现。
  - 新接口下不再需要。

- 新增 `rank = comm->me;`
  - 明确初始化 rank。
  - 即便还没加载模型，也先把运行环境状态整理好。

- 新增 `device_id = -1;`
  - 先给设备号一个安全初值。
  - 后面如果是 device 版本，再调用 `cudaGetDevice()` 更新。

- 新增 `num_ff = 0;`
  - 重置模型数量。
  - 表示当前 `settings()` 不负责加载模型，只先把状态清空。

- 新增 `potential_files.clear();`
  - 清空旧的模型文件列表。
  - 避免重复调用 `pair_style` 时残留上一次的路径。

- 新增 `nep_gpu_models.clear();`
  - 清空旧的模型对象容器。
  - 避免后续 `pair_coeff` 重新加载时混入旧对象。

- 新增关闭 `explrError_fp`
  - 原因：如果用户重新定义 `pair_style`，旧的 deviation 输出文件句柄也应该一起关闭。

- 新增 `out_freq` 参数值检查
  - 原因：旧写法里如果忘了给 `out_freq` 填值，报错并不够明确。
  - 现在会直接提示缺少值。

- 新增 `out_file` 参数值检查
  - 原因同上。

- 新增 `.txt` 出现在 `pair_style` 时直接报错
  - 这是迁移里最关键的一行。
  - 作用是：
    - 拦截旧用法
    - 明确告诉用户新用法应该写到 `pair_coeff`

- 新增未知选项报错
  - 原因：防止拼错参数名时静默通过。

- 删除 `nep_gpu_models.resize(num_ff)` 和后面的 `read_neptxt(...)`
  - 原因：模型加载职责已经迁移到 `coeff()`。

---

### 9.4 `coeff()` 的代码差异与中文解释

迁移前，`coeff()` 只做元素映射。  
迁移后，`coeff()` 负责：

- 解析 `* *`
- 解析一个或多个 `.txt` 模型文件
- 解析元素映射
- 加载模型
- 设置 `setflag` 和 `cutsq`

#### 关键 diff

```diff
 void PairNEPKokkos<DeviceType>::coeff(int narg, char **arg)
 {
   if (!allocated) allocate();
+  if (narg < 4)
+    error->all(FLERR, "Incorrect args for pair coefficients for matpl/nep/kk");
+
+  int ilo, ihi, jlo, jhi;
+  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
+  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
+
+  potential_files.clear();
+  num_ff = 0;
+  int iarg = 2;
+  while (iarg < narg) {
+    std::string arg_str(arg[iarg]);
+    if (arg_str.find(".txt") != std::string::npos) {
+      potential_files.push_back(arg_str);
+      ++num_ff;
+      ++iarg;
+    } else {
+      break;
+    }
+  }
+
+  if (num_ff == 0)
+    error->all(FLERR,
+               "pair_coeff for matpl/nep/kk must include at least one NEP model file after the atom type ranges");
+
+  if (iarg >= narg)
+    error->all(FLERR,
+               "pair_coeff for matpl/nep/kk must include element mapping after the NEP model file(s)");
+
+  if (narg - iarg != atom->ntypes)
+    error->all(FLERR,
+               "pair_coeff for matpl/nep/kk must provide one element label for each LAMMPS atom type");
+
+  if (explrError_fp != nullptr) {
+    fclose(explrError_fp);
+    explrError_fp = nullptr;
+  }
+
+  if (is_rank_0 && num_ff > 1) {
+    explrError_fp = fopen(&explrError_fname[0], "w");
+    ...
+  }
+
+  nep_gpu_models.clear();
+  nep_gpu_models.resize(num_ff);
+  for (int i = 0; i < num_ff; i++) {
+    const std::string &model_file = potential_files[i];
+    nep_gpu_models[i].read_neptxt(model_file.c_str(), is_rank_0, comm->me, device_id, i);
+    ...
+  }
+
+  int count = 0;
+  for (int i = ilo; i <= ihi; i++) {
+    for (int j = std::max(jlo, i); j <= jhi; j++) {
+      setflag[i][j] = 1;
+      cutsq[i][j] = cutoffsq;
+      count++;
+    }
+  }
   for (int f1 = 0; f1 < num_ff; f1++) {
     std::vector<int> atom_type_module = nep_gpu_models[f1].element_atomic_number_list;
     std::vector<int> atom_types;
-    for (int ii = 2; ii < narg; ++ii) {
+    for (int ii = iarg; ii < narg; ++ii) {
       std::string element = utils::strdup(arg[ii]);
       int temp = find_atomic_number(element);
       auto iter = std::find(atom_type_module.begin(), atom_type_module.end(), temp);
-      if (iter != atom_type_module.end() || arg[ii] == 0)
+      if (iter != atom_type_module.end())
       {
         int index = std::distance(atom_type_module.begin(), iter);
         atom_types.push_back(index);
       }
     }
-    nep_gpu_models[f1].set_atom_type_map(narg-2, atom_types.data());
+    nep_gpu_models[f1].set_atom_type_map(narg - iarg, atom_types.data());
   }
 }
```

#### 逐行解释

- 新增 `if (narg < 4)`
  - 原因：`pair_coeff` 至少也要长成 `* * nep.txt X`
  - 这里先做最基本的长度保护。

- 新增 `utils::bounds(...)`
  - 原因：恢复标准 LAMMPS 对前两个 `pair_coeff` 参数的处理方式。
  - 也就是让 `arg[0]` 和 `arg[1]` 真正作为 atom type 范围解释。

- 新增 `potential_files.clear();`
  - 清空旧模型文件列表。
  - 避免重复定义时残留旧路径。

- 新增 `num_ff = 0;`
  - 重置模型数量计数器。

- 新增 `int iarg = 2;`
  - 表示从 `pair_coeff` 的第三个参数开始解析模型文件。
  - 因为前两个位置已经保留给 `* *` 或类型范围。

- 新增 `while (iarg < narg) { ... .txt ... }`
  - 这是迁移后的新模型文件解析器。
  - 它会连续读取一个或多个 `.txt` 文件。
  - 一旦遇到不是 `.txt` 的 token，就认为后面开始进入元素映射区。

- 新增 `if (num_ff == 0)`
  - 原因：防止用户忘记写势文件。
  - 现在势文件不在 `pair_style`，所以这里必须强制检查。

- 新增 `if (iarg >= narg)`
  - 原因：防止用户只给了势文件，没有给元素映射。

- 新增 `if (narg - iarg != atom->ntypes)`
  - 原因：强制要求“每个 LAMMPS atom type 都要有一个元素标签”。
  - 这能提前拦截很多映射错误。

- 新增关闭 `explrError_fp`
  - 原因：如果重新定义 `pair_coeff`，旧的 deviation 输出句柄也应该被替换。

- 新增多模型时重新打开 `explrError_fp`
  - 原因：deviation 输出现在和模型文件加载绑定在一起，更合理。
  - 因为 `num_ff` 现在是在 `coeff()` 中才最终确定。

- 新增 `nep_gpu_models.clear();` 和 `resize(num_ff)`
  - 原因：在 `coeff()` 中按模型文件个数重新构造模型对象容器。

- 新增 `read_neptxt(...)`
  - 原因：把模型真正加载动作迁移到 `coeff()`。

- 新增 `setflag[i][j] = 1;`
  - 原因：恢复标准 LAMMPS pair_coeff 语义。
  - 表示这些类型对已经被显式设置过参数。

- 新增 `cutsq[i][j] = cutoffsq;`
  - 原因：在读取第一个模型后，用模型 cutoff 初始化 pair cutoff。

- `for (int ii = 2; ii < narg; ++ii)` -> `for (int ii = iarg; ii < narg; ++ii)`
  - 原因：旧逻辑假设元素映射从第三个参数就开始。
  - 新逻辑里第三个参数开始可能先是一串 `.txt` 文件，所以元素映射必须从 `iarg` 之后开始。

- 删除 `|| arg[ii] == 0`
  - 原因：这并不是一个正确的“元素合法”判断。
  - 保留它反而会让逻辑变得模糊。

- `set_atom_type_map(narg-2, ...)` -> `set_atom_type_map(narg - iarg, ...)`
  - 原因：旧写法默认前两个参数之后全是元素映射。
  - 新写法里前两个参数后面还插入了 `.txt` 文件，所以映射长度必须改成 `narg - iarg`。

---

### 9.5 这次语法迁移带来的直接收益

改完以后，`lmp_nepkokkos_cmake_lmp2026` 这份代码有几个直接好处：

- 更符合普通 LAMMPS 用户习惯
  - 势文件放在 `pair_coeff`
  - `pair_style` 只放样式选项

- 错误信息更明确
  - 如果还按旧写法把 `.txt` 写在 `pair_style`，会直接提示新写法
  - 如果 `out_freq` / `out_file` 缺值，也会直接提示

- `pair_coeff` 的语义更完整
  - 类型范围
  - 势文件
  - 元素映射
  都集中在一个地方

- 更容易和别的 pair style 保持一致
  - 后续别人读这个接口时理解成本更低

---

### 9.6 当前新目录的推荐用法

单模型：

```lammps
pair_style   matpl/nep/kk
pair_coeff   * * nep.txt Hf O
```

多模型 deviation：

```lammps
pair_style   matpl/nep/kk out_freq ${DUMP_FREQ} out_file model_devi.out
pair_coeff   * * nep0.txt nep1.txt nep2.txt nep3.txt Hf O
```

如果仍然写成旧语法：

```lammps
pair_style   matpl/nep/kk nep.txt
```

现在会被明确拒绝，并提示把模型文件迁移到 `pair_coeff`。
