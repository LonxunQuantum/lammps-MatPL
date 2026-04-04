# NEP KOKKOS 在 LAMMPS 2026 中的标准包改造方案

本文档回答的问题是：

- 为什么当前 `lmp_nepkokkos_cmake_lmp2026` 不是一个标准的 LAMMPS package
- 如果要把它改造成更标准的 LAMMPS 2026 package，推荐怎么拆目录和接 CMake
- 哪些路径可行，哪些路径虽然能工作但不推荐

这里说的“标准 package”，更准确地说，是“按 LAMMPS 2026 的 package/CMake 机制接入”，而不是继续依赖每次构建前手工 patch 根 `cmake/CMakeLists.txt`。

补充说明：

- 本文档最初是按“CPU + KOKKOS 一起并入同一个 package”来讨论的；
- 但当前仓库里已经实际落地的 `lammps-MatPL/MATPL-NEP/` 骨架，已经根据后续需求调整成“只负责 KOKKOS/GPU 扩展”的版本；
- 也就是说，当前实际实现中，`MATPL-NEP` 不再携带 `pair_nep.* / nep_cpu.*`，而是要求外部先提供基础的 CPU NEP 库。

## 1. 先说结论

推荐采用下面这条路线：

- 新建一个正式 package，名字建议为 `MATPL-NEP`
- 把 CPU 版本 `pair_nep.*` 和 `nep_cpu.*` 放到 `src/MATPL-NEP/`
- 把 KOKKOS 版本 `pair_nep_kokkos.*` 放到 `src/MATPL-NEP/kokkos/`
- 把 CUDA 核心 `nep_gpu/` 放到 `src/MATPL-NEP/nep_gpu/`
- 新增 `cmake/Modules/Packages/MATPL-NEP.cmake`
- 在 `cmake/CMakeLists.txt` 里做两处一次性的正式注册：
  - 把 `MATPL-NEP` 加入 `STANDARD_PACKAGES`
  - 在第二个 `include(Packages/...)` 循环里，把 `MATPL-NEP` 放到 `KOKKOS` 后面

这样做以后：

- 不再需要 `kknep-patch_2026.sh` 这种“每次编译前 patch 根 CMake”的脚本
- `pair_nep_kokkos.*` 的启用条件可以被明确限制为：
  - `PKG_MATPL-NEP=yes`
  - `PKG_KOKKOS=yes`
  - `Kokkos_ENABLE_CUDA=yes`
- 包结构会从“散落在 `src/`、`src/KOKKOS/`、`src/nep_gpu/` 的 overlay”变成一个可维护的标准 package

## 2. 当前库为什么不是标准 package

当前 `lmp_nepkokkos_cmake_lmp2026` 的工作方式，本质上是：

- 先把文件复制到 `lammps2026/src/`
- 再把 KOKKOS 文件复制到 `lammps2026/src/KOKKOS/`
- 再把 `nep_gpu/` 复制到 `lammps2026/src/nep_gpu/`
- 最后再 patch `lammps2026/cmake/CMakeLists.txt`

这和标准 package 的差别主要有 5 点：

1. 它没有正式注册到 `STANDARD_PACKAGES` 或 `SUFFIX_PACKAGES`
2. 它没有自己的 `cmake/Modules/Packages/<PKG>.cmake`
3. 它不是一个自洽的 `src/<PKG>/` 目录，而是拆散复制到多个地方
4. 它依赖额外脚本给根 `CMakeLists.txt` 注入 `target_sources(...)`
5. 它的 `.cu` 源文件不在标准 package 的收集路径里，所以必须手动补构建规则

一句话总结：

- 标准 package 是“LAMMPS 认识这个包，因此自动按 package 机制收集源文件”
- 当前库是“LAMMPS 不认识这个包，因此只能靠脚本把文件硬塞进去”

## 3. 为什么不建议继续沿用当前 patch 方式

继续保留现在的 `kknep-patch_2026.sh` 路线，短期能用，但长期有几个维护问题：

### 3.1 根 CMake 的 patch 点很脆

当前脚本依赖于：

- 找到 `project(lammps ... LANGUAGES ...)`
- 找到特定的 `foreach(PKG_WITH_INCL CORESHELL ...)`
- 再在固定位置后插入 `if(PKG_NEP_KK)` 代码块

一旦上游 `cmake/CMakeLists.txt` 调整了段落顺序、注释、缩进或循环结构，脚本就可能失效。

### 3.2 代码分布不自然

现在逻辑上属于同一个功能的代码分散在：

- `src/pair_nep.*`
- `src/nep_cpu.*`
- `src/nep_gpu/...`
- `src/KOKKOS/pair_nep_kokkos.*`

这会让后续维护者很难一眼看清“NEP package 的完整边界”。

### 3.3 很难做精确的条件编译

当前脚本是额外加一个 `if(PKG_NEP_KK)` 块，直接把：

- `nep_gpu/*.cu`
- `KOKKOS/pair_nep_kokkos.cpp`

塞给 `lammps` 目标。

这种方式虽然能编，但没有很好地融入 LAMMPS 现有的 package 生命周期。

### 3.4 不利于后续再扩展 DP / D3 / 更多 MatPL 组件

如果以后还要把 DP、D3 或更多 MatPL 相关接口继续接进来，继续走 overlay + patch 的模式，根 CMake 只会越来越重，越来越难维护。

## 4. 推荐的目标结构

推荐的目标目录结构如下：

```txt
lammps2026/
├── cmake/
│   └── Modules/
│       └── Packages/
│           └── MATPL-NEP.cmake
└── src/
    └── MATPL-NEP/
        ├── pair_nep.cpp
        ├── pair_nep.h
        ├── nep_cpu.cpp
        ├── nep_cpu.h
        ├── kokkos/
        │   ├── pair_nep_kokkos.cpp
        │   └── pair_nep_kokkos.h
        └── nep_gpu/
            ├── force/
            │   ├── nepkk.cu
            │   ├── nepkk.cuh
            │   └── nep_kernal_function.cuh
            └── utilities/
                ├── error.cu
                ├── error.cuh
                ├── gpu_vector.cu
                ├── gpu_vector.cuh
                ├── common.cuh
                └── nep_utilities.cuh
```

这个结构有 3 个关键优点：

1. `MATPL-NEP` 的代码边界清晰
2. CPU / KOKKOS / CUDA 核心都归到同一个 package 下
3. `pair_nep_kokkos.*` 可以不再污染 `src/KOKKOS/`

## 5. 为什么把 `pair_nep_kokkos.*` 放到 `src/MATPL-NEP/kokkos/`，而不是 `src/KOKKOS/`

这一步很关键，也是本方案和“继续 patch”的最大区别。

### 5.1 如果继续放在 `src/KOKKOS/`

表面上看，放在 `src/KOKKOS/` 很像 LAMMPS 原生做法，但这里有个实际问题：

- `KOKKOS.cmake` 会通过 `RegisterStylesExt(...)` 自动扫描 `_kokkos` 样式
- 一旦基础样式 `pair_nep.h` 被注册了，而 `src/KOKKOS/pair_nep_kokkos.h` 又存在
- 只要 `PKG_KOKKOS=yes`，它就会尝试把 `pair_nep_kokkos.cpp` 编进来

这会带来一个副作用：

- 即使 `Kokkos_ENABLE_CUDA=no`
- 也可能提前触发 `pair_nep_kokkos.cpp` 的编译

而当前这版 `pair_nep_kokkos.cpp` 明确依赖：

- `cuda_runtime.h`
- `nep_gpu/*.cu`
- GPU 设备路径

也就是说，它并不是一个“只开了 KOKKOS 就总能编”的 `_kokkos` 变体。

### 5.2 放到 `src/MATPL-NEP/kokkos/` 的好处

把 `pair_nep_kokkos.*` 放到 package 自己的子目录里，标准包模块就可以自己决定：

- 什么时候注册 `matpl/nep/kk`
- 什么时候把 `pair_nep_kokkos.cpp` 加入 `target_sources`

这样就能把启用条件收紧为：

- `PKG_MATPL-NEP=yes`
- `PKG_KOKKOS=yes`
- `Kokkos_ENABLE_CUDA=yes`

这比把它直接丢进 `src/KOKKOS/` 更可控。

## 6. 推荐的 CMake 改造方式

### 6.1 根 `cmake/CMakeLists.txt` 需要做的两处正式注册

#### 变化 1：把 `MATPL-NEP` 加入 `STANDARD_PACKAGES`

推荐 diff：

```diff
 set(STANDARD_PACKAGES
   ...
   MACHDYN
+  MATPL-NEP
   MANIFOLD
   MANYBODY
   ...
 )
```

逐行解释：

- `+  MATPL-NEP`
  这一行的作用是把 `MATPL-NEP` 变成 LAMMPS CMake 认识的正式 package。
  加进去之后，CMake 才会自动生成 `PKG_MATPL-NEP` 选项，并在标准 package 循环中扫描 `src/MATPL-NEP/`。

#### 变化 2：在第二个 `include(Packages/...)` 循环中加入 `MATPL-NEP`

推荐 diff：

```diff
-foreach(PKG_WITH_INCL CORESHELL DPD-BASIC DPD-SMOOTH MC MISC PHONON QEQ OPENMP KOKKOS OPT INTEL GPU)
+foreach(PKG_WITH_INCL CORESHELL DPD-BASIC DPD-SMOOTH MC MISC PHONON QEQ OPENMP KOKKOS MATPL-NEP OPT INTEL GPU)
   if(PKG_${PKG_WITH_INCL})
     include(Packages/${PKG_WITH_INCL})
   endif()
 endforeach()
```

逐行解释：

- `+ MATPL-NEP`
  这一项不是为了让 package “被发现”，而是为了让 `MATPL-NEP.cmake` 有机会执行自定义逻辑。

- 为什么放在 `KOKKOS` 后面：
  因为 `MATPL-NEP.cmake` 需要用到已经初始化好的 KOKKOS 构建环境，并且需要在 `KOKKOS` 已知的情况下判断是否允许启用 `matpl/nep/kk`。

### 6.2 新增 `cmake/Modules/Packages/MATPL-NEP.cmake`

建议这个模块只负责 3 件事：

1. 只有在 `PKG_KOKKOS=yes` 且 `Kokkos_ENABLE_CUDA=yes` 时，才注册 `pair_nep_kokkos.*`
2. 收集 `nep_gpu/*.cu`
3. 添加 package 的额外 include 目录

推荐骨架如下：

```cmake
set(MATPL_NEP_SOURCES_DIR ${LAMMPS_SOURCE_DIR}/MATPL-NEP)

if(PKG_KOKKOS)
  if(NOT Kokkos_ENABLE_CUDA)
    message(FATAL_ERROR
      "MATPL-NEP KOKKOS interface requires Kokkos_ENABLE_CUDA=yes")
  endif()

  enable_language(CUDA)

  set_property(GLOBAL PROPERTY MATPL_NEP_KOKKOS_SOURCES "")
  RegisterStylesExt(${MATPL_NEP_SOURCES_DIR}/kokkos kokkos MATPL_NEP_KOKKOS_SOURCES)
  get_property(MATPL_NEP_KOKKOS_SOURCES GLOBAL PROPERTY MATPL_NEP_KOKKOS_SOURCES)

  file(GLOB MATPL_NEP_CUDA_SOURCES CONFIGURE_DEPENDS
    ${MATPL_NEP_SOURCES_DIR}/nep_gpu/force/*.cu
    ${MATPL_NEP_SOURCES_DIR}/nep_gpu/utilities/*.cu
  )

  target_sources(lammps PRIVATE
    ${MATPL_NEP_KOKKOS_SOURCES}
    ${MATPL_NEP_CUDA_SOURCES}
  )

  target_include_directories(lammps PRIVATE
    ${MATPL_NEP_SOURCES_DIR}
    ${MATPL_NEP_SOURCES_DIR}/kokkos
    ${MATPL_NEP_SOURCES_DIR}/nep_gpu/force
    ${MATPL_NEP_SOURCES_DIR}/nep_gpu/utilities
  )
endif()
```

逐段解释：

- `set(MATPL_NEP_SOURCES_DIR ...)`
  定义 package 根目录，后面所有路径都基于它展开。

- `if(PKG_KOKKOS)`
  只在启用了 KOKKOS 包时，才进一步考虑 `matpl/nep/kk`。

- `if(NOT Kokkos_ENABLE_CUDA) ...`
  这是当前方案里很重要的一道硬限制。
  因为目前 `pair_nep_kokkos.cpp` 明确依赖 CUDA 设备路径，所以不建议在“只开了 KOKKOS host backend”时尝试编这个接口。

- `enable_language(CUDA)`
  这是用 package 模块自我启用 CUDA 的关键步骤。
  有了这一步，就不再需要 patch 根 `project(... LANGUAGES CUDA)`。

- `RegisterStylesExt(${MATPL_NEP_SOURCES_DIR}/kokkos kokkos ...)`
  这一行的意思是：
  去 `src/MATPL-NEP/kokkos/` 里查找与基础样式同名的 `_kokkos` 变体，例如：
  - 基础样式：`pair_nep.h`
  - 扩展样式：`pair_nep_kokkos.h`

- `file(GLOB MATPL_NEP_CUDA_SOURCES ...)`
  显式把 `.cu` 文件收集起来。
  这是标准 package 自动扫描做不到的，所以必须在这里补。

- `target_sources(lammps PRIVATE ...)`
  把 `pair_nep_kokkos.cpp` 和 `nep_gpu/*.cu` 加入 LAMMPS 主目标。

- `target_include_directories(lammps PRIVATE ...)`
  让编译器能找到：
  - `pair_nep.h`
  - `pair_nep_kokkos.h`
  - `nepkk.cuh`
  - `error.cuh`
  等头文件。

## 7. 目录迁移建议

从当前 `lmp_nepkokkos_cmake_lmp2026` 到目标 package 的建议迁移如下：

### 7.1 CPU 文件

```txt
lmp_nepkokkos_cmake_lmp2026/pair_nep.cpp   -> lammps2026/src/MATPL-NEP/pair_nep.cpp
lmp_nepkokkos_cmake_lmp2026/pair_nep.h     -> lammps2026/src/MATPL-NEP/pair_nep.h
lmp_nepkokkos_cmake_lmp2026/nep_cpu.cpp    -> lammps2026/src/MATPL-NEP/nep_cpu.cpp
lmp_nepkokkos_cmake_lmp2026/nep_cpu.h      -> lammps2026/src/MATPL-NEP/nep_cpu.h
```

### 7.2 KOKKOS 文件

```txt
lmp_nepkokkos_cmake_lmp2026/KOKKOS/pair_nep_kokkos.cpp
  -> lammps2026/src/MATPL-NEP/kokkos/pair_nep_kokkos.cpp

lmp_nepkokkos_cmake_lmp2026/KOKKOS/pair_nep_kokkos.h
  -> lammps2026/src/MATPL-NEP/kokkos/pair_nep_kokkos.h
```

### 7.3 CUDA 核心文件

```txt
lmp_nepkokkos_cmake_lmp2026/nep_gpu/*
  -> lammps2026/src/MATPL-NEP/nep_gpu/*
```

### 7.4 删除或退役的脚本

```txt
lmp_nepkokkos_cmake_lmp2026/kknep-patch_2026.sh
```

这个脚本在标准包化之后应该退役，因为：

- 不再需要复制文件到多个位置
- 不再需要 patch 根 `cmake/CMakeLists.txt`

## 8. 推荐的演进步骤

建议分 3 步做，不要一口气把所有东西一起改：

### 第一步：只做包结构和 CMake 接入重构

目标：

- 不改业务逻辑
- 只改目录结构和构建接入方式

要验证的事情：

- `pair_style matpl/nep` 可以编进来
- `pair_style matpl/nep/kk` 可以编进来
- `lmp -h` 里能看到相关 pair style

### 第二步：统一 CPU 与 KOKKOS 版输入语法

当前 KOKKOS 版 `matpl/nep/kk` 已经被你改成：

```txt
pair_style  matpl/nep/kk
pair_coeff  * * nep.txt Hf O
```

但 CPU 版 `matpl/nep` 仍然还是旧语法。

这不影响 package 化本身，但会影响用户体验。
因此建议作为第二步把 CPU 版也统一到同样的 `pair_coeff` 语法。

### 第三步：补文档和测试

至少补这几类内容：

- README 里的编译命令
- README 里的单模型和多模型示例
- 最小回归测试输入
- 说明 `matpl/nep/kk` 目前要求 `PKG_KOKKOS=yes` 且 `Kokkos_ENABLE_CUDA=yes`

## 9. 为什么不推荐的两条路线

### 9.1 不推荐路线 A：继续使用“复制 + patch 根 CMake”

缺点：

- 对上游 CMake 布局敏感
- 构建逻辑分散
- 不利于后续维护
- 很难扩展成真正的 package 体系

### 9.2 不推荐路线 B：把 `pair_nep_kokkos.*` 直接继续放在 `src/KOKKOS/`

缺点：

- 会和 `KOKKOS.cmake` 的自动 `_kokkos` 样式发现机制耦合得过紧
- 很难只在 `Kokkos_ENABLE_CUDA=yes` 时启用
- 容易在“开了 KOKKOS 但没开 CUDA”时误触发编译

## 10. 这个方案和当前库相比，真正减少了什么改动

看起来这个方案还是需要改 `lammps2026/cmake/CMakeLists.txt`，好像并不是“零改动”。

这里要区分两种改动：

### 当前方案的改动类型

- 每次拿新的 LAMMPS 源码，都要跑脚本 patch 根 CMake
- patch 是基于文本插入
- 容易受上游段落结构影响

### 标准包化方案的改动类型

- 对上游只做一次性、正式的 package 注册
- 之后的构建逻辑都收敛到 `Packages/MATPL-NEP.cmake`
- 不再依赖脆弱的 awk 文本 patch

所以这个方案减少的不是“改动次数为 0”，而是把改动从“每次构建前的非正式 patch”变成“正式 package 注册 + 自己的 package 模块”。

## 11. 方案的最终形态

如果按这个方案收口，最终用户的构建方式应该类似：

```bash
cmake -C ../cmake/presets/basic.cmake \
  -DPKG_MATPL-NEP=yes \
  -DPKG_KOKKOS=yes \
  -DKokkos_ENABLE_CUDA=yes \
  -DKokkos_ENABLE_OPENMP=yes \
  -DFFT_KOKKOS=CUFFT \
  ../cmake
```

用户不再需要：

- 手动复制 `pair_nep.cpp` 到 `src/`
- 手动复制 `pair_nep_kokkos.cpp` 到 `src/KOKKOS/`
- 手动复制 `nep_gpu/` 到 `src/`
- 手动 patch 根 `cmake/CMakeLists.txt`

## 12. 建议的实际落地顺序

如果你准备真的开始改，我建议按下面顺序推进：

1. 先在一个新分支里建 `src/MATPL-NEP/` 目录，把文件移动过去
2. 新建 `cmake/Modules/Packages/MATPL-NEP.cmake`
3. 只做最小的 `CMakeLists.txt` 正式注册
4. 先验证能否对象级编译通过
5. 再验证 `lmp -h` 能否正确列出 `matpl/nep` 和 `matpl/nep/kk`
6. 最后再做 CPU/KOKKOS 输入语法统一

## 13. 我的推荐

如果目标是：

- 只服务 LAMMPS 2026
- 不追求和 2024 共用一套源码
- 减少后续维护成本

那么最值得做的，就是：

- 采用 `MATPL-NEP` 标准 package 方案
- `pair_nep_kokkos.*` 放到 package 自己的 `kokkos/` 子目录
- 用 `MATPL-NEP.cmake` 管理 `RegisterStylesExt(...)` 和 `.cu` 源文件
- 彻底退役 `kknep-patch_2026.sh`

这是当前代码基础上，最稳、最清晰、也最容易长期维护的一条路。
