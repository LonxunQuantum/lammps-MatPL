# nep with KOKKOS

In this LAMMPS Kokkos version of NEP, neighbor list computations are handled by Kokkos on the GPU, while energy and force calculations are performed through custom C++/CUDA operators. Compared to the earlier MatPL-2025.3 version, memory consumption has been reduced by nearly one-third, and inference speed has been significantly accelerated. Furthermore, it demonstrates excellent parallel efficiency across nodes, enabling scalability to multi-node clusters.

* This Kokkos interface for NEP supports the nep.txt forcefield files trained with both `MatPL-2025.3(and later version)` and `GPUMD`.

* The code implementation is based on the NEP file parsing, descriptor calculation, and force evaluation routines from GPUMD 4.6 (https://github.com/brucefan1983/GPUMD). Building on this foundation, we have adapted it to LAMMPS's KOKKOS neighbor list conventions, meticulously optimized the kernels for descriptor and force calculations, and incorporated a multi-model deviation calculation feature for active learning.

# how to compile

## 1. copy files to lammpsfir

We recommend using the `kknep-pach.sh` script to automatically copy the nep code to the LAMMPS directory and modify the CMakeLists.txt file. The command is as follows:

```bash
bash kknep-patch.sh the/path/of/lammps/rootdir
```

* Currently, only versions stable_29Aug2024_update4 and lammps-stable_2Aug2023_update4 are supported (igher versions are not supported ) 
* The lammps-stable_2Aug2023_update4 is more faster than stable_29Aug2024_update4 in our test.

## 2. load the compilation environment

Compilation requires OpenMPI, CUDA, CMake, and GCC. We recommend using CUDA/11.6, OpenMPI/4.1.6, CMake/3.21.6, and GCC 8.5. Note that CUDA 11.8 and later versions require a more recent driver.

## 3. doing compilation

```bash
mkdir build & cd build

cmake -C ../cmake/presets/basic.cmake \
    -DPKG_MESONT=no \
    -DPKG_JPEG=no \
    -DPKG_KOKKOS=yes \
    -DPKG_NEP_KK=yes \
    -DKokkos_ENABLE_CUDA=yes \
    -DKokkos_ENABLE_OPENMP=yes \
    -DKokkos_ENABLE_CUDA_LAMBDA=yes \
    -DFFT_KOKKOS=CUFFT \
    -DTEST_TIME=ON \
    ../cmake

cmake --build . -j N #Number of cores for parallel compilation
```

```bash
# If you also need to compile the DP interface, please import the PyTorch path, import the MKL library, and enable the C++ STD17 standard for compilation.

export Torch_DIR=\$(python -c \import torch; print(torch.utils.cmake_prefix_path)\)/Torch

# Then, add the following option in cmake:
-DTorch_DIR=\${Torch_DIR}
-DCMAKE_CXX_STANDARD=17
-DPKG_MATPLDP=yes

# For the D3 interface, please add the following option in cmake. Note that D3 requires CUDA support and cannot be used in combination with matpl/nep/kk.
-DPKG_MATPLD3=yes

# NEP adopts single-precision inference by default. For double-precision inference, please add the following option in cmake.
-DPREC_NEPINFER=ON
```

# how to use
Reference case ./examples

```txt
pair_style   matpl/nep/kk
pair_coeff   * * nep.txt Hf O
```

```txt
# for multi-model deviation
pair_style   matpl/nep/kk  out_freq ${DUMP_FREQ} out_file model_devi.out
pair_coeff   * * nep0.txt nep1.txt nep2.txt nep3.txt Hf O

```

## Migration from old syntax to new syntax

This `lmp_nepkokkos_cmake_lmp2026` directory now uses a more standard LAMMPS interface:

- `pair_style` only parses global options such as `out_freq` and `out_file`
- NEP model file paths are now parsed in `pair_coeff`

### 1. Input script difference

#### single-model case

```diff
-pair_style   matpl/nep/kk  nep.txt
-pair_coeff   * * Hf O
+pair_style   matpl/nep/kk
+pair_coeff   * * nep.txt Hf O
```

Line-by-line explanation:

- `-pair_style   matpl/nep/kk  nep.txt`
  In the old syntax, the model file `nep.txt` was appended directly to `pair_style`.
- `-pair_coeff   * * Hf O`
  In the old syntax, `pair_coeff` only provided element mapping, and did not contain the model file path.
- `+pair_style   matpl/nep/kk`
  In the new syntax, `pair_style` only declares the pair style itself.
- `+pair_coeff   * * nep.txt Hf O`
  In the new syntax, `pair_coeff` now carries both the model file path and the element mapping, which is closer to common LAMMPS conventions.

#### multi-model deviation case

```diff
-pair_style   matpl/nep/kk  nep0.txt nep1.txt nep2.txt nep3.txt out_freq ${DUMP_FREQ} out_file model_devi.out
-pair_coeff   * * Hf O
+pair_style   matpl/nep/kk  out_freq ${DUMP_FREQ} out_file model_devi.out
+pair_coeff   * * nep0.txt nep1.txt nep2.txt nep3.txt Hf O
```

Line-by-line explanation:

- `-pair_style   matpl/nep/kk  nep0.txt nep1.txt nep2.txt nep3.txt out_freq ${DUMP_FREQ} out_file model_devi.out`
  In the old syntax, both the model file list and the runtime options were mixed together in `pair_style`.
- `-pair_coeff   * * Hf O`
  In the old syntax, `pair_coeff` still only contained element labels.
- `+pair_style   matpl/nep/kk  out_freq ${DUMP_FREQ} out_file model_devi.out`
  In the new syntax, only true pair-style options remain in `pair_style`.
- `+pair_coeff   * * nep0.txt nep1.txt nep2.txt nep3.txt Hf O`
  In the new syntax, the full model list is moved into `pair_coeff`, followed by one element label for each LAMMPS atom type.

### 2. Code difference in `pair_nep_kokkos.cpp`

The migration is implemented by moving model-file parsing from `settings()` to `coeff()`.

#### `settings()` change

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

Line-by-line explanation:

- `- if (narg < 1) ...`
  This old check assumed `pair_style` must contain at least one extra argument, because the old syntax expected model files here.
- `- num_ff = 0;` to `- while(iarg < narg) { ... .txt ... }`
  This block was the old model-file parser. It scanned `pair_style` arguments and collected every `.txt` file.
- `+ rank = comm->me;`
  This explicitly initializes the MPI rank even when no model file is loaded yet.
- `+ device_id = -1;`
  This initializes `device_id` to a safe sentinel value before optional CUDA device query.
- `+ num_ff = 0;`
  This now resets the model count before `pair_coeff` later reloads the true file list.
- `+ potential_files.clear();`
  This ensures stale model paths from a previous invocation are removed.
- `+ nep_gpu_models.clear();`
  This ensures old model objects are cleared before reconfiguration.
- `+ if (explrError_fp != nullptr) { ... }`
  This closes any old deviation output handle before resetting the style.
- `+ if (iarg + 1 >= narg) ... out_freq`
  This adds explicit validation that `out_freq` is followed by a value.
- `+ if (iarg + 1 >= narg) ... out_file`
  This adds explicit validation that `out_file` is followed by a filename.
- `+ else if (std::string(arg[iarg]).find(".txt") != std::string::npos) { ... }`
  This is the key migration guard: model files are no longer accepted in `pair_style`.
- `+ else { error->all(...) }`
  This rejects unknown `pair_style` options early instead of silently ignoring them.
- `- nep_gpu_models.resize(num_ff); ...`
  These lines were removed from `settings()` because model loading now belongs to `pair_coeff`.

#### `coeff()` change

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
+    error->all(FLERR, "pair_coeff for matpl/nep/kk must include at least one NEP model file after the atom type ranges");
+  if (iarg >= narg)
+    error->all(FLERR, "pair_coeff for matpl/nep/kk must include element mapping after the NEP model file(s)");
+  if (narg - iarg != atom->ntypes)
+    error->all(FLERR, "pair_coeff for matpl/nep/kk must provide one element label for each LAMMPS atom type");
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

Line-by-line explanation:

- `+ if (narg < 4) ...`
  This adds a minimum sanity check for `pair_coeff * * nep.txt X`.
- `+ utils::bounds(...)`
  These two lines restore standard LAMMPS handling for the first two `pair_coeff` arguments, which are the atom-type ranges.
- `+ potential_files.clear();`
  This clears any previous model-file list before re-reading the new one.
- `+ num_ff = 0;`
  This resets the model counter before parsing the new file list.
- `+ int iarg = 2;`
  Parsing now starts after the two atom-type range arguments.
- `+ while (iarg < narg) { ... .txt ... }`
  This is the new model-file parser, now located in `coeff()`.
- `+ if (num_ff == 0) ...`
  This enforces that at least one NEP model file must appear in `pair_coeff`.
- `+ if (iarg >= narg) ...`
  This enforces that element mapping must still be present after the model files.
- `+ if (narg - iarg != atom->ntypes) ...`
  This enforces one element label for each LAMMPS atom type.
- `+ nep_gpu_models.clear();`
  This removes any old model objects before loading the new file list.
- `+ nep_gpu_models.resize(num_ff);`
  This allocates exactly one model object per NEP file.
- `+ nep_gpu_models[i].read_neptxt(...)`
  Actual model loading is now performed in `coeff()`, not in `settings()`.
- `+ setflag[i][j] = 1;`
  This explicitly marks the selected type pairs as active in standard LAMMPS fashion.
- `+ cutsq[i][j] = cutoffsq;`
  This initializes pair cutoffs after the first model has supplied the NEP cutoff.
- `- for (int ii = 2; ii < narg; ++ii)`
  The old code assumed element labels always started immediately after `* *`.
- `+ for (int ii = iarg; ii < narg; ++ii)`
  The new code starts element parsing only after all `.txt` files have been consumed.
- `- if (iter != atom_type_module.end() || arg[ii] == 0)`
  The old extra condition `arg[ii] == 0` was not a correct test for a valid element token.
- `+ if (iter != atom_type_module.end())`
  The new condition only accepts elements that really exist in the loaded model.
- `- set_atom_type_map(narg-2, ...)`
  The old size formula assumed there were no model files inside `pair_coeff`.
- `+ set_atom_type_map(narg - iarg, ...)`
  The new size formula correctly counts only the element-mapping part.

### 3. Practical migration note

If your old input script contains:

```txt
pair_style matpl/nep/kk nep.txt
```

you must now move `nep.txt` into `pair_coeff`.

If your old input script contains multiple model files in `pair_style`, move the full model list into `pair_coeff`, but leave `out_freq` and `out_file` in `pair_style`.

# Citation
* If you use the LAMMPS interface of this nep-kokkos here, you are suggested to cite the following (`The article will be released soon`):

  * https://github.com/LonxunQuantum/MatPL

* If you directly or indirectly use the `NEP` class here, you are suggested to cite the following paper:

  * Ke Xu, Hekai Bu, Shuning Pan, Eric Lindgren, Yongchao Wu, Yong Wang, Jiahui Liu, Keke Song, Bin Xu, Yifan Li, Tobias Hainer, Lucas Svensson, Julia Wiktor, Rui Zhao, Hongfu Huang, Cheng Qian, Shuo Zhang, Zezhu Zeng, Bohan Zhang, Benrui Tang, Yang Xiao, Zihan Yan, Jiuyang Shi, Zhixin Liang, Junjie Wang, Ting Liang, Shuo Cao, Yanzhou Wang, Penghua Ying, Nan Xu, Chengbing Chen, Yuwen Zhang, Zherui Chen, Xin Wu, Wenwu Jiang, Esme Berger, Yanlong Li, Shunda Chen, Alexander J. Gabourie, Haikuan Dong, Shiyun Xiong, Ning Wei, Yue Chen, Jianbin Xu, Feng Ding, Zhimei Sun, Tapio Ala-Nissila, Ari Harju, Jincheng Zheng, Pengfei Guan, Paul Erhart, Jian Sun, Wengen Ouyang, Yanjing Su, Zheyong Fan, [GPUMD 4.0: A high-performance molecular dynamics package for versatile materials simulations with machine-learned potentials]( https://doi.org/10.1002/mgea.70028), MGE Advances **3**, e70028 (2025).

* If you use the LAMMPS interface of the `NEP` class, a proper citation for LAMMPS is also suggested. 
