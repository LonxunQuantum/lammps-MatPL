# MATPL-NEP

`MATPL-NEP` is a package bundle for integrating the MatPL NEP interfaces into a clean LAMMPS 2026 source tree in a more standard package-style layout.

Original Repositories:
 - https://github.com/LonxunQuantum/lammps-MatPL.git
 - https://github.com/brucefan1983/NEP_CPU.git

## Directory layout

```txt
MATPL-NEP/
├── cmake/
│   └── Modules/
│       └── Packages/
│           ├── MATPL-NEP.cmake
│           └── USER-NEP.cmake
├── install_to_lammps2026.sh
└── src/
    ├── MATPL-NEP/
    │   ├── kokkos/
    │   │   ├── pair_nep_kokkos.cpp
    │   │   └── pair_nep_kokkos.h
    │   └── nep_gpu/
    │       ├── force/
    │       └── utilities/
    └── USER-NEP/
        ├── pair_NEP.cpp
        ├── pair_NEP.h
        ├── ...
        ├── nep.cpp
        └── nep.h

```

## What this package bundle includes

- `USER-NEP`: the CPU NEP package source tree
- `MATPL-NEP`: the KOKKOS/GPU NEP accelerator package source tree
- CUDA core implementation under `src/MATPL-NEP/nep_gpu/`
- Package CMake modules under `cmake/Modules/Packages/`
- A helper installer for a clean LAMMPS 2026 source tree

## How to install into a clean LAMMPS 2026 tree

Run:

```bash
bash install_to_lammps2026.sh /path/to/lammps2026 single
```

If you install the double-precision GPU variant, use:

```bash
bash install_to_lammps2026.sh /path/to/lammps2026 double
```

The installer will:

- copy `src/USER-NEP/` into the target LAMMPS tree
- copy `src/MATPL-NEP/` into the target LAMMPS tree
- copy `cmake/Modules/Packages/MATPL-NEP.cmake`
- copy `cmake/Modules/Packages/USER-NEP.cmake`
- patch `cmake/CMakeLists.txt` to register both `USER-NEP` and `MATPL-NEP`
- normalize both packages into `STANDARD_PACKAGES`
- also normalize both packages into the `PKG_WITH_INCL` loop

Important:

- this installer is intended for a clean LAMMPS 2026 source tree
- if the tree already contains the old overlay-style KOKKOS files such as `src/nep_gpu/` or `src/KOKKOS/pair_nep_kokkos.cpp`, the installer will stop and ask you to clean those first
- `USER-NEP` and `MATPL-NEP` both use package-specific CMake hooks
- both package names are inserted into `STANDARD_PACKAGES`, so CMake will still generate `PKG_USER-NEP` and `PKG_MATPL-NEP` automatically with default `OFF`
- both package names are also inserted into `PKG_WITH_INCL`, so their package-specific CMake hooks still run when you enable them

## Typical CMake configure options

```bash
cmake -C ../cmake/presets/basic.cmake \
    -DPKG_MESONT=no \
    -DPKG_JPEG=no \
    -DPKG_USER-NEP=yes \
    -DPKG_KOKKOS=yes \
    -DPKG_MATPL-NEP=yes \
    -DKokkos_ENABLE_CUDA=yes \
    -DKokkos_ENABLE_OPENMP=yes \
    -DKokkos_ENABLE_CUDA_LAMBDA=yes \
    -DFFT_KOKKOS=CUFFT \
    -DTEST_TIME=ON \
    ../cmake

cmake --build . -j N #Number of cores for parallel compilation
```

When the `double` MATPL-NEP GPU sources are installed, also add:

```bash
-DPREC_NEPINFER=ON
```

Otherwise the CMake cache variable exists, but the CUDA/C++ compile units will still build with the default single-precision NEP inference path.

## Current input syntax

### CPU style

The CPU package provides:

```txt
pair_style   nep
pair_coeff   * * nep.txt Hf O
```

### KOKKOS style

The KOKKOS style has already been migrated to the newer interface:

```txt
pair_style   matpl/nep/kk
pair_coeff   * * nep.txt Hf O
```

For multi-model deviation (May not be stable, still in processing):

```txt
pair_style   matpl/nep/kk  out_freq ${DUMP_FREQ} out_file model_devi.out
pair_coeff   * * nep0.txt nep1.txt nep2.txt nep3.txt Hf O
```
