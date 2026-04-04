# MATPL-NEP

`MATPL-NEP` is a standalone KOKKOS/GPU package source tree for integrating the MatPL NEP accelerator interface into a clean LAMMPS 2026 source tree in a more standard package-style layout.

## Directory layout

```txt
MATPL-NEP/
├── cmake/
│   └── Modules/
│       └── Packages/
│           └── MATPL-NEP.cmake
├── install_to_lammps2026.sh
└── src/
    └── MATPL-NEP/
        ├── kokkos/
        │   ├── pair_nep_kokkos.cpp
        │   └── pair_nep_kokkos.h
        └── nep_gpu/
            ├── force/
            └── utilities/
```

## What this package tree includes

- KOKKOS pair style source: `matpl/nep/kk`
- CUDA core implementation under `nep_gpu/`
- A package-specific CMake module: `MATPL-NEP.cmake`
- A helper installer for a clean LAMMPS 2026 source tree
- No CPU NEP implementation is included in this package tree

## How to install into a clean LAMMPS 2026 tree

Run:

```bash
bash install_to_lammps2026.sh /path/to/lammps2026
```

The installer will:

- copy `src/MATPL-NEP/` into the target LAMMPS tree
- copy `cmake/Modules/Packages/MATPL-NEP.cmake`
- patch `cmake/CMakeLists.txt` to register `MATPL-NEP` as a standard package

Important:

- this installer is intended for a clean LAMMPS 2026 source tree
- `MATPL-NEP` does not install the base CPU NEP library
- you must provide a separate NEP CPU library that defines `pair_nep.h` / `PairNEP` / `matpl/nep`
- if the tree already contains the old overlay-style KOKKOS files such as `src/nep_gpu/` or `src/KOKKOS/pair_nep_kokkos.cpp`, the installer will stop and ask you to clean those first

## Typical CMake configure options

```bash
cmake -C ../cmake/presets/basic.cmake \
  -DPKG_MATPL-NEP=yes \
  -DPKG_KOKKOS=yes \
  -DKokkos_ENABLE_CUDA=yes \
  -DKokkos_ENABLE_OPENMP=yes \
  -DFFT_KOKKOS=CUFFT \
  ../cmake
```

Before configuring, make sure the separate CPU NEP library has already been installed into the same LAMMPS tree so that `pair_nep.h` is visible to the build system.

## Current input syntax

### KOKKOS style

The KOKKOS style has already been migrated to the newer interface:

```txt
pair_style   matpl/nep/kk
pair_coeff   * * nep.txt Hf O
```

For multi-model deviation:

```txt
pair_style   matpl/nep/kk  out_freq ${DUMP_FREQ} out_file model_devi.out
pair_coeff   * * nep0.txt nep1.txt nep2.txt nep3.txt Hf O
```
