MatPL for lammps

```txt
├── dp_lmps_demo/
├── LICENSE
├── MATPL-NEP/
├── lmp_for_cmake/
├── lmp_nepkokkos_cmake/
├── Makefile.mpi
├── MATPL/
├── nep_lmps_demo/
└── README.md
```
- `lmp_nepkokkos_cmake` is the LAMMPS force field interface for MatPL-2026.3, supporting NEP KOKKOS GPU acceleration, NEP CPU version, and DP (CPU and GPU acceleration). It is compiled using CMake.

- `MATPL-NEP` is a standalone KOKKOS/GPU package-style source tree for integrating the MatPL NEP accelerator interface into a clean LAMMPS 2026 source tree with a more standard package/CMake layout. The base CPU NEP library is expected to be provided separately.

- `MATPL` and `Makefile.mpi` are the LAMMPS force field interface for MatPL-2025.3, supporting NEP (CPU and GPU acceleration) and DP (CPU and GPU acceleration). They are compiled using Make. `nep_lmps_demo` and `dp_lmps_demo` are MD test examples for the NEP and DP force fields of MatPL-2025.3, respectively.

- The key advancement in the 2026.3 version is its use of KOKKOS to offload neighbor list construction to the GPU. This fundamental change, combined with further kernel optimizations, accelerates the NEP GPU acceleration by over an order of magnitude compared to the 2025.3 version.

- The MatPL-2026.3 LAMMPS interface requires the LAMMPS source code version not to exceed 2024.
```

For installation instructions, please refer to http://doc.lonxun.com/MatPL/install/
