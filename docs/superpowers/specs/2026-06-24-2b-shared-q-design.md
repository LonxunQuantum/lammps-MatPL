# Two-Body Descriptor Shared-q Design

## Goal

Remove the per-neighbor global read-modify-write of radial descriptors in
`calc_2b_descriptor_sharemem` without increasing the 80-register V3-A kernel
or reintroducing local-memory spills.

## Scope

- Modify only `nep_gpu/force/nep_kernal_function.cuh` and the corresponding
  launch in `nep_gpu/force/nepkk.cu`.
- Preserve the C2, `g_Fp`, `g_NL_angular`, and `g_NN_angular` layouts.
- Preserve the neighbor traversal and floating-point accumulation order.
- Do not change block size, three-body descriptor, ANN, backward-force, or ZBL
  code.

## Design

The dynamic shared-memory allocation is split into the existing compact C2
coefficients followed by one radial descriptor accumulator per thread:

```text
s_c[compact_c2_elements]
s_q[n_max_radial_plus1][blockDim.x]
```

`s_q[n * blockDim.x + threadIdx.x]` gives conflict-free accesses when a warp
updates one descriptor channel. Each active thread initializes its own entries,
adds each neighbor's register-resident `gn12[n]`, and writes each radial
descriptor to `g_Fp` once after the neighbor loop. `reset_nep_data` already
zeros `g_Fp`; the final store therefore assigns the accumulated value.

The launch adds
`BLOCK_SIZE32 * n_max_radial_plus1 * sizeof(NEP_FLOAT)` bytes to the existing
C2 allocation. WMoTaV adds 640 bytes, for 3.52 KiB dynamic shared memory.

## Verification

- Existing V3-A NCU report is the RED baseline: about 32.48M global-store
  instructions.
- Build must finish with target `lmp` at 100%.
- WMoTaV force and per-atom energy differences must remain below `1e-5`.
- NCU must report zero local loads/stores and significantly fewer global
  loads/stores.
- Registers must not exceed the V3-A value of 80 and achieved occupancy must
  remain at least 45%.
- NSYS kernel time must improve by at least 3% relative to V3-A; otherwise the
  implementation is rejected.

