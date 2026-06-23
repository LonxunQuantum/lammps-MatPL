# NEP 3B Descriptor/ANN Split Design

## Scope

Optimize `calc_3b_descriptor_sharemem` without templates and without changing
`backward_force_3b_dqnl`.

## Design

1. Replace the per-thread `fn12[MAX_NUM_N]` array with a streaming helper that
   generates each Chebyshev basis value and immediately accumulates its C3
   coefficient contribution. Preserve basis order and floating-point operation
   order as closely as possible.
2. Make the 3B descriptor kernels responsible only for angular descriptor and
   `sum_fxyz` generation. Write raw angular q values directly to the existing
   descriptor-major `g_Fp[d * nlocal + atomi]` storage.
3. Launch a separate thread-per-atom ANN kernel. It reads raw q values from
   `g_Fp`, applies `Q_SCALER` while calculating each hidden-neuron dot product,
   accumulates energy and derivatives, and overwrites `g_Fp` only after all raw
   q reads for the atom are complete. This makes the in-place transition safe
   without an extra global buffer.
4. Retain the generic runtime dimensions and the shared/non-shared C3 paths.
   Keep the initial ANN launch at 64 threads per block so the split is measured
   independently from block-size tuning.

## Verification

- Compile the patched LAMMPS build with CUDA 12.4.
- Compare energy, force, and virial output against commit `dbb5b2b` using the
  same WMoTaV input.
- Inspect kernel resources with `cuobjdump` and Nsight Compute.
- Compare the sum of descriptor and ANN kernel time with the original 22.05 ms
  average, not descriptor time alone.

