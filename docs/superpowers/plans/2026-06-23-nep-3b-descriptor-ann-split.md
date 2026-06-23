# NEP 3B Descriptor/ANN Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `fn12[MAX_NUM_N]` and split 3B descriptor generation from ANN evaluation without template specialization.

**Architecture:** Descriptor kernels stream the angular basis/C3 dot product and write raw q values directly to SoA `g_Fp`. A new thread-per-atom ANN kernel consumes those q values and overwrites the same atom's entries with scaled derivatives after finishing all reads.

**Tech Stack:** CUDA 12.4, C++/CUDA, LAMMPS Kokkos, Nsight Compute.

---

### Task 1: Establish failing structural and numerical checks

**Files:**
- Modify: `nep_gpu/utilities/nep_utilities.cuh`
- Modify: `nep_gpu/force/nep_kernal_function.cuh`
- Modify: `nep_gpu/force/nepkk.cu`

- [x] Record the `dbb5b2b` WMoTaV energy/force/virial baseline.
- [x] Run source assertions that fail while `fn12[MAX_NUM_N]` remains and the ANN call remains inside descriptor kernels.

### Task 2: Stream the C3 basis dot product

- [x] Add a runtime-sized streaming helper beside `find_fn` that returns `gn12` without an array.
- [x] Replace the `fn12` allocation and coefficient loop in both 3B descriptor variants.
- [x] Compile and compare numerical output with the baseline.

### Task 3: Split descriptor and ANN

- [x] Change `find_q` to support a strided destination and write angular q directly to SoA `g_Fp`.
- [x] Remove ANN parameters, q/Fp arrays, scaling, energy, and ANN calls from both descriptor kernels.
- [x] Add a generic thread-per-atom ANN kernel that reads q from SoA `g_Fp` and performs safe in-place overwrite after all neuron work.
- [x] Launch the ANN kernel after either descriptor path using 64 threads per block.

### Task 4: Verify

- [x] Run source assertions and CUDA compilation.
- [x] Run WMoTaV numerical comparison for energy, force, and virial.
- [x] Inspect register and stack resources.
- [x] Profile descriptor plus ANN and compare their combined time and stalls with the 22.05 ms baseline.
