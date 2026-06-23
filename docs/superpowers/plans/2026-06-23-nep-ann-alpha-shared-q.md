# NEP ANN Alpha Cache and Shared-Q Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ANN derivative local-memory traffic in V2-A, then remove repeated global q reads in V2-B without templates.

**Architecture:** V2-A splits ANN forward/energy from a shared-alpha tiled derivative kernel using a persistent SoA alpha buffer. V2-B stages scaled q in strided shared memory inside the forward kernel. Each version is built, numerically compared, profiled, and committed independently.

**Tech Stack:** CUDA 12.4, C++/CUDA, LAMMPS Kokkos, Nsight Systems, Nsight Compute.

---

### Task 1: V2-A Red Test and Baseline

**Files:**
- Modify: `nep_gpu/force/nepkk.cuh`
- Modify: `nep_gpu/force/nepkk.cu`
- Modify: `nep_gpu/force/nep_kernal_function.cuh`

- [ ] Save the `e5db547` sorted WMoTaV dump.
- [ ] Assert that `Fp[MAX_DIM]` is absent and both `apply_ann_forward` and `apply_ann_derivative` exist; verify the assertion fails on `e5db547`.

### Task 2: Implement and Verify V2-A

- [ ] Add `ann_alpha` to `NEPKK_Data` and resize it to `max_nlocal * annmb.num_neurons1`.
- [ ] Implement forward/energy output to `ann_alpha[n * N + n1]`.
- [ ] Implement an eight-atom, 256-thread derivative tile with shared alpha.
- [ ] Patch LAMMPS and run `cmake --build . -j4`; expect exit 0.
- [ ] Compare per-atom force and energy against `e5db547`; require maximum absolute differences below `1e-5`.
- [ ] Inspect resources and run NSYS plus five-launch full NCU for both V2-A kernels.
- [ ] Commit as `opt: cache ANN alpha for tiled derivatives`.

### Task 3: V2-B Red Test

- [ ] Assert that forward declares dynamic shared q and loads scaled q before its synchronization point; verify the assertion fails on V2-A.

### Task 4: Implement and Verify V2-B

- [ ] Add strided shared-q loading to forward and replace inner global q reads.
- [ ] Launch forward with `annmb.dim * 64 * sizeof(NEP_FLOAT)` dynamic shared memory.
- [ ] Build and compare per-atom force and energy against V2-A with the same `1e-5` threshold.
- [ ] Inspect resources and run NSYS plus five-launch full NCU for forward and derivative.
- [ ] Commit as `opt: stage ANN descriptors in shared memory`.

### Task 5: Report

- [ ] Compare e5db547, V2-A, and V2-B kernel sums, L2 throughput, local/global/shared instructions, occupancy, long-scoreboard, and LG-throttle metrics.

