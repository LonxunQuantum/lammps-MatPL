# NULL mapping validation

The production changes are copied byte for byte from the tested development
snapshot `49338ab5a18b6f8c52aae520e2cfb4aa3d564b61`. The implementation was
completed in `7215354` (CPU), `1edc50b` (Kokkos) and `ef4fb39` (DP).
This branch starts directly from main `21fabf34eadc57a717f4c7fbce9bb80aad5fd991`;
development commits containing generated evidence are not its ancestors.

## Scope

- NEP CPU, NEP Kokkos/CUDA and DP: complete type mappings and NULL filtering.
- Correct model-specific ordering, hybrid accumulation, atomic output and MPI
  behavior required by these mappings.
- Installation of the shared mapping header by `kknep-patch.sh`.
- Two small existing C++ regression tests and a portable runner.

The reduction selects files; it does not reimplement or trim the tested
algorithms. Runtime sources retain the validated behavior.

## Recorded physical verification

- CPU mapping M01-M10 and 35 groups of physical assertions passed.
- DP CPU/GPU each passed 83 main checks and 3 supplemental empty-rank checks.
- Historical inputs: 56/56 passed across CPU/GPU and NVT/NPT direct/rerun.
- NEP + Stillinger-Weber: 16/16 passed across CPU/GPU, MPI and substyle order.
- FP32 NULL invariants and hybrid checks passed; NULL ML outputs were zero.
- MatPL NEP FP32 + patched official Mold square/well: 351 checks passed,
  1000-step single/two-rank checks passed, Compute Sanitizer reported 0 errors,
  and the original 2100-atom case completed 1,000,000 steps.
  Production job 4772445 completed with exit 0:0; all 1001 thermo samples and
  final coordinates were finite, with 2000 Si and 100 mold atoms preserved.
  The frozen binary's 1000-step release test, job 4772644, also passed.

The Mold result additionally requires the separately supplied official Mold
sources and flat-well virial correction. It is not supplied by NULL alone.

## Remaining qualifications

The original Stage 4 acceptance is not fully passed: the DP GPU performance
cell measured approximately 2% slower than baseline, exceeding its 1% gate;
three performance cells were inconclusive. FP32/FP64 qualification and strict
QNEP PPPM repeatability retain documented failures, including failures
reproduced by the original binaries. NULL is unsupported for charge-enabled
models. These limitations are not removed by the successful Mold case.

The small tests in this branch cover mapping semantics and type compaction;
they do not replace the full numerical, MPI, GPU or performance suites.
Generated logs, models and full development evidence remain in the original
development/test directories rather than in this source-only preparation.
