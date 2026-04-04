set(MATPL_NEP_SOURCES_DIR ${LAMMPS_SOURCE_DIR}/MATPL-NEP)

if(NOT PKG_KOKKOS)
  message(FATAL_ERROR
    "MATPL-NEP is a KOKKOS-only NEP accelerator package. Enable PKG_KOKKOS=yes.")
endif()

if(NOT Kokkos_ENABLE_CUDA)
  message(FATAL_ERROR
    "MATPL-NEP requires Kokkos_ENABLE_CUDA=yes because this package only provides matpl/nep/kk and nep_gpu CUDA sources.")
endif()

enable_language(CUDA)

set_property(TARGET lammps PROPERTY CUDA_STANDARD 17)
set_property(TARGET lammps PROPERTY CUDA_STANDARD_REQUIRED ON)

if(DEFINED KOKKOS_CUDA_ARCHITECTURES AND NOT "${KOKKOS_CUDA_ARCHITECTURES}" STREQUAL "")
  set(MATPL_NEP_CUDA_ARCHITECTURES "${KOKKOS_CUDA_ARCHITECTURES}")
elseif(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
  set(MATPL_NEP_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(FATAL_ERROR
    "MATPL-NEP needs a CUDA architecture for its .cu sources, but none was provided. "
    "Please set -DCMAKE_CUDA_ARCHITECTURES=80 (or 86/89/90 as appropriate), "
    "or enable a Kokkos_ARCH_* option so KOKKOS_CUDA_ARCHITECTURES is populated.")
endif()

set_property(TARGET lammps PROPERTY CUDA_ARCHITECTURES "${MATPL_NEP_CUDA_ARCHITECTURES}")

if(PREC_NEPINFER)
  target_compile_definitions(lammps PRIVATE PREC_NEPINFER)
  message(STATUS "MATPL-NEP: enabling PREC_NEPINFER for double-precision NEP inference")
endif()

RegisterStyles(${MATPL_NEP_SOURCES_DIR}/kokkos)

set(MATPL_NEP_KOKKOS_SOURCES
  ${MATPL_NEP_SOURCES_DIR}/kokkos/pair_nep_kokkos.cpp
)

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

message(STATUS "MATPL-NEP: enabling matpl/nep/kk with KOKKOS + CUDA (arch=${MATPL_NEP_CUDA_ARCHITECTURES})")
