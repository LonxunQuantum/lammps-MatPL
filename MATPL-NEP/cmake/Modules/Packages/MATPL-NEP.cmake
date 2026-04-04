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

set_property(GLOBAL PROPERTY MATPL_NEP_KOKKOS_SOURCES "")
RegisterStylesExt(${MATPL_NEP_SOURCES_DIR}/kokkos kokkos MATPL_NEP_KOKKOS_SOURCES)
get_property(MATPL_NEP_KOKKOS_SOURCES GLOBAL PROPERTY MATPL_NEP_KOKKOS_SOURCES)

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

message(STATUS "MATPL-NEP: enabling matpl/nep/kk with KOKKOS + CUDA")
