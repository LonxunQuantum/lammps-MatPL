######################################################################
# MatPL packages
######################################################################

option(NEP_NV_GPU_BACKEND "Build NEP with the NVIDIA backend naming path" OFF)
option(NEP_ANN_TC_COMPILED "Compile ANN Tensor Core kernels when the selected backend supports them" OFF)

if(PKG_NEP_KK)
  if(NOT PKG_KOKKOS)
    message(FATAL_ERROR "NEP_KK requires KOKKOS package. Enable with -DPKG_KOKKOS=yes")
  endif()

  if(NOT Kokkos_ENABLE_CUDA)
    message(FATAL_ERROR "NEP_KK requires CUDA support. Enable with -DKokkos_ENABLE_CUDA=yes")
  endif()

  message(STATUS "NEP_KK: Building with mandatory KOKKOS and CUDA")

  if(NEP_NV_GPU_BACKEND)
    message(STATUS "NEP_KK: Using NVIDIA backend path (NEP_NV_GPU_BACKEND=ON)")
    if(CMAKE_CUDA_ARCHITECTURES)
      message(STATUS "NEP_KK: Respecting user-provided CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
      set_property(TARGET lammps PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    else()
      set(CMAKE_CUDA_ARCHITECTURES 80;86 CACHE STRING "CUDA architectures for NVIDIA WMMA NEP build" FORCE)
      set_property(TARGET lammps PROPERTY CUDA_ARCHITECTURES "80;86")
      message(STATUS "NEP_KK: Defaulting CMAKE_CUDA_ARCHITECTURES to 80;86 for NVIDIA Tensor Core kernels")
    endif()

    set(_nep_ann_tc_compiled 0)
    if(NEP_ANN_TC_COMPILED)
      foreach(_nep_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        set(_nep_arch_norm "${_nep_arch}")
        string(REGEX REPLACE "-(real|virtual)$" "" _nep_arch_norm "${_nep_arch_norm}")
        if(_nep_arch_norm MATCHES "^[0-9]+$" AND _nep_arch_norm GREATER_EQUAL 80)
          set(_nep_ann_tc_compiled 1)
        endif()
      endforeach()
    endif()

    if(NEP_ANN_TC_COMPILED AND _nep_ann_tc_compiled)
      message(STATUS "NEP_KK: ANN Tensor Core kernels are compiled for NVIDIA (SM80+ detected)")
    else()
      if(NEP_ANN_TC_COMPILED)
        message(STATUS "NEP_KK: ANN TC was requested, but the NVIDIA build targets pre-SM80 architectures only; compiling ANN in SIMT-only mode")
      else()
        message(STATUS "NEP_KK: ANN TC was not requested; compiling ANN in SIMT-only mode")
      endif()
    endif()
    target_compile_definitions(lammps PRIVATE NEP_NV_GPU_BACKEND=1 NEP_ANN_TC_COMPILED=${_nep_ann_tc_compiled})
  else()
    message(STATUS "NEP_KK: Using non-NVIDIA GPU backend path")
    if(NEP_ANN_TC_COMPILED)
      message(STATUS "NEP_KK: ANN Tensor Core kernels are explicitly enabled for the non-NVIDIA backend")
      target_compile_definitions(lammps PRIVATE NEP_NV_GPU_BACKEND=0 NEP_ANN_TC_COMPILED=1)
    else()
      message(STATUS "NEP_KK: ANN Tensor Core kernels are disabled; compiling ANN in SIMT-only mode")
      target_compile_definitions(lammps PRIVATE NEP_NV_GPU_BACKEND=0 NEP_ANN_TC_COMPILED=0)
    endif()
  endif()


  set(MATPL_NEP_KOKKOS_SOURCES
    ${LAMMPS_SOURCE_DIR}/KOKKOS/pair_nep_kokkos.cpp
    ${LAMMPS_SOURCE_DIR}/KOKKOS/compute_matpl_heatflux_kokkos.cpp
    ${LAMMPS_SOURCE_DIR}/KOKKOS/fix_matpl_heatflux_ave_kokkos.cpp
  )

  get_property(KOKKOS_PKG_SOURCES GLOBAL PROPERTY KOKKOS_PKG_SOURCES)
  if(KOKKOS_PKG_SOURCES)
    list(APPEND KOKKOS_PKG_SOURCES ${LAMMPS_SOURCE_DIR}/KOKKOS/compute_matpl_heatflux_kokkos.cpp)
    list(REMOVE_DUPLICATES KOKKOS_PKG_SOURCES)
    set_property(GLOBAL PROPERTY KOKKOS_PKG_SOURCES "${KOKKOS_PKG_SOURCES}")
  endif()

  if(EXISTS ${LAMMPS_SOURCE_DIR}/KOKKOS/compute_matpl_heatflux_kokkos.h)
    AddStyleHeader(${LAMMPS_SOURCE_DIR}/KOKKOS/compute_matpl_heatflux_kokkos.h COMPUTE)
  endif()

  if(EXISTS ${LAMMPS_SOURCE_DIR}/KOKKOS/fix_matpl_heatflux_ave_kokkos.h)
    AddStyleHeader(${LAMMPS_SOURCE_DIR}/KOKKOS/fix_matpl_heatflux_ave_kokkos.h FIX)
  endif()


  target_sources(lammps PRIVATE ${MATPL_NEP_KOKKOS_SOURCES})
  target_include_directories(lammps PRIVATE
    ${LAMMPS_SOURCE_DIR}/KOKKOS
    ${LAMMPS_SOURCE_DIR}
  )
  target_link_libraries(lammps PRIVATE ${CMAKE_DL_LIBS})
endif()

option(PREC_NEPINFER "Use double precision" OFF)
if(PREC_NEPINFER AND PKG_NEP_KK)
  message(STATUS "PREC_NEPINFER is ON: Using double precision for NEP model.")
  add_compile_definitions(PREC_NEPINFER)
endif()

if(PKG_MATPLDP)
  find_package(Torch REQUIRED)

  file(GLOB MATPLDP_SOURCES CONFIGURE_DEPENDS
    ${LAMMPS_SOURCE_DIR}/MATPLDP/*.cpp
  )

  target_link_libraries(lammps PUBLIC ${TORCH_LIBRARIES})
  target_sources(lammps PRIVATE ${MATPLDP_SOURCES})
  target_include_directories(lammps PRIVATE ${LAMMPS_SOURCE_DIR}/MATPLDP)
  RegisterStyles(${LAMMPS_SOURCE_DIR}/MATPLDP)
endif()

if(PKG_MATPLD3)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "PKG_MATPLD3 requires CUDA (nvcc). Please install CUDA and make nvcc available.")
  endif()

  file(GLOB MATPLD3_SOURCES CONFIGURE_DEPENDS
    ${LAMMPS_SOURCE_DIR}/MATPLD3/*.cu
    ${LAMMPS_SOURCE_DIR}/MATPLD3/*.cpp
  )

  if(NOT MATPLD3_SOURCES)
    message(FATAL_ERROR "No source files found in ${LAMMPS_SOURCE_DIR}/MATPLD3")
  endif()

  target_sources(lammps PRIVATE ${MATPLD3_SOURCES})
  target_include_directories(lammps PRIVATE
    ${LAMMPS_SOURCE_DIR}/MATPLD3
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  RegisterStyles(${LAMMPS_SOURCE_DIR}/MATPLD3)
endif()