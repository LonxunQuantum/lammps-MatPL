#!/bin/bash
set -euo pipefail

# Check for help option
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "This script is used to copy the MatPL-2026.3 lammps interface files to the corresponding lammps directory:"
    echo "bash kknep-patch.sh lammps/rootdir"
    exit 0
fi

# Check if lammps root directory is provided
if [ $# -ne 1 ]; then
    echo "Error: Please provide the lammps source directory as an argument"
    echo "Usage: $0 <lammpsroot>"
    exit 1
fi

LAMMPSROOT=$1

# Check if the provided directory exists
if [ ! -d "$LAMMPSROOT" ]; then
    echo "Error: Directory '$LAMMPSROOT' does not exist"
    exit 1
fi

echo "Starting patch process for LAMMPS source directory: $LAMMPSROOT"

# Backup CMakeLists.txt before modification with timestamp
CMAKE_FILE="$LAMMPSROOT/cmake/CMakeLists.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -f "$CMAKE_FILE" ]; then
    cp "$CMAKE_FILE" "$CMAKE_FILE.bk.$TIMESTAMP"
    echo "  - CMakeLists.txt backed up as CMakeLists.txt.bk.$TIMESTAMP"
else
    echo "  Error: CMakeLists.txt not found in $LAMMPSROOT/cmake/"
    exit 1
fi

# Modify CMakeLists.txt to add NEP_KK package and update project settings
echo "Modifying CMakeLists.txt to add CUDA support and NEP_KK package..."

# Set C++ standard to 17
if ! grep -q "set(CMAKE_CXX_STANDARD 17)" "$CMAKE_FILE"; then
    sed -i "/project(lammps/a set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)" "$CMAKE_FILE"
    echo "  - C++17 standard enabled in CMakeLists.txt"
fi

# First modification: Update project to include CUDA
PROJECT_LINE='project(lammps CXX)'
NEW_PROJECT_LINE='project(lammps CXX CUDA)'

# Check if project line already has CUDA to avoid duplication
if grep -q "$NEW_PROJECT_LINE" "$CMAKE_FILE"; then
    echo "  - Project already includes CUDA, no change needed"
elif grep -q "$PROJECT_LINE" "$CMAKE_FILE"; then
    sed -i "s|${PROJECT_LINE}|${NEW_PROJECT_LINE}|g" "$CMAKE_FILE"
    echo "  - Project configuration updated to include CUDA"
elif grep -q "LANGUAGES CXX C" "$CMAKE_FILE"; then
    sed -i "/LANGUAGES CXX C/c\        LANGUAGES CXX C CUDA)" "$CMAKE_FILE"
    echo "  - Project configuration updated to include CUDA"
else
    echo "Error: Unsupported LAMMPS version detected"
    echo "Current LAMMPS version is not supported."
    echo "Supported versions: 2023~2025 released versions of LAMMPS"
    echo "Please check and try again with a supported version."
    exit 1
fi

# Second modification: Add NEP_KK package after the packages foreach loop
# Find the position to insert the new content
INSERT_POINT=$(grep -n "foreach(PKG_WITH_INCL CORESHELL" "$CMAKE_FILE" | tail -1 | cut -d: -f1)

if [ -z "$INSERT_POINT" ]; then
    echo "  Error: Could not find the insertion point for NEP_KK package in CMakeLists.txt"
    echo "  Please check the file structure and modify manually"
    exit 1
fi

# Check if NEP_KK package is already added to avoid duplication
if grep -q "PKG_NEP_KK" "$CMAKE_FILE"; then
    echo "  - NEP_KK package already exists in CMakeLists.txt, skipping addition"
else
    # Calculate the line number after the foreach block
    END_LINE=$((INSERT_POINT + 5))

    # Create a temporary file for the modified content
    TEMP_FILE=$(mktemp)

    # Copy lines before the insertion point
    head -n $END_LINE "$CMAKE_FILE" > "$TEMP_FILE"

    # Append the new NEP_KK content
    cat >> "$TEMP_FILE" << 'EOF'

######################################################################
# package of NEP with KOKKOS
######################################################################
if(PKG_NEP_KK)
  # Mandatory dependency check
  if(NOT PKG_KOKKOS)
    message(FATAL_ERROR "NEP_KK requires KOKKOS package. Enable with -DPKG_KOKKOS=yes")
  endif()
  
  if(NOT Kokkos_ENABLE_CUDA)
    message(FATAL_ERROR "NEP_KK requires CUDA support. Enable with -DKokkos_ENABLE_CUDA=yes")
  endif()
  
  message(STATUS "NEP_KK: Building open-source Kokkos wrapper (GPU kernels via libnep_gpu.so)")

  # Collect source files — only the open-source Kokkos wrapper
  # GPU kernels are provided by libnep_gpu.so at runtime (dlopen)
  file(GLOB NEP_KOKKOS_SOURCES
    ${LAMMPS_SOURCE_DIR}/KOKKOS/pair_nep_kokkos.cpp
    ${LAMMPS_SOURCE_DIR}/KOKKOS/compute_matpl_heatflux_kokkos.cpp
    ${LAMMPS_SOURCE_DIR}/KOKKOS/fix_matpl_heatflux_ave_kokkos.cpp
  )
  target_sources(lammps PRIVATE ${NEP_KOKKOS_SOURCES})
  # Register only the MatPL heat flux styles (NOT all KOKKOS styles)
  # Using AddStyleHeader instead of RegisterStyles to avoid pulling in
  # headers from disabled packages (DPD-BASIC, SPIN, DIPOLE, etc.)
  if(EXISTS ${LAMMPS_SOURCE_DIR}/KOKKOS/compute_matpl_heatflux_kokkos.h)
    AddStyleHeader(${LAMMPS_SOURCE_DIR}/KOKKOS/compute_matpl_heatflux_kokkos.h COMPUTE)
  endif()
  if(EXISTS ${LAMMPS_SOURCE_DIR}/KOKKOS/fix_matpl_heatflux_ave_kokkos.h)
    AddStyleHeader(${LAMMPS_SOURCE_DIR}/KOKKOS/fix_matpl_heatflux_ave_kokkos.h FIX)
  endif()
  if(EXISTS ${LAMMPS_SOURCE_DIR}/KOKKOS/pair_nep_kokkos.h)
    AddStyleHeader(${LAMMPS_SOURCE_DIR}/KOKKOS/pair_nep_kokkos.h PAIR)
  endif()
  target_include_directories(lammps PRIVATE
    ${LAMMPS_SOURCE_DIR}/KOKKOS
    ${LAMMPS_SOURCE_DIR}
  )
  target_link_libraries(lammps PRIVATE ${CMAKE_DL_LIBS})
endif()
#####################################################################
# package of NEP with double precision
#####################################################################
option(PREC_NEPINFER "Use double precision" OFF)
if(PREC_NEPINFER AND PKG_NEP_KK)
    message(STATUS "PREC_NEPINFER is ON: Using double precision for NEP model.")
    add_compile_definitions(PREC_NEPINFER)
endif()

######################################################################
# package of DP with KOKKOS
######################################################################
if(PKG_MATPLDP)
  # 1. 查找 PyTorch
  find_package(Torch REQUIRED)
  # 2. 链接 PyTorch 库到 LAMMPS 库（使用 PUBLIC 以便传递给可执行文件）
  target_link_libraries(lammps PUBLIC ${TORCH_LIBRARIES})
  # 3. 添加包源文件（假设文件位于 src/MATPLDP/ 下）
  file(GLOB MATPLDP_SOURCES ${LAMMPS_SOURCE_DIR}/MATPLDP/*.cpp)
  target_sources(lammps PRIVATE ${MATPLDP_SOURCES})
  # 4. 添加包含目录（如果头文件需要被其他源文件引用）
  target_include_directories(lammps PRIVATE ${LAMMPS_SOURCE_DIR}/MATPLDP)
  RegisterStyles(${LAMMPS_SOURCE_DIR}/MATPLDP)
endif()

######################################################################
# package of D3
######################################################################
if(PKG_MATPLD3)
  # ==================== CUDA 强制检查（和 NEP_KK 一样） ====================
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "PKG_MATPLD3 requires CUDA (nvcc). 请确保 CUDA 已安装！")
  endif()
  
  message(STATUS "PKG_MATPLD3: Building with CUDA (toolkit ${CUDAToolkit_VERSION})")

  file(GLOB MATPLD3_SOURCES ${CONFIGURE_DEPENDS}
    ${LAMMPS_SOURCE_DIR}/MATPLD3/*.cu
    ${LAMMPS_SOURCE_DIR}/MATPLD3/*.cpp
  )

  message(STATUS "LAMMPS_SOURCE_DIR = ${LAMMPS_SOURCE_DIR}")
  message(STATUS "Looking for MATPLD3 files in: ${LAMMPS_SOURCE_DIR}/MATPLD3")
  if(MATPLD3_SOURCES)
    message(STATUS "MATPLD3 sources found: ${MATPLD3_SOURCES}")
  else()
    message(FATAL_ERROR "No source files found in ${LAMMPS_SOURCE_DIR}/MATPLD3")
  endif()
  target_sources(lammps PRIVATE ${MATPLD3_SOURCES})
  target_include_directories(lammps PRIVATE 
    ${LAMMPS_SOURCE_DIR}/MATPLD3
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  RegisterStyles(${LAMMPS_SOURCE_DIR}/MATPLD3)
endif()

######################################################################
EOF

    # Append the remaining content
    tail -n +$((END_LINE + 1)) "$CMAKE_FILE" >> "$TEMP_FILE"

    # Replace the original file
    mv "$TEMP_FILE" "$CMAKE_FILE"

    echo "  - CMakeLists.txt modified successfully with NEP_KK package"
fi

echo "CMakeLists.txt modification completed successfully!"

# Now copy files after successful CMakeLists.txt modification
echo "Starting file copy process..."

# Copy CPU files to src/
echo "Copying CPU files to src/ directory..."
CPU_FILES=("nep_cpu.cpp" "nep_cpu.h" "pair_nep.cpp" "pair_nep.h")
for file in "${CPU_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp -f "$file" "$LAMMPSROOT/src/"
        echo "  - $file copied successfully"
    else
        echo "  Warning: $file not found in current directory"
    fi
done

# Copy nep_gpu_loader.h (open-source wrapper) to src/
echo "Copying open-source NEP GPU loader to src/..."
if [ -f "nep_gpu_loader.h" ]; then
    cp -f nep_gpu_loader.h "$LAMMPSROOT/src/"
    echo "  - nep_gpu_loader.h copied successfully"
else
    echo "  Warning: nep_gpu_loader.h not found in current directory"
fi

echo ""
echo "=== IMPORTANT: libnep_gpu.so and License ==="
echo "Pre-compiled libnep_gpu.so files for various GPU architectures are provided"
echo "in the nep_gpu/ directory of this package. Choose the one matching your GPU"
echo "and CUDA version. See README.md for the GPU compatibility table."
echo ""
echo "A valid license file (summer_holiday.lic) is also provided in nep_gpu/."
echo "For license renewal, contact matpl@pwmat.com."
echo ""

# Copy KOKKOS files to src/KOKKOS/
echo "Copying KOKKOS files to src/KOKKOS/ directory..."
if [ -d "KOKKOS" ]; then
    mkdir -p "$LAMMPSROOT/src/KOKKOS"
    if [ -f "KOKKOS/pair_nep_kokkos.cpp" ]; then
        cp -f "KOKKOS/pair_nep_kokkos.cpp" "$LAMMPSROOT/src/KOKKOS/"
        echo "  - pair_nep_kokkos.cpp copied successfully"
    else
        echo "  Warning: KOKKOS/pair_nep_kokkos.cpp not found"
    fi
    
    if [ -f "KOKKOS/pair_nep_kokkos.h" ]; then
        cp -f "KOKKOS/pair_nep_kokkos.h" "$LAMMPSROOT/src/KOKKOS/"
        echo "  - pair_nep_kokkos.h copied successfully"
    else
        echo "  Warning: KOKKOS/pair_nep_kokkos.h not found"
    fi

    for f in compute_matpl_heatflux_kokkos.cpp compute_matpl_heatflux_kokkos.h \
             fix_matpl_heatflux_ave_kokkos.cpp fix_matpl_heatflux_ave_kokkos.h; do
        if [ -f "KOKKOS/$f" ]; then
            cp -f "KOKKOS/$f" "$LAMMPSROOT/src/KOKKOS/"
            echo "  - $f copied successfully"
        else
            echo "  Warning: KOKKOS/$f not found"
        fi
    done
else
    echo "  Warning: KOKKOS directory not found in current directory"
fi

# Copy MATPLDP and MATPLD3 directories to src/
echo "Copying MATPLDP and MATPLD3 directories to src/..."
if [ -d "MATPLDP" ]; then
    cp -rf MATPLDP "$LAMMPSROOT/src/"
    echo "  - MATPLDP directory copied successfully"
else
    echo "  Warning: MATPLDP directory not found in current directory"
fi

if [ -d "MATPLD3" ]; then
    cp -rf MATPLD3 "$LAMMPSROOT/src/"
    echo "  - MATPLD3 directory copied successfully"
else
    echo "  Warning: MATPLD3 directory not found in current directory"
fi

echo "File copy process completed successfully!"
echo ""
echo "Patch process completed successfully!"
echo ""
echo "=== Pre-requisite: libnep_gpu.so ==="
echo "Pre-compiled libnep_gpu.so files are included in the nep_gpu/ directory."
echo "Place the appropriate .so (matching your GPU architecture and CUDA version)"
echo "in LD_LIBRARY_PATH or set NEP_GPU_LIB_PATH to its full path."
echo ""
echo "The evaluation license (summer_holiday.lic) is also provided in nep_gpu/."
echo "Set NEP_LICENSE_PATH to the license file path, or place it at"
echo "~/.nep_license/license.json."
echo ""
echo "See README.md for detailed GPU compatibility table and setup instructions."
echo ""
echo "=== LAMMPS Compilation ==="
echo "Compilation Environment:"
echo "Recommended: cuda/11.6 (for Kokkos GPU neighbor lists) openmpi4.1.4 cmake/3.21+ gcc8+"
echo ""
echo "Compilation Process:"
echo "cd $LAMMPSROOT"
echo "mkdir build && cd build"
echo "cmake -C ../cmake/presets/basic.cmake \\"
echo "    -DPKG_MESONT=no \\"
echo "    -DPKG_JPEG=no \\"
echo "    -DPKG_KOKKOS=yes \\"
echo "    -DPKG_NEP_KK=yes \\"
echo "    -DKokkos_ENABLE_CUDA=yes \\"
echo "    -DKokkos_ENABLE_OPENMP=yes \\"
echo "    -DKokkos_ENABLE_CUDA_LAMBDA=yes \\"
echo "    -DFFT_KOKKOS=CUFFT \\"
echo "    -DKokkos_ARCH_AMPERE86=ON \\"
echo "    ../cmake"
echo ""
echo "cmake --build . --parallel 4"
echo ""
echo ""
echo "If you also need to compile the DP interface, please import the PyTorch path, import the MKL library, and enable the C++ STD17 standard for compilation."
echo "export Torch_DIR=\$(python -c \"import torch; print(torch.utils.cmake_prefix_path)\")/Torch"
echo "Then, add the following option in cmake:"
echo "-DTorch_DIR=\${Torch_DIR} \\"
echo "-DCMAKE_CXX_STANDARD=17 \\"
echo "-DPKG_MATPLDP=yes \\"
echo ""
echo "For the D3 interface, please add the following option in cmake. Note that D3 requires CUDA support and cannot be used in combination with matpl/nep/kk."
echo "-DPKG_MATPLD3=yes \\"
echo ""

######### 手动复制文件 #########
# # 1. 复制 CPU 相关源文件到 src/
# cp nep_cpu.cpp nep_cpu.h pair_nep.cpp pair_nep.h <LAMMPSROOT>/src/

# # 2. 复制 nep_gpu 目录（包含 GPU 加速的 NEP 代码）到 src/
# cp -r nep_gpu <LAMMPSROOT>/src/

# # 3. 复制 KOKKOS 版本的 NEP 文件到 src/KOKKOS/（若目录不存在则先创建）
# mkdir -p <LAMMPSROOT>/src/KOKKOS
# cp KOKKOS/pair_nep_kokkos.cpp KOKKOS/pair_nep_kokkos.h <LAMMPSROOT>/src/KOKKOS/

# # 4. 复制 MATPLDP 接口目录到 src/
# cp -r MATPLDP <LAMMPSROOT>/src/

# # 5. 复制 MATPLD3 接口目录到 src/
# cp -r MATPLD3 <LAMMPSROOT>/src/
######### 手动复制文件 #########
