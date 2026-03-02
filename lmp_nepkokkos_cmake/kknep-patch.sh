#!/bin/bash

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

# First modification: Update project to include CUDA
PROJECT_LINE='project(lammps CXX)'
NEW_PROJECT_LINE='project(lammps CXX CUDA)'

# Check if project line already has CUDA to avoid duplication
if grep -q "$NEW_PROJECT_LINE" "$CMAKE_FILE"; then
    echo "  - Project already includes CUDA, no change needed"
elif grep -q "$PROJECT_LINE" "$CMAKE_FILE"; then
    sed -i "s|${PROJECT_LINE}|${NEW_PROJECT_LINE}|g" "$CMAKE_FILE"
    echo "  - Project configuration updated to include CUDA"
else
    echo "Error: Unsupported LAMMPS version detected"
    echo "Current LAMMPS version is not supported."
    echo "Supported versions: 2023 and 2024 released versions of LAMMPS"
    echo "Unsupported versions: 2025 released versions, develop or stable branch code"
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
  
  message(STATUS "NEP_KK: Building with mandatory KOKKOS and CUDA")
  
  # Collect source files
  file(GLOB NEP_KK_SOURCES
    ${LAMMPS_SOURCE_DIR}/nep_gpu/force/*.cu
    ${LAMMPS_SOURCE_DIR}/nep_gpu/utilities/*.cu
  )
  
  file(GLOB NEP_KOKKOS_SOURCES 
    ${LAMMPS_SOURCE_DIR}/KOKKOS/pair_nep_kokkos.cpp
  )
  # Simply add to compilation - let Kokkos configuration handle these files
  target_sources(lammps PRIVATE ${NEP_KOKKOS_SOURCES} ${NEP_KK_SOURCES})
  # Only set include directories
  target_include_directories(lammps PRIVATE 
    ${LAMMPS_SOURCE_DIR}/KOKKOS
    ${LAMMPS_SOURCE_DIR}/nep_gpu/force
    ${LAMMPS_SOURCE_DIR}/nep_gpu/utilities
  )
endif()
###################################
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

# Copy nep_gpu directory to src/
echo "Copying nep_gpu directory to src/..."
if [ -d "nep_gpu" ]; then
    cp -rf nep_gpu "$LAMMPSROOT/src/"
    echo "  - nep_gpu directory copied successfully"
else
    echo "  Warning: nep_gpu directory not found in current directory"
fi

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
else
    echo "  Warning: KOKKOS directory not found in current directory"
fi

echo "File copy process completed successfully!"
echo ""
echo "Patch process completed successfully!"
echo ""
echo "Compilation Environment:"
echo "Recommended compilation environment: cuda/11.6 (with nvcc compiler) openmpi4.1.4 cmake/3.31.6 gcc8.n"
echo ""
echo "Compilation Process:"
echo "cd $LAMMPSROOT"
echo "mkdir build & cd build"
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
echo "    -DTEST_TIME=ON \\"
echo "    ../cmake"
echo ""
echo "cmake --build . --parallel 4 #(number of parallel compilation cores)"
