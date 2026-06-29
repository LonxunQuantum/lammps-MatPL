#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <lammpsroot>"
    echo "Patch a LAMMPS source tree for MatPL packages using a modular CMake integration."
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAMMPSROOT=$1
CMAKE_FILE="$LAMMPSROOT/cmake/CMakeLists.txt"
PACKAGES_DIR="$LAMMPSROOT/cmake/Modules/Packages"
MODULES_DIR="$LAMMPSROOT/cmake/Modules"
MATPL_MODULE="$MODULES_DIR/MatPLPackages.cmake"
SOURCE_MATPL_MODULE="$SCRIPT_DIR/MatPLPackages.cmake"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ ! -d "$LAMMPSROOT" ]]; then
    echo "Error: Directory '$LAMMPSROOT' does not exist"
    exit 1
fi

if [[ ! -f "$CMAKE_FILE" ]]; then
    echo "Error: CMakeLists.txt not found in $LAMMPSROOT/cmake/"
    exit 1
fi

if [[ ! -f "$SOURCE_MATPL_MODULE" ]]; then
    echo "Error: Source module '$SOURCE_MATPL_MODULE' not found"
    exit 1
fi

backup_file() {
    local file_path=$1
    if [[ -f "$file_path" ]]; then
        cp "$file_path" "$file_path.bk.$TIMESTAMP"
        echo "  - Backed up $(basename "$file_path") as $(basename "$file_path").bk.$TIMESTAMP"
    fi
}

insert_before_first_match() {
    local file_path=$1
    local marker_line=$2
    local block=$3
    local temp_file
    temp_file=$(mktemp)

    awk -v marker_line="$marker_line" -v block="$block" '
        BEGIN { inserted = 0 }
        {
            if (!inserted && $0 == marker_line) {
                print block
                inserted = 1
            }
            print
        }
        END {
            if (!inserted) {
                exit 1
            }
        }
    ' "$file_path" > "$temp_file"

    mv "$temp_file" "$file_path"
}

insert_after_named_foreach() {
    local file_path=$1
    local foreach_line=$2
    local block=$3
    local temp_file
    temp_file=$(mktemp)

    awk -v foreach_line="$foreach_line" -v block="$block" '
        BEGIN { in_target = 0; inserted = 0 }
        {
            print
            if (!in_target && $0 == foreach_line) {
                in_target = 1
                next
            }
            if (in_target && $0 ~ /^endforeach\(\)$/) {
                print block
                inserted = 1
                in_target = 0
            }
        }
        END {
            if (!inserted) {
                exit 1
            }
        }
    ' "$file_path" > "$temp_file"

    mv "$temp_file" "$file_path"
}

copy_file_if_exists() {
    local source_file=$1
    local destination_dir=$2

    if [[ -f "$source_file" ]]; then
        cp -f "$source_file" "$destination_dir/"
        echo "  - Copied $(basename "$source_file")"
    else
        echo "  Warning: $(basename "$source_file") not found in $SCRIPT_DIR"
    fi
}

copy_dir_if_exists() {
    local source_dir=$1
    local destination_dir=$2

    if [[ -d "$source_dir" ]]; then
        cp -rf "$source_dir" "$destination_dir/"
        echo "  - Copied $(basename "$source_dir") directory"
    else
        echo "  Warning: $(basename "$source_dir") directory not found in $SCRIPT_DIR"
    fi
}

echo "Starting modular MatPL patch process for LAMMPS source directory: $LAMMPSROOT"

if grep -q 'NEP_KK requires KOKKOS package' "$CMAKE_FILE" && ! grep -q 'include(Packages/MATPL)' "$CMAKE_FILE"; then
    echo "Error: Detected an existing inline MatPL patch block in CMakeLists.txt"
    echo "Please restore the original LAMMPS CMakeLists.txt before using matpl-patch.sh"
    exit 1
fi

backup_file "$CMAKE_FILE"
mkdir -p "$PACKAGES_DIR"
mkdir -p "$MODULES_DIR"
backup_file "$MATPL_MODULE"

echo "Updating CMakeLists.txt with modular MatPL hooks..."

if grep -Eq 'enable_language\(CUDA\)|LANGUAGES[[:space:]]+CXX[[:space:]]+C[[:space:]]+CUDA|project\(lammps CXX CUDA\)' "$CMAKE_FILE"; then
    echo "  - CUDA language support already enabled"
elif grep -q '^set(SOVERSION 0)' "$CMAKE_FILE"; then
    insert_before_first_match "$CMAKE_FILE" 'set(SOVERSION 0)' 'enable_language(CUDA)'
    echo "  - Added enable_language(CUDA) before set(SOVERSION 0)"
elif grep -q 'project(lammps CXX)' "$CMAKE_FILE"; then
    sed -i 's|project(lammps CXX)|project(lammps CXX CUDA)|g' "$CMAKE_FILE"
    echo "  - Project configuration updated to include CUDA"
else
    echo "Error: Unsupported LAMMPS version detected"
    echo "Could not find a safe anchor for enabling CUDA language support"
    exit 1
fi

if grep -q 'option(PKG_NEP_KK ' "$CMAKE_FILE"; then
    echo "  - MatPL package options already present"
else
    insert_after_named_foreach \
        "$CMAKE_FILE" \
        'foreach(PKG ${STANDARD_PACKAGES} ${SUFFIX_PACKAGES})' \
        'option(PKG_NEP_KK "Build MatPL NEP KOKKOS package" OFF)
option(PKG_MATPLDP "Build MatPL DP package" OFF)
option(PKG_MATPLD3 "Build MatPL D3 package" OFF)'
    echo "  - Added MatPL package options"
fi

if grep -q 'include(MatPLPackages)' "$CMAKE_FILE"; then
    echo "  - MatPL package include hook already present"
else
    insert_after_named_foreach \
        "$CMAKE_FILE" \
                'foreach(PKG_WITH_INCL CORESHELL DPD-BASIC DPD-SMOOTH MC MISC PHONON QEQ OPENMP KOKKOS OPT INTEL GPU)' \
        'if(PKG_NEP_KK OR PKG_MATPLDP OR PKG_MATPLD3)
    include(MatPLPackages)
endif()'
    echo "  - Added modular MatPL include hook"
fi

cp "$SOURCE_MATPL_MODULE" "$MATPL_MODULE"
echo "  - Wrote modular package file to $MATPL_MODULE"

echo "Starting file copy process..."

mkdir -p "$LAMMPSROOT/src"
copy_file_if_exists "$SCRIPT_DIR/nep_cpu.cpp" "$LAMMPSROOT/src"
copy_file_if_exists "$SCRIPT_DIR/nep_cpu.h" "$LAMMPSROOT/src"
copy_file_if_exists "$SCRIPT_DIR/pair_nep.cpp" "$LAMMPSROOT/src"
copy_file_if_exists "$SCRIPT_DIR/pair_nep.h" "$LAMMPSROOT/src"

copy_dir_if_exists "$SCRIPT_DIR/nep_gpu" "$LAMMPSROOT/src"

mkdir -p "$LAMMPSROOT/src/KOKKOS"
copy_dir_if_exists "$SCRIPT_DIR/KOKKOS" "$LAMMPSROOT/src"

copy_dir_if_exists "$SCRIPT_DIR/MATPLDP" "$LAMMPSROOT/src"
copy_dir_if_exists "$SCRIPT_DIR/MATPLD3" "$LAMMPSROOT/src"

echo ""
echo "Patch process completed successfully!"
echo ""
echo "Compilation Process:"
echo "cd $LAMMPSROOT"
echo "mkdir build && cd build"
echo "cmake -C ../cmake/presets/basic.cmake \\" 
echo "    -DPKG_MESONT=no \\" 
echo "    -DPKG_JPEG=no \\" 
echo "    -DPKG_KOKKOS=yes \\" 
echo "    -DPKG_NEP_KK=yes \\" 
echo "    -DNEP_NV_GPU_BACKEND=ON \\" 
echo "    -DNEP_ANN_TC_COMPILED=ON \\" 
echo "    -DCMAKE_CUDA_ARCHITECTURES=80;86 \\" 
echo "    -DKokkos_ENABLE_CUDA=yes \\" 
echo "    -DKokkos_ENABLE_OPENMP=yes \\" 
echo "    -DKokkos_ENABLE_CUDA_LAMBDA=yes \\" 
echo "    -DFFT_KOKKOS=CUFFT \\" 
echo "    -DKokkos_ARCH_AMPERE86=ON \\" 
echo "    -DTEST_TIME=ON \\" 
echo "    ../cmake"
echo ""
echo "Optional MatPL switches:"
echo "  -DPREC_NEPINFER=ON"
echo "  -DNEP_NV_GPU_BACKEND=ON"
echo "  -DNEP_ANN_TC_COMPILED=ON"
echo "  -DCMAKE_CUDA_ARCHITECTURES=80;86"
echo "  -DPKG_MATPLDP=yes -DTorch_DIR=... -DCMAKE_CXX_STANDARD=17"
echo "  -DPKG_MATPLD3=yes"