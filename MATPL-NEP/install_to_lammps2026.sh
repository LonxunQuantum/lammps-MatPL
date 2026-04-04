#!/bin/bash

# This script is used to install the USER-NEP and MATPL-NEP package trees into a clean LAMMPS 2026 source tree
#     - MATPL-NEP requires KOKKOS and CUDA to be enabled
#         - single: use single precision for NEP GPU
#         - double: use double precision for NEP GPU
#     - USER-NEP is a CPU NEP package, it can be enabled independently of MATPL-NEP


set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Install the USER-NEP and MATPL-NEP package trees into a clean LAMMPS 2026 source tree"
    echo "Usage:"
    echo "  bash install_to_lammps2026.sh /path/to/lammps2026 [single|double]"
    exit 0
fi

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Error: please provide the LAMMPS source directory and optional precision mode"
    echo "Usage: $0 /path/to/lammps2026 [single|double]"
    exit 1
fi


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAMMPSROOT=$1
PRECISION_MODE=${2:-single}
CMAKE_FILE="$LAMMPSROOT/cmake/CMakeLists.txt"
MATPL_PKG_DST_DIR="$LAMMPSROOT/src/MATPL-NEP"
USER_PKG_DST_DIR="$LAMMPSROOT/src/USER-NEP"
PKG_CMAKE_DIR="$LAMMPSROOT/cmake/Modules/Packages"
MATPL_PKG_CMAKE_FILE="$PKG_CMAKE_DIR/MATPL-NEP.cmake"
USER_PKG_CMAKE_FILE="$PKG_CMAKE_DIR/USER-NEP.cmake"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case "$PRECISION_MODE" in
    single)
        NEP_GPU_SRC_DIR="$SCRIPT_DIR/src/MATPL-NEP/nep_gpu"
        ;;
    double)
        NEP_GPU_SRC_DIR="$SCRIPT_DIR/src/MATPL-NEP/nep_gpu_double"
        ;;
    *)
        echo "Error: unsupported precision mode '$PRECISION_MODE'"
        echo "Please use 'single' or 'double'."
        exit 1
        ;;
esac


# Check if the LAMMPS source tree exists, the CMakeLists.txt file exists, and the NEP_GPU source directory exists
if [ ! -d "$LAMMPSROOT" ]; then
    echo "Error: directory '$LAMMPSROOT' does not exist"
    exit 1
fi

if [ ! -f "$CMAKE_FILE" ]; then
    echo "Error: '$CMAKE_FILE' not found"
    exit 1
fi

if [ ! -d "$NEP_GPU_SRC_DIR" ]; then
    echo "Error: source NEP GPU directory '$NEP_GPU_SRC_DIR' not found"
    exit 1
fi

mkdir -p "$LAMMPSROOT/src"
mkdir -p "$PKG_CMAKE_DIR"

# Backup the original CMakeLists.txt file
cp "$CMAKE_FILE" "$CMAKE_FILE.bk.$TIMESTAMP"

# Copy the MATPL-NEP package tree to the LAMMPS source tree
rm -rf "$MATPL_PKG_DST_DIR"
cp -rf "$SCRIPT_DIR/src/MATPL-NEP" "$LAMMPSROOT/src/"

# Copy the USER-NEP package tree to the LAMMPS source tree, using the corresponding precision mode provided by the user
rm -rf "$USER_PKG_DST_DIR"
cp -rf "$SCRIPT_DIR/src/USER-NEP" "$LAMMPSROOT/src/"
rm -rf "$MATPL_PKG_DST_DIR/nep_gpu" "$MATPL_PKG_DST_DIR/nep_gpu_double"
cp -rf "$NEP_GPU_SRC_DIR" "$MATPL_PKG_DST_DIR/nep_gpu"

# Copy the package modules to the package CMake directory
cp -f "$SCRIPT_DIR/cmake/Modules/Packages/MATPL-NEP.cmake" "$MATPL_PKG_CMAKE_FILE"
cp -f "$SCRIPT_DIR/cmake/Modules/Packages/USER-NEP.cmake" "$USER_PKG_CMAKE_FILE"

TMP_FILE=$(mktemp)
awk '
BEGIN {
    in_standard_packages = 0
    inserted_user_standard_pkg = 0
    inserted_matpl_standard_pkg = 0
    normalized_include_loop = 0
}
{
    line = $0

    # Remove legacy manual options if an older installer inserted them.
    if (line ~ /^[[:space:]]*option[(]PKG_USER-NEP[[:space:]]/) {
        next
    }
    if (line ~ /^[[:space:]]*option[(]PKG_MATPL-NEP[[:space:]]/) {
        next
    }

    if (line ~ /^[[:space:]]*set[(]STANDARD_PACKAGES/) {
        in_standard_packages = 1
    }

    # Normalize STANDARD_PACKAGES by removing old occurrences first.
    if (in_standard_packages && line ~ /^[[:space:]]*USER-NEP[[:space:]]*$/) {
        next
    }
    if (in_standard_packages && line ~ /^[[:space:]]*MATPL-NEP[[:space:]]*$/) {
        next
    }

    # Reinsert both package names in a stable position.
    if (line ~ /^[[:space:]]*MACHDYN[[:space:]]*$/) {
        print line
        print "  USER-NEP"
        print "  MATPL-NEP"
        inserted_user_standard_pkg = 1
        inserted_matpl_standard_pkg = 1
        next
    }

    if (in_standard_packages && line ~ /[)][[:space:]]*$/) {
        in_standard_packages = 0
    }

    # Normalize the package-specific include loop as well.
    if (line ~ /^[[:space:]]*foreach[[:space:]]*[(][[:space:]]*PKG_WITH_INCL[[:space:]]+CORESHELL/) {
        gsub(/[[:space:]]+USER-NEP/, "", line)
        gsub(/[[:space:]]+MATPL-NEP/, "", line)
        sub(/KOKKOS/, "KOKKOS USER-NEP MATPL-NEP", line)
        normalized_include_loop = 1
    }

    print line
}
END {
    if (!inserted_user_standard_pkg) exit 2
    if (!inserted_matpl_standard_pkg) exit 3
    if (!normalized_include_loop) exit 4
}' "$CMAKE_FILE" > "$TMP_FILE"

STATUS=$?
if [ $STATUS -ne 0 ]; then
    rm -f "$TMP_FILE"
    echo "Error: failed to patch '$CMAKE_FILE'"
    echo "A backup is available at '$CMAKE_FILE.bk.$TIMESTAMP'"
    exit 1
fi

mv "$TMP_FILE" "$CMAKE_FILE"

echo "USER-NEP and MATPL-NEP package trees installed successfully."
echo "Copied USER-NEP sources to: $USER_PKG_DST_DIR"
echo "Copied MATPL-NEP sources to: $MATPL_PKG_DST_DIR"
echo "Selected NEP GPU mode: $PRECISION_MODE ($NEP_GPU_SRC_DIR -> $MATPL_PKG_DST_DIR/nep_gpu)"
echo "Copied package modules to:"
echo "  $USER_PKG_CMAKE_FILE"
echo "  $MATPL_PKG_CMAKE_FILE"
echo "Patched CMake package registration in: $CMAKE_FILE"
if [ "$PRECISION_MODE" = "double" ]; then
    echo "Note: configure CMake with -DPREC_NEPINFER=ON for the double-precision MATPL-NEP GPU sources."
fi
echo


if [ "$PRECISION_MODE" = "single" ]; then
    echo "Typical configure options (single precision):"
    echo "  -DPKG_USER-NEP=yes"
    echo "  -DPKG_MATPL-NEP=yes"
    echo "  -DPKG_KOKKOS=yes"
    echo "  -DKokkos_ENABLE_CUDA=yes"
else
    echo "Typical configure options (double precision):"
    echo "  -DPKG_USER-NEP=yes"
    echo "  -DPKG_MATPL-NEP=yes"
    echo "  -DPKG_KOKKOS=yes"
    echo "  -DKokkos_ENABLE_CUDA=yes"
    echo "  -DPREC_NEPINFER=ON"
fi
