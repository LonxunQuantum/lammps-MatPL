#!/bin/bash

set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Install the MATPL-NEP package tree into a clean LAMMPS 2026 source tree"
    echo "Usage:"
    echo "  bash install_to_lammps2026.sh /path/to/lammps2026"
    exit 0
fi

if [ $# -ne 1 ]; then
    echo "Error: please provide the LAMMPS source directory"
    echo "Usage: $0 /path/to/lammps2026"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAMMPSROOT=$1
CMAKE_FILE="$LAMMPSROOT/cmake/CMakeLists.txt"
PKG_DST_DIR="$LAMMPSROOT/src/MATPL-NEP"
PKG_CMAKE_DIR="$LAMMPSROOT/cmake/Modules/Packages"
PKG_CMAKE_FILE="$PKG_CMAKE_DIR/MATPL-NEP.cmake"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ ! -d "$LAMMPSROOT" ]; then
    echo "Error: directory '$LAMMPSROOT' does not exist"
    exit 1
fi

if [ ! -f "$CMAKE_FILE" ]; then
    echo "Error: '$CMAKE_FILE' not found"
    exit 1
fi

for legacy_path in \
    "$LAMMPSROOT/src/nep_gpu" \
    "$LAMMPSROOT/src/KOKKOS/pair_nep_kokkos.cpp" \
    "$LAMMPSROOT/src/KOKKOS/pair_nep_kokkos.h"
do
    if [ -e "$legacy_path" ]; then
        echo "Error: found legacy NEP overlay path '$legacy_path'"
        echo "Please start from a clean LAMMPS 2026 source tree, or remove the old overlay-style NEP patch first."
        exit 1
    fi
done

mkdir -p "$LAMMPSROOT/src"
mkdir -p "$PKG_CMAKE_DIR"

cp "$CMAKE_FILE" "$CMAKE_FILE.bk.$TIMESTAMP"

rm -rf "$PKG_DST_DIR"
cp -rf "$SCRIPT_DIR/src/MATPL-NEP" "$LAMMPSROOT/src/"
cp -f "$SCRIPT_DIR/cmake/Modules/Packages/MATPL-NEP.cmake" "$PKG_CMAKE_FILE"

TMP_FILE=$(mktemp)
awk '
BEGIN {
    inserted_pkg = 0
    inserted_loop = 0
}
{
    line = $0

    if (!inserted_pkg && line ~ /^[[:space:]]*MACHDYN[[:space:]]*$/) {
        print line
        print "  MATPL-NEP"
        inserted_pkg = 1
        next
    }

    if (!inserted_loop && line ~ /^[[:space:]]*foreach[[:space:]]*[(][[:space:]]*PKG_WITH_INCL[[:space:]]+CORESHELL/) {
        sub(/KOKKOS/, "KOKKOS MATPL-NEP", line)
        inserted_loop = 1
        print line
        next
    }

    print line
}
END {
    if (!inserted_pkg) exit 2
    if (!inserted_loop) exit 3
}' "$CMAKE_FILE" > "$TMP_FILE"

STATUS=$?
if [ $STATUS -ne 0 ]; then
    rm -f "$TMP_FILE"
    echo "Error: failed to patch '$CMAKE_FILE'"
    echo "A backup is available at '$CMAKE_FILE.bk.$TIMESTAMP'"
    exit 1
fi

mv "$TMP_FILE" "$CMAKE_FILE"

echo "MATPL-NEP package tree installed successfully."
echo "Copied package sources to: $PKG_DST_DIR"
echo "Copied package module to: $PKG_CMAKE_FILE"
echo "Patched CMake package registration in: $CMAKE_FILE"
echo
echo "Typical configure options:"
echo "  -DPKG_MATPL-NEP=yes"
echo "  -DPKG_KOKKOS=yes"
echo "  -DKokkos_ENABLE_CUDA=yes"
