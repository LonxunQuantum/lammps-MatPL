#!/bin/bash

# Check for help option
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Patch LAMMPS CMakeLists.txt for NEP_KK + CUDA support"
    echo "Usage:"
    echo "  bash kknep-patch.sh <lammpsroot>"
    exit 0
fi

# Check arguments
if [ $# -ne 1 ]; then
    echo "Error: Please provide the LAMMPS source directory as an argument"
    echo "Usage: $0 <lammpsroot>"
    exit 1
fi

LAMMPSROOT=$1

# Check if the provided directory exists
if [ ! -d "$LAMMPSROOT" ]; then
    echo "Error: Directory '$LAMMPSROOT' does not exist"
    exit 1
fi

CMAKE_FILE="$LAMMPSROOT/cmake/CMakeLists.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ ! -f "$CMAKE_FILE" ]; then
    echo "Error: CMakeLists.txt not found in $LAMMPSROOT/cmake/"
    exit 1
fi

echo "Starting patch process for: $LAMMPSROOT"

# Backup
cp "$CMAKE_FILE" "$CMAKE_FILE.bk.$TIMESTAMP"
echo "  - Backup created: $CMAKE_FILE.bk.$TIMESTAMP"

replace_file() {
    local src="$1"
    local tmp="$2"
    mv "$tmp" "$src"
}

###############################################################################
# 1) Add CUDA inside project(... LANGUAGES ...)
#
# Strategy:
#   - find project(lammps ...) block
#   - locate the LANGUAGES line inside it
#   - if CUDA is missing, insert CUDA before the closing ')'
#
# This is robust against indentation / extra spaces / multiline project block.
###############################################################################
echo "Patching project(... LANGUAGES ...) to include CUDA..."

if grep -Eq '^[[:space:]]*LANGUAGES([[:space:]]+[^)]*)?[[:space:]]CUDA([[:space:]]*[^)]*)?\)[[:space:]]*$' "$CMAKE_FILE"; then
    echo "  - CUDA already present in project(...) LANGUAGES"
else
    TMP_FILE=$(mktemp)

    awk '
    BEGIN {
        in_project = 0
        patched = 0
    }
    {
        line = $0

        # Enter project(lammps ...) block
        if (!in_project && line ~ /^[[:space:]]*project[[:space:]]*\([[:space:]]*lammps([[:space:]]|$)/) {
            in_project = 1
        }

        if (in_project && line ~ /^[[:space:]]*LANGUAGES[[:space:]]+/) {
            # If CUDA not already in this LANGUAGES line, insert before final ")"
            if (line !~ /(^|[[:space:]])CUDA([[:space:]]|\))/) {
                sub(/\)[[:space:]]*$/, " CUDA)", line)
                patched = 1
            }
        }

        print line

        # Leave project block when we hit a line ending with ")"
        if (in_project && line ~ /\)[[:space:]]*$/) {
            in_project = 0
        }
    }
    END {
        if (!patched) exit 2
    }' "$CMAKE_FILE" > "$TMP_FILE"

    status=$?
    if [ $status -eq 0 ]; then
        replace_file "$CMAKE_FILE" "$TMP_FILE"
        echo "  - Added CUDA to project(... LANGUAGES ...)"
    else
        rm -f "$TMP_FILE"
        echo "Error: Could not find LANGUAGES line inside project(lammps ...)"
        echo "Please check whether this CMakeLists.txt matches the expected LAMMPS layout."
        exit 1
    fi
fi

###############################################################################
# 2) Add NEP_KK implementation block after accelerator include block
###############################################################################
echo "Adding NEP_KK build block..."

if grep -Eq '^[[:space:]]*if[[:space:]]*\([[:space:]]*PKG_NEP_KK[[:space:]]*\)[[:space:]]*$' "$CMAKE_FILE"; then
    echo "  - NEP_KK block already exists"
else
    TMP_FILE=$(mktemp)

    awk '
    BEGIN { inserted = 0; in_block = 0 }
    {
        print

        if ($0 ~ /^[[:space:]]*foreach[[:space:]]*\([[:space:]]*PKG_WITH_INCL[[:space:]]+CORESHELL/) {
            in_block = 1
            next
        }

        if (in_block && $0 ~ /^[[:space:]]*endforeach[[:space:]]*\([[:space:]]*\)[[:space:]]*$/) {
            print ""
            print "######################################################################"
            print "# package of NEP with KOKKOS"
            print "######################################################################"
            print "if(PKG_NEP_KK)"
            print "  if(NOT PKG_KOKKOS)"
            print "    message(FATAL_ERROR \"NEP_KK requires KOKKOS package. Enable with -DPKG_KOKKOS=yes\")"
            print "  endif()"
            print ""
            print "  if(NOT Kokkos_ENABLE_CUDA)"
            print "    message(FATAL_ERROR \"NEP_KK requires CUDA support. Enable with -DKokkos_ENABLE_CUDA=yes\")"
            print "  endif()"
            print ""
            print "  message(STATUS \"NEP_KK: Building with mandatory KOKKOS and CUDA\")"
            print ""
            print "  file(GLOB NEP_KK_SOURCES CONFIGURE_DEPENDS"
            print "    ${LAMMPS_SOURCE_DIR}/nep_gpu/force/*.cu"
            print "    ${LAMMPS_SOURCE_DIR}/nep_gpu/utilities/*.cu"
            print "  )"
            print ""
            print "  file(GLOB NEP_KOKKOS_SOURCES CONFIGURE_DEPENDS"
            print "    ${LAMMPS_SOURCE_DIR}/KOKKOS/pair_nep_kokkos.cpp"
            print "  )"
            print ""
            print "  target_sources(lammps PRIVATE ${NEP_KOKKOS_SOURCES} ${NEP_KK_SOURCES})"
            print ""
            print "  target_include_directories(lammps PRIVATE"
            print "    ${LAMMPS_SOURCE_DIR}/KOKKOS"
            print "    ${LAMMPS_SOURCE_DIR}/nep_gpu/force"
            print "    ${LAMMPS_SOURCE_DIR}/nep_gpu/utilities"
            print "  )"
            print "endif()"
            inserted = 1
            in_block = 0
        }
    }
    END {
        if (!inserted) exit 2
    }' "$CMAKE_FILE" > "$TMP_FILE"

    status=$?
    if [ $status -eq 0 ]; then
        replace_file "$CMAKE_FILE" "$TMP_FILE"
        echo "  - Added NEP_KK build block"
    else
        rm -f "$TMP_FILE"
        echo "Error: Could not find accelerator package include block for NEP_KK insertion"
        exit 1
    fi
fi

echo "Patch completed successfully."
echo ""
echo "Typical configure options:"
echo "  -D PKG_KOKKOS=yes -D Kokkos_ENABLE_CUDA=yes -D PKG_NEP_KK=yes"
echo ""
echo ""


###############################################################################
# Copy source files
###############################################################################
echo "Starting patch process for: $LAMMPSROOT"
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
    cp -rf "nep_gpu" "$LAMMPSROOT/src/"
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
