#!/bin/bash

# Compared to install.sh, 
# this script only copies the custom pair files for d3 and nep to 
# the corresponding LAMMPS directory, 
# without initiating the subsequent compilation steps

# Check if lammps_root_dir is provided
if [ -z "$1" ]; then
    echo "Error: Please provide the LAMMPS source root directory as the first argument"
    echo "Usage: $0 <lammps_root_dir> [install_dir]"
    exit 1
fi

# Set variables
LAMMPS_ROOT_DIR="$1"
INSTALL_DIR="${2:-$LAMMPS_ROOT_DIR/build}"  # Default install path is lammps_root_dir/build
BUILD_DIR="$LAMMPS_ROOT_DIR/build"
CMAKE_DIR="$LAMMPS_ROOT_DIR/cmake"
PRESETS_DIR="$CMAKE_DIR/presets"
MATPL_DIR="./MATPL"

# Check if LAMMPS root directory exists
if [ ! -d "$LAMMPS_ROOT_DIR" ]; then
    echo "Error: LAMMPS root directory $LAMMPS_ROOT_DIR does not exist"
    exit 1
fi

# Step 1: Copy files, backing up existing CMakeLists.txt and basic.cmake
echo "Copying CMakeLists.txt and basic.cmake..."
# Check and backup CMakeLists.txt
if [ -f "$CMAKE_DIR/CMakeLists.txt" ]; then
    echo "Backing up $CMAKE_DIR/CMakeLists.txt to CMakeLists.txt.bk"
    mv "$CMAKE_DIR/CMakeLists.txt" "$CMAKE_DIR/CMakeLists.txt.bk"
fi
cp ./CMakeLists.txt "$CMAKE_DIR/"

# Check and backup basic.cmake
if [ -f "$PRESETS_DIR/basic.cmake" ]; then
    echo "Backing up $PRESETS_DIR/basic.cmake to basic.cmake.bk"
    mv "$PRESETS_DIR/basic.cmake" "$PRESETS_DIR/basic.cmake.bk"
fi
cp ./basic.cmake "$PRESETS_DIR/"

# Copy other files
echo "Copying pair_d3.* and MATPL files..."
cp ./pair_d3.cu ./pair_d3.h ./pair_d3_pars.h "$LAMMPS_ROOT_DIR/src/"
cp "$MATPL_DIR"/{dftd3para.h,nep_cpu.cpp,nep_cpu.h,pair_matpl.cpp,pair_matpl.h} "$LAMMPS_ROOT_DIR/src/"
# Recursively copy NEP_GPU directory
if [ -d "$MATPL_DIR/NEP_GPU" ]; then
    echo "Copying NEP_GPU directory to $LAMMPS_ROOT_DIR/src/NEP_GPU..."
    cp -r "$MATPL_DIR/NEP_GPU" "$LAMMPS_ROOT_DIR/src/"
else
    echo "Error: NEP_GPU directory not found in $MATPL_DIR"
    exit 1
fi

# Step 2: Create build directory
echo "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

