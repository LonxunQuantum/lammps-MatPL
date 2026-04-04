#!/bin/bash

# Run the simulation
mpirun -n 2 /tools/yazhuoSoftware/NEP-lammps/lammps2026/build/lmp -k on g 4 -sf kk -pk kokkos newton on neigh half gpu/aware off -in HfO2.in