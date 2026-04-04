#!/bin/bash

# Run the simulation
mpirun -n 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half gpu/aware off -in HfO2-gpu.in

mpirun -n 20 lmp -in HfO2-cpu.in