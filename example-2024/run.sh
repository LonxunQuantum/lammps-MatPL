#!/bin/bash

# Run the simulation
mpirun -n 2 lmp -k on g 4 -sf kk -pk kokkos newton on neigh half gpu/aware off -in HfO2.in