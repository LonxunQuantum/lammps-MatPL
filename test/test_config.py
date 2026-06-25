SYSTEMS = ["HfO2", "water", "C", "Cu", "W", "WMoTaV", "U"]
ENSEMBLES = ["NPT", "NVT"]

LAMMPS_CMD = [
    "mpirun",
    "-np", "1",
    "--bind-to", "numa",
    "lmp_mpi",
    "-k", "on", "g", "1",
    "-sf", "kk",
    "-pk", "kokkos",
    "-in", "lmp.in",
]
