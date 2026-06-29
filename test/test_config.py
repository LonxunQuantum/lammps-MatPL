SYSTEMS = ["HfO2", "water", "C", "Cu", "W", "WMoTaV", "U"]
ENSEMBLES = ["NPT", "NVT"]

LAMMPS_CMD = [
    "mpirun",
    "-np", "1",
    "--bind-to", "numa",
    "lmp",
    "-k", "on", "g", "1",
    "-sf", "kk",
    "-pk", "kokkos", "neigh", "half", "newton", "on",
    "-in", "lmp.in",
]
