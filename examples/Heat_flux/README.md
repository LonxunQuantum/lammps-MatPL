# Heat Flux Example

This example calculates the thermal conductivity of graphene using the MatPL-NEP potential in LAMMPS, based on the non-equilibrium molecular dynamics (NEMD) method.

## Files

| File | Description |
| ------ | ------ |
| `graphene.in` | LAMMPS input script for heat flux simulation |
| `graphene.data` | Graphene structure data file |
| `nep.txt` | NEP potential file |
| `job.sh` | SLURM job submission script |
| `compute_Energy_Temp.ref` | FP64 reference output: energy and temperature from thermostats |
| `compute_HeatFlux.ref` | FP64 reference output: heat flux from atomic computations |
| `calc_heatflux.py` | Visualization script, generates `HeatFlux.png` |
| `compare_heatflux_paths.py` | Compare legacy and MatPL heat flux outputs from a dual-path run |
| `check_test.py` | Automated test validation script (MIX vs FP64) |
| `clean_outputs.sh` | Remove generated `.out`, `.lammpstrj`, and `log.*` files |

## Method

The simulation applies Langevin thermostats at two ends of the graphene sheet (350 K and 250 K) to establish a temperature gradient. The accumulated heat is computed in two independent ways:

1. **From atoms**: integrating the heat flux computed from atomic stress tensors
2. **From thermostats**: tracking energy added/removed by the hot/cold thermostats

If the two curves agree, the heat flux calculation is validated.

## Running the Simulation

```bash
# Submit via SLURM
sbatch job.sh

# Or run directly (adjust GPU/MPI settings as needed)
mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos -in graphene.in
```

The example exposes two runtime switches:

- `heatflux_mode`: choose which atom-derived heat-flux implementation to run
- `thermo_flux_mode`: choose whether to also write the thermostat-derived heat-flow file `compute_Energy_Temp.out`

## Switching Heat Flux Paths

The input file defines `variable heatflux_mode index 1` and `variable thermo_flux_mode index 1` near the top of `graphene.in`:

- `0`: legacy LAMMPS path using `ke/atom + pe/atom + centroid/stress/atom + heat/flux`
- `1`: new MatPL direct path using `compute matpl/heatflux/kk`
- `2`: run both paths together and write two files for comparison

For `thermo_flux_mode`:

- `0`: do not write `compute_Energy_Temp.out`
- `1`: write `compute_Energy_Temp.out` so `calc_heatflux.py` can overlay the thermostat-derived curve

Typical runs:

```bash
# New MatPL path only
mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos -var heatflux_mode 1 -var thermo_flux_mode 1 -in graphene.in

# Legacy path only
mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos -var heatflux_mode 0 -var thermo_flux_mode 1 -in graphene.in

# Dual-path correctness check
mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos -var heatflux_mode 2 -var thermo_flux_mode 1 -in graphene.in

# Atom-derived path only, without thermostat output
mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos -var heatflux_mode 1 -var thermo_flux_mode 0 -in graphene.in
```

When `heatflux_mode = 2`, the script writes:

- `compute_HeatFlux_legacy.out`
- `compute_HeatFlux_matpl.out`

Use this mode only for correctness checks. It still computes the old host-side path, so it is not a clean performance benchmark.

The supplied [examples/Heat_flux/job.sh](examples/Heat_flux/job.sh) forwards the shell variables `HEATFLUX_MODE` and `THERMO_FLUX_MODE` to these two LAMMPS variables. For example:

```bash
HEATFLUX_MODE=2 THERMO_FLUX_MODE=1 sbatch job.sh
```

## Validating Results

To compare the new path against the old path from the same run:

```bash
python3 compare_heatflux_paths.py
```

Or specify files explicitly:

```bash
python3 compare_heatflux_paths.py compute_HeatFlux_legacy.out compute_HeatFlux_matpl.out
```

The script compares the accumulated heat-flux curve over the default `[0, 1.5] ns` window and reports max error and `R-squared`.

For `compute_HeatFlux.out`, `compute_HeatFlux.ref`, `compute_HeatFlux_legacy.out`, and `compute_HeatFlux_matpl.out`, the six reported heat-flux columns follow the same ordering:

- columns 1-3: total heat flux in `x/y/z`
- columns 4-6: convective heat-flux contribution in `x/y/z`

So the virial contribution is the difference between these two blocks, i.e. `(1-4, 2-5, 3-6)`.

This overlay is only a qualitative consistency check. If MatPL is built without `-DPREC_NEPINFER=ON`, the atom-derived and thermostat-derived curves from `calc_heatflux.py` often do not line up tightly, and the thermostat-derived curve usually deviates more from the FP64 reference than the atom-derived MatPL heat-flux output. In that build mode, use the atom-path output for regression checks and do not treat the thermostat overlay as a strict pass/fail criterion.

The test validates MIX-precision results against the FP64 reference stored in `compute_HeatFlux.ref`. Only `compute_HeatFlux.out` is compared — the thermostat-energy output is expected to have some deviation under MIX precision and is not checked.

After the MIX-precision simulation finishes, run:

```bash
python3 check_test.py
```

This compares:

- `./compute_HeatFlux.out`
- `./compute_HeatFlux.ref`

If needed, you can still specify an explicit directory or reference file:

```bash
python3 check_test.py <test_dir> [ref_file]
```

Example:

```bash
# Compare the current run output against the committed FP64 reference in the same directory
python3 check_test.py .

# Or compare a run directory against an explicit reference file
python3 check_test.py ./mix_run ./compute_HeatFlux.ref
```

The script reports PASS/FAIL based on:

- Relative max error of accumulated heat flux < 5%
- R-squared > 0.99

To generate the comparison plot:

```bash
python3 calc_heatflux.py <path>
```

If `thermo_flux_mode = 1`, this produces `HeatFlux.png` showing both the atom-derived and thermostat-derived curves overlaid. If `thermo_flux_mode = 0`, the script plots only the atom-derived curve.

## Cleaning Generated Files

To remove generated run outputs while keeping the committed `.ref` reference files, run:

```bash
bash clean_outputs.sh
```

## Expected Output

A successful test shows that the MIX-precision accumulated heat flux curve closely matches the FP64 reference, confirming numerical consistency of the MatPL-NEP potential under mixed precision.

![HeatFlux](HeatFlux.png)
