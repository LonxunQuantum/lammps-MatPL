// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_matpl_heatflux_ave_kokkos.h"

#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "modify.h"
#include "update.h"

#include <cstdio>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

namespace {

bigint nextvalid_heatflux(bigint ntimestep, const int nevery, const int nrepeat,
                          const int nfreq)
{
  bigint nvalid = (ntimestep / nfreq) * nfreq + nfreq;
  if (nvalid - nfreq == ntimestep && nrepeat == 1)
    nvalid = ntimestep;
  else
    nvalid -= static_cast<bigint>(nrepeat - 1) * nevery;
  if (nvalid < ntimestep) nvalid += nfreq;
  return nvalid;
}

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixMatPLHeatFluxAveKokkos<DeviceType>::FixMatPLHeatFluxAveKokkos(LAMMPS *lmp, int narg,
                                                                 char **arg) :
  Fix(lmp, narg, arg), heatflux_compute(nullptr), nvalid(0), nrepeat(0), nfreq(0),
  irepeat(0), append_mode(0), fp(nullptr)
{
  if (narg != 9)
    error->all(FLERR, "Illegal fix matpl/heatflux/ave/kk command");

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  nrepeat = utils::inumeric(FLERR, arg[4], false, lmp);
  nfreq = utils::inumeric(FLERR, arg[5], false, lmp);

  if (nevery <= 0) error->all(FLERR, 3, "Illegal fix matpl/heatflux/ave/kk nevery value: {}", nevery);
  if (nrepeat <= 0) error->all(FLERR, 4, "Illegal fix matpl/heatflux/ave/kk nrepeat value: {}", nrepeat);
  if (nfreq <= 0) error->all(FLERR, 5, "Illegal fix matpl/heatflux/ave/kk nfreq value: {}", nfreq);
  if (nfreq % nevery || nrepeat * nevery > nfreq)
    error->all(FLERR, "Inconsistent fix matpl/heatflux/ave/kk nevery/nrepeat/nfreq values");

  heatflux_compute_id = arg[6];
  heatflux_compute = modify->get_compute_by_id(heatflux_compute_id);
  if (!heatflux_compute)
    error->all(FLERR, "Compute ID {} for fix matpl/heatflux/ave/kk does not exist", heatflux_compute_id);

  if ((strcmp(arg[7], "file") != 0) && (strcmp(arg[7], "append") != 0))
    error->all(FLERR, "Expected 'file' or 'append' keyword in fix matpl/heatflux/ave/kk command");

  append_mode = (strcmp(arg[7], "append") == 0);
  if (comm->me == 0) {
    fp = fopen(arg[8], append_mode ? "a" : "w");
    if (fp == nullptr)
      error->one(FLERR, "Cannot open fix matpl/heatflux/ave/kk file {}: {}", arg[8],
                 utils::getsyserror());
    if (!append_mode) {
      fprintf(fp, "# Time-averaged data for fix %s\n", id);
      fprintf(fp, "# TimeStep c_%s[1] c_%s[2] c_%s[3] c_%s[4] c_%s[5] c_%s[6]\n",
              heatflux_compute->id, heatflux_compute->id, heatflux_compute->id,
              heatflux_compute->id, heatflux_compute->id, heatflux_compute->id);
      fflush(fp);
    }
  }

  kokkosable = 1;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
  time_depend = 1;

  for (double &value : sample_sum) value = 0.0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixMatPLHeatFluxAveKokkos<DeviceType>::~FixMatPLHeatFluxAveKokkos()
{
  if (fp && comm->me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixMatPLHeatFluxAveKokkos<DeviceType>::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixMatPLHeatFluxAveKokkos<DeviceType>::init()
{
  heatflux_compute = modify->get_compute_by_id(heatflux_compute_id);
  if (!heatflux_compute)
    error->all(FLERR, Error::NOLASTLINE,
               "Compute {} for fix matpl/heatflux/ave/kk no longer exists",
               heatflux_compute_id);

  if (!heatflux_compute->vector_flag || heatflux_compute->size_vector < 6)
    error->all(FLERR, "Fix matpl/heatflux/ave/kk requires a compute that produces at least 6 vector values");

  irepeat = 0;
  for (double &value : sample_sum) value = 0.0;

  nvalid = nextvalid_heatflux(update->ntimestep, nevery, nrepeat, nfreq);

  modify->addstep_compute(nvalid);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixMatPLHeatFluxAveKokkos<DeviceType>::setup(int /*vflag*/)
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixMatPLHeatFluxAveKokkos<DeviceType>::end_of_step()
{
  const bigint ntimestep = update->ntimestep;
  if (ntimestep != nvalid) return;

  modify->clearstep_compute();

  if (!(heatflux_compute->invoked_flag & Compute::INVOKED_VECTOR)) {
    heatflux_compute->compute_vector();
    heatflux_compute->invoked_flag |= Compute::INVOKED_VECTOR;
  }

  for (int i = 0; i < 6; ++i)
    sample_sum[i] += heatflux_compute->vector[i];

  ++irepeat;
  if (irepeat < nrepeat) {
    nvalid += nevery;
    modify->addstep_compute(nvalid);
    return;
  }

  double averaged[6];
  for (int i = 0; i < 6; ++i) {
    averaged[i] = sample_sum[i] / static_cast<double>(nrepeat);
    sample_sum[i] = 0.0;
  }
  irepeat = 0;

  nvalid = ntimestep + nfreq - static_cast<bigint>(nrepeat - 1) * nevery;
  modify->addstep_compute(nvalid);

  if (fp && comm->me == 0) {
    fprintf(fp, "%lld %g %g %g %g %g %g\n",
            static_cast<long long>(ntimestep), averaged[0], averaged[1], averaged[2],
            averaged[3], averaged[4], averaged[5]);
    fflush(fp);
    if (ferror(fp))
      error->one(FLERR, Error::NOLASTLINE,
                 "Error writing fix matpl/heatflux/ave/kk output: {}",
                 utils::getsyserror());
  }
}

namespace LAMMPS_NS {
template class FixMatPLHeatFluxAveKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixMatPLHeatFluxAveKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS