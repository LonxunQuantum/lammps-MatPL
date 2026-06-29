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

#include "compute_matpl_heatflux_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory.h"
#include "pair_nep_kokkos.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputeMatPLHeatFluxKokkos<DeviceType>::ComputeMatPLHeatFluxKokkos(LAMMPS *lmp, int narg,
                                                                   char **arg) :
  Compute(lmp, narg, arg), atomKK(nullptr), mvv2e(0.0)
{
  if (narg != 3) error->all(FLERR, "Illegal compute matpl/heatflux/kk command");

  vector_flag = 1;
  size_vector = 6;
  extvector = 1;
  timeflag = 1;
  peatomflag = 0;
  pressatomflag = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = V_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK;
  datamask_modify = EMPTY_MASK;

  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputeMatPLHeatFluxKokkos<DeviceType>::~ComputeMatPLHeatFluxKokkos()
{
  if (copymode) return;
  delete[] vector;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeMatPLHeatFluxKokkos<DeviceType>::init()
{
  auto *pair = dynamic_cast<PairNEPKokkos<DeviceType> *>(force->pair);
  if (!pair)
    error->all(FLERR,
               "Compute matpl/heatflux/kk requires pair_style matpl/nep/kk with matching Kokkos execution space");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeMatPLHeatFluxKokkos<DeviceType>::compute_vector()
{
  auto *pair = dynamic_cast<PairNEPKokkos<DeviceType> *>(force->pair);
  if (!pair)
    error->all(FLERR,
               "Compute matpl/heatflux/kk requires pair_style matpl/nep/kk with matching Kokkos execution space");

  if (!force->pair->eatom || !force->pair->cvatom)
    error->all(FLERR,
               "Compute matpl/heatflux/kk could not access pair per-atom energy or centroid virial data");

  if (!lmp->kokkos || lmp->kokkos->auto_sync) {
    atomKK->sync(execution_space, datamask_read);
  }

  invoked_vector = update->ntimestep;

  mvv2e = force->mvv2e;

  int nlocal = atom->nlocal;
  v = atomKK->k_v.view<DeviceType>();
  if (atomKK->rmass)
    rmass = atomKK->k_rmass.view<DeviceType>();
  else {
    atomKK->k_mass.sync<DeviceType>();
    mass = atomKK->k_mass.view<DeviceType>();
  }
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  pair->update_heatflux_reverse();
  heatflux = pair->heatflux_reverse_view();

  HF tally;
  copymode = 1;
  if (atomKK->rmass)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeMatPLHeatFluxVector<1> >(0, nlocal), *this, tally);
  else
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeMatPLHeatFluxVector<0> >(0, nlocal), *this, tally);
  copymode = 0;

  double data[6] = {
    tally.jc0 + tally.jv0,
    tally.jc1 + tally.jv1,
    tally.jc2 + tally.jv2,
    tally.jc0,
    tally.jc1,
    tally.jc2,
  };
  MPI_Allreduce(data, vector, 6, MPI_DOUBLE, MPI_SUM, world);
}

template<class DeviceType>
template<int RMASS>
KOKKOS_INLINE_FUNCTION
void ComputeMatPLHeatFluxKokkos<DeviceType>::operator()(TagComputeMatPLHeatFluxVector<RMASS>,
                                                        const int &i, HF &tally) const
{
  if (!(mask[i] & groupbit)) return;

  double massone = 0.0;
  if (RMASS)
    massone = rmass[i];
  else
    massone = mass[type[i]];

  const double vx = v(i, 0);
  const double vy = v(i, 1);
  const double vz = v(i, 2);
  const double ke = 0.5 * mvv2e * massone * (vx * vx + vy * vy + vz * vz);
  const int base = i * 10;
  const double eng = static_cast<double>(heatflux(base + 0)) + ke;

  tally.jc0 += eng * vx;
  tally.jc1 += eng * vy;
  tally.jc2 += eng * vz;

  tally.jv0 += static_cast<double>(heatflux(base + 1)) * vx + static_cast<double>(heatflux(base + 4)) * vy + static_cast<double>(heatflux(base + 5)) * vz;
  tally.jv1 += static_cast<double>(heatflux(base + 7)) * vx + static_cast<double>(heatflux(base + 2)) * vy + static_cast<double>(heatflux(base + 6)) * vz;
  tally.jv2 += static_cast<double>(heatflux(base + 8)) * vx + static_cast<double>(heatflux(base + 9)) * vy + static_cast<double>(heatflux(base + 3)) * vz;
}

namespace LAMMPS_NS {
template class ComputeMatPLHeatFluxKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeMatPLHeatFluxKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS