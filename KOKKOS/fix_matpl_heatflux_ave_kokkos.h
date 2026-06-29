/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(matpl/heatflux/ave/kk,FixMatPLHeatFluxAveKokkos<LMPDeviceType>);
FixStyle(matpl/heatflux/ave/kk/device,FixMatPLHeatFluxAveKokkos<LMPDeviceType>);
FixStyle(matpl/heatflux/ave/kk/host,FixMatPLHeatFluxAveKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_MATPL_HEATFLUX_AVE_KOKKOS_H
#define LMP_FIX_MATPL_HEATFLUX_AVE_KOKKOS_H

#include "fix.h"
#include "kokkos_type.h"

#include <string>

namespace LAMMPS_NS {

template<class DeviceType>
class FixMatPLHeatFluxAveKokkos : public Fix {
 public:
  typedef DeviceType device_type;

  FixMatPLHeatFluxAveKokkos(class LAMMPS *, int, char **);
  ~FixMatPLHeatFluxAveKokkos() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void end_of_step() override;

 private:
  class Compute *heatflux_compute;
   std::string heatflux_compute_id;
  bigint nvalid;
  int nrepeat;
  int nfreq;
  int irepeat;
   int append_mode;
  double sample_sum[6];
  FILE *fp;
};

}    // namespace LAMMPS_NS

#endif
#endif