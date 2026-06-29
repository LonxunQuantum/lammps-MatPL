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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(matpl/heatflux/kk,ComputeMatPLHeatFluxKokkos<LMPDeviceType>);
ComputeStyle(matpl/heatflux/kk/device,ComputeMatPLHeatFluxKokkos<LMPDeviceType>);
ComputeStyle(matpl/heatflux/kk/host,ComputeMatPLHeatFluxKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_MATPL_HEATFLUX_KOKKOS_H
#define LMP_COMPUTE_MATPL_HEATFLUX_KOKKOS_H

#include "compute.h"
#include "kokkos_type.h"

#include <utility>

namespace LAMMPS_NS {

template<int RMASS>
struct TagComputeMatPLHeatFluxVector {};

template<class DeviceType>
class ComputeMatPLHeatFluxKokkos : public Compute {
 public:
   struct s_hf {
      double jc0, jc1, jc2;
      double jv0, jv1, jv2;

      KOKKOS_INLINE_FUNCTION
      s_hf() : jc0(0.0), jc1(0.0), jc2(0.0), jv0(0.0), jv1(0.0), jv2(0.0) {}

      KOKKOS_INLINE_FUNCTION
      s_hf &operator+=(const s_hf &rhs)
      {
         jc0 += rhs.jc0;
         jc1 += rhs.jc1;
         jc2 += rhs.jc2;
         jv0 += rhs.jv0;
         jv1 += rhs.jv1;
         jv2 += rhs.jv2;
         return *this;
      }
   };

  typedef DeviceType device_type;
   typedef s_hf HF;
   typedef HF value_type;
   typedef ArrayTypes<DeviceType> AT;
   using heatflux_view_type = decltype(std::declval<DAT::tdual_xfloat_1d>().template view<DeviceType>());

  ComputeMatPLHeatFluxKokkos(class LAMMPS *, int, char **);
  ~ComputeMatPLHeatFluxKokkos() override;

  void init() override;
  void compute_vector() override;

   KOKKOS_INLINE_FUNCTION
   void init(HF &value) const
   {
      value = HF();
   }

   KOKKOS_INLINE_FUNCTION
   void join(HF &dst, const HF &src) const
   {
      dst += src;
   }

   template<int RMASS>
   KOKKOS_INLINE_FUNCTION
   void operator()(TagComputeMatPLHeatFluxVector<RMASS>, const int &, HF &) const;

 protected:
  class AtomKokkos *atomKK;
   typename AT::t_v_array_randomread v;
   typename AT::t_float_1d_randomread rmass;
   typename AT::t_float_1d_randomread mass;
   typename AT::t_int_1d_randomread type;
   typename AT::t_int_1d_randomread mask;
   heatflux_view_type heatflux;
  double mvv2e;
};

}    // namespace LAMMPS_NS

#endif
#endif