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
/* The nep KOKKOS interface enables neighboring build operations to be moved to the device side, accelerating the overall speed of MD. Author wuxingxing@pwmat.com and MatPL development team. 2026. Beijing Lonxun Quantum Co.,Ltd.*/

#ifdef PAIR_CLASS
// clang-format off
PairStyle(matpl/nep/kk, PairNEPKokkos<LMPDeviceType>);
PairStyle(matpl/nep/kk/device, PairNEPKokkos<LMPDeviceType>);
PairStyle(matpl/nep/kk/host, PairNEPKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_NEP_KOKKOS_H
#define LMP_PAIR_NEP_KOKKOS_H

#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"
#include "nepkk.cuh"
#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace LAMMPS_NS {

template<class DeviceType>
class PairNEPKokkos : public Pair {
 public:
  enum {EnabledNeighFlags=HALF|HALFTHREAD|FULL};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairNEPKokkos(class LAMMPS *);
  ~PairNEPKokkos() override;
  
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  void allocate();
  int find_atomic_number(std::string& key);
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  std::tuple<double, double, double, double, double, double> calc_max_error(
      const int* ilist, const double* h_full_f, const double* h_full_e, int num_ff, int inum,
      int nlocal, int nall, int rank);
 protected:

  int me;
  int num_ff;
  bool is_rank_0;
  int nprocs_total;
  int global_nall = 0;
  int global_nlocal = 0;
  bool reverse_force = false;
  std::vector<NEPKK> nep_gpu_models;  // NEP model instance
  std::vector<std::string> potential_files;

  std::string explrError_fname = "explr.error";
  std::FILE *explrError_fp = nullptr;
  int out_freq = 1;
  double cutoff, cutoffsq;
  
  // Kokkos data
  int inum, nlocal, nall;
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_double_1d k_full_f;  // flattened [num_ff][nall][3]
  HAT::t_double_1d h_full_f;
  DAT::tdual_double_1d k_full_e;  // flattened [num_ff][nlocal]
  HAT::t_double_1d h_full_e;
  
  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;
  
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  typename AT::t_neighbors_2d d_neighbors;
  
  int need_dup;
  int neighflag, newton_pair;
  int eflag, vflag;
  int rank, device_id;
  std::vector<double> h_etot_virial_global;
  // NEP data structures - Kokkos Views for raw pointers

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout> dup_f;
  DupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout> dup_eatom;
  DupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> dup_vatom;

  NonDupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout> ndup_f;
  NonDupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout> ndup_eatom;
  NonDupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> ndup_vatom;

  friend void pair_virial_fdotr_compute<PairNEPKokkos>(PairNEPKokkos*);
};

}

#endif
#endif
