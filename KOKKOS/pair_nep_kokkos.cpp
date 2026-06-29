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
/* The nep KOKKOS interface enables neighboring build operations to be moved to the device side, accelerating the overall speed of MD. Author wuxingxing@pwmat.com and MatPL development team. 2026. Beijing Lonxun Quantum Co.,Ltd.*/
#include "pair_nep_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "modify.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "suffix.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairNEPKokkos<DeviceType>::PairNEPKokkos(LAMMPS *lmp) : PairNEP(lmp)
{
  centroidstressflag = CENTROID_AVAIL;
  local_maxeatom = 0;
  local_maxvatom = 0;
  local_maxcvatom = 0;
  respa_enable = 0;

  restartinfo = 0;
  manybody_flag = 1;
  single_enable = 0;
  one_coeff = 1;

  suffix_flag |= Suffix::KOKKOS;
  kokkosable = 1;
  reverse_comm_device = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  me = comm->me;
  h_etot_virial_global.resize(10);
  // 这种方式可以直接输出到log.lammps中
  // if (comm->me == 0) {
  //   utils::logmesg(lmp, "=== PairNEPKokkos Constructor Finished ===\n");
  // }
}

namespace {
constexpr int HEATFLUX_REVERSE_WIDTH = 10;

bool is_matpl_heatflux_style(const char *style)
{
  return strcmp(style, "matpl/heatflux/kk") == 0 || strcmp(style, "matpl/heatflux/kk/device") == 0 ||
         strcmp(style, "matpl/heatflux/kk/host") == 0;
}
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairNEPKokkos<DeviceType>::~PairNEPKokkos()
{
  if (copymode) return;

  if (allocated) {
    if (eatom)  memoryKK->destroy_kokkos(k_eatom, eatom);
    if (vatom)  memoryKK->destroy_kokkos(k_vatom, vatom);
    if (cvatom) memoryKK->destroy_kokkos(k_cvatom, cvatom);
  }
  memory->destroy(setflag);
  memory->destroy(cutsq);
  h_etot_virial_global = decltype(h_etot_virial_global)();
  if (is_rank_0 && explrError_fp != nullptr) {
        fclose(explrError_fp);
        explrError_fp = nullptr;
  }

  // NEP model cleanup if needed
  // nep_gpu_model.cleanup(); // Assuming NEP has a cleanup method
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNEPKokkos<DeviceType>::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n+1, n+1, "pair:setflag");
  memory->create(cutsq, n+1, n+1, "pair:cutsq");

  // for (int i = 1; i <= n; i++) {
  //   for (int j = i; j <= n; j++) {
  //     setflag[i][j] = 0;
  //   }
  // }
}
/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNEPKokkos<DeviceType>::settings(int narg, char **arg)
{
  is_rank_0 = (comm->me == 0);

  if (narg < 1) error->all(FLERR, "Illegal pair_style command");
  int iarg = 0;  // index of first forcefield file
  // Parse potential file
  num_ff = 0;
  while(iarg < narg) { // nep1.txt nep2.txt ... nepn.txt out_freq 10
    std::string arg_str(arg[iarg]);
    if (arg_str.find(".txt") != std::string::npos) {
      potential_files.push_back(arg_str);
      num_ff ++;
      iarg++;
    } else {
      break; // for out_freq
    }
  }
  while (iarg < narg) {
    if (strcmp(arg[iarg], "out_freq") == 0) {
        out_freq = utils::inumeric(FLERR, arg[++iarg], false, lmp);
    } else if (strcmp(arg[iarg], "out_file") == 0) {
        explrError_fname = arg[++iarg];
    } 
    iarg++;    
  }

  if (is_rank_0 and num_ff > 1) {
      explrError_fp = fopen(explrError_fname.c_str(), "w");
      if (!explrError_fp)
        error->one(FLERR, "Cannot open explorer error file: " + explrError_fname);
      fprintf(explrError_fp, "%9s %16s %16s %16s %16s %16s %16s\n",
      "#    step", "avg_devi_f", "min_devi_f", "max_devi_f",
      "avg_devi_e", "min_devi_e", "max_devi_e");
      fflush(explrError_fp);
  }
    
  nprocs_total = comm->nprocs;
  // 移除硬编码device_id=0和手动cudaSetDevice
  // Kokkos已在LAMMPS init时设置当前设备（基于 -k on g Ng）
  // 如果NEP需要查询当前ID，可用：int device_id = Kokkos::device_id();
  // 但通常无需手动设置，避免覆盖Kokkos
  
  if (std::is_same<DeviceType, LMPDeviceType>::value) {
    // 可选：查询并日志当前设备（调试用）
    cudaGetDevice(&device_id);
    rank = comm->me;
    // if (is_rank_0) {
    // }
    // 无需 cudaSetDevice(device_id); // 已由Kokkos设置
  }
  // Resolve libnep_gpu.so path: NEP_GPU_LIB_PATH env var, then default name
  std::string lib_path = "libnep_gpu.so";
  const char* env_lib = std::getenv("NEP_GPU_LIB_PATH");
  if (env_lib && env_lib[0] != '\0') lib_path = env_lib;

  // Resolve license file path: NEP_LICENSE_PATH env var, then ~/.nep_license/license.json
  std::string license_path;
  const char* env_lic = std::getenv("NEP_LICENSE_PATH");
  if (env_lic && env_lic[0] != '\0') {
    license_path = env_lic;
  } else {
    const char* home = std::getenv("HOME");
    license_path = std::string(home ? home : ".") + "/.nep_license/license.json";
  }

  nep_gpu_models.resize(num_ff);
  for (int i=0; i < num_ff; i++) {
    if (!nep_gpu_models[i].load_library(lib_path.c_str())) {
      error->all(FLERR, "Failed to load NEP GPU library '" + lib_path +
                 "': " + nep_gpu_models[i].error() +
                 "\n  Set NEP_GPU_LIB_PATH to the correct path.");
    }
    if (nep_gpu_models[i].license_load(license_path.c_str()) != 0) {
      std::string detail = nep_gpu_models[i].get_error();
      error->all(FLERR, "NEP GPU license check failed (file: " + license_path +
                 ")\n  Reason: " + detail +
                 "\n  Set NEP_LICENSE_PATH to the correct license file path.");
    }
    if (is_rank_0 && i == 0) {
      utils::logmesg(lmp, "NEP GPU library loaded: " + lib_path + "\n");
      utils::logmesg(lmp, "NEP GPU license verified: " + license_path + "\n");
    }

    std::string model_file = potential_files[i];
    int read_rc = nep_gpu_models[i].read_potential(model_file.c_str(), is_rank_0, comm->me, device_id, i);
    if (read_rc != 0) {
      error->all(FLERR, "Failed to read NEP potential file: " + model_file +
                 " (error: " + std::string(nep_gpu_models[i].get_error()) + ")");
    }
    if (i == 0) {
      cutoff = nep_gpu_models[i].get_rc_radial();
      if (nep_gpu_models[i].get_zbl_enabled()) {
        double zbl_rc = nep_gpu_models[i].get_zbl_rc_outer();
        if (zbl_rc > cutoff) cutoff = zbl_rc;
      }
      cutoffsq = cutoff * cutoff;
    }
    if (is_rank_0) {
      utils::logmesg(lmp, "NEP Kokkos potential " + model_file + " loaded successfully\n");
      utils::logmesg(lmp, "  cutoff=" + std::to_string(cutoff) +
        " n_types=" + std::to_string(nep_gpu_models[i].get_num_types()) +
        " n_max_radial=" + std::to_string(nep_gpu_models[i].get_n_max_radial()) +
        " n_max_angular=" + std::to_string(nep_gpu_models[i].get_n_max_angular()) + "\n");
    }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNEPKokkos<DeviceType>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  for (int f1 = 0; f1 < num_ff; f1++) {
    std::vector<int> atom_type_module = nep_gpu_models[f1].get_element_list();
    std::vector<int> atom_types;
    for (int ii = 2; ii < narg; ++ii) {
      std::string element = utils::strdup(arg[ii]);  // LAMMPS提供的安全转换
      int temp = find_atomic_number(element);
      // int temp = std::stoi(arg[ii]);
      auto iter = std::find(atom_type_module.begin(), atom_type_module.end(), temp);
      if (iter != atom_type_module.end())
      {
          int index = std::distance(atom_type_module.begin(), iter);
          atom_types.push_back(index);
      }
      else
      {
          error->all(FLERR, "This element is not included in the nep file: " + potential_files[f1]);
      }
    }
    if (nep_gpu_models[f1].set_atom_type_map(narg-2, atom_types.data()) != 0) {
      error->all(FLERR, "Failed to set atom type map for potential: " + potential_files[f1]);
    }
  }
  int ntypes = atom->ntypes;
  // For NEP, all types are handled by the model; set all pairs
  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutoffsq;
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairNEPKokkos<DeviceType>::init_style()
{
  if (strstr(update->integrate_style,"respa"))
    error->all(FLERR,"Pair NEP does not support rRESPA");

  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);

  // if (is_rank_0) printf("======== in init_style: neighflag = %d =========\n", neighflag);
  // if (force->newton_pair == 0)
  //   error->all(FLERR,"Pair style matpl/nep/kk requires newton pair on");
  newton_pair = force->newton_pair;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairNEPKokkos<DeviceType>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  cutsq[i][j] = cutoffsq;
  return cutoff;
}

/* ----------------------------------------------------------------------
   find atomic number for element key
------------------------------------------------------------------------- */

template<class DeviceType>
int PairNEPKokkos<DeviceType>::find_atomic_number(std::string& key)
{
  // Implement element to atomic number mapping if needed
  // For now, assume types are directly mapped
  std::transform(key.begin(), key.end(), key.begin(), ::tolower);
  if (key.length() == 1) { key += " "; }
  key.resize(2);

std::vector<std::string> element_table = {
    "h ", "he", "li", "be", "b ", "c ", "n ", "o ", "f ", "ne",
    "na", "mg", "al", "si", "p ", "s ", "cl", "ar", "k ", "ca",
    "sc", "ti", "v ", "cr", "mn", "fe", "co", "ni", "cu", "zn",
    "ga", "ge", "as", "se", "br", "kr", "rb", "sr", "y ", "zr",
    "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn",
    "sb", "te", "i ", "xe", "cs", "ba", "la", "ce", "pr", "nd",
    "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb",
    "lu", "hf", "ta", "w ", "re", "os", "ir", "pt", "au", "hg",
    "tl", "pb", "bi", "po", "at", "rn", "fr", "ra", "ac", "th",
    "pa", "u ", "np", "pu",
    "am", "cm", "bk", "cf", "es", "fm", "md", "no", "lr",
    "rf", "db", "sg", "bh", "hs", "mt", "ds", "rg", "cn",
    "nh", "fl", "mc", "lv", "ts", "og"
};

std::vector<std::string> element_table2 = {
    "1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 ", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
    "51", "52", "53", "54", "55", "56", "57", "58", "59", "60",
    "61", "62", "63", "64", "65", "66", "67", "68", "69", "70",
    "71", "72", "73", "74", "75", "76", "77", "78", "79", "80",
    "81", "82", "83", "84", "85", "86", "87", "88", "89", "90",
    "91", "92", "93", "94",
    "95", "96", "97", "98", "99", "100", "101", "102", "103",
    "104", "105", "106", "107", "108", "109", "110", "111", "112",
    "113", "114", "115", "116", "117", "118"
};
  
  for (size_t i = 0; i < element_table.size(); ++i) {
      if (element_table[i] == key) {
          int atomic_number = i + 1;
          return atomic_number;
      }else if(element_table2[i] == key) {
          int atomic_number = i + 1;
          return atomic_number;
      }
  }

  // if not the case
  return -1;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
bool PairNEPKokkos<DeviceType>::need_active_matpl_heatflux_device(bigint ntimestep)
{
  for (int i = 0; i < modify->ncompute; ++i) {
    Compute *compute = modify->compute[i];
    if (!compute) continue;
    if (!is_matpl_heatflux_style(compute->style)) continue;
    if (compute->matchstep(ntimestep)) return true;
  }

  return false;
}

template<class DeviceType>
void PairNEPKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;
  no_virial_fdotr_compute = 1;
  ev_init(eflag, vflag, 0);
  // size_t total, used, free;
  bigint ntimestep = update->ntimestep;
  const bool neighbor_rebuilt = (neighbor && neighbor->ago == 0);
  bool is_devi_step = (num_ff > 1) && (ntimestep % out_freq == 0);
  const bool need_heatflux_device = need_active_matpl_heatflux_device(ntimestep);
  const int need_eatom_device = eflag_atom || need_heatflux_device;
  const int need_cvatom_device = cvflag_atom || need_heatflux_device;
  const int need_vflag_either_device = vflag_either || need_heatflux_device;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK); //同步原子数据到设备
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  type = atomKK->k_type.template view<DeviceType>();

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  inum = list->inum;

  cur_atom_max = max(cur_atom_max, atom->nmax);
  double *virial_per_atom_ptr = nullptr;
  double *cvirial_per_atom_ptr = nullptr;

  if (need_eatom_device) {
    if (maxeatom < cur_atom_max) maxeatom = cur_atom_max;
    if (eatom == nullptr || static_cast<int>(k_eatom.extent(0)) < maxeatom)
      memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }

  // 9-component centroid virial (independent from 6-component virial below)
  if (need_cvatom_device) {
    if (local_maxcvatom < cur_atom_max) local_maxcvatom = cur_atom_max;
    if (cvatom == nullptr || static_cast<int>(k_cvatom.extent(0)) < local_maxcvatom)
      memoryKK->create_kokkos(k_cvatom, cvatom, local_maxcvatom, "pair:cvatom");
    d_cvatom = k_cvatom.view<DeviceType>();
    Kokkos::deep_copy(d_cvatom, 0.0);
    cvirial_per_atom_ptr = d_cvatom.data();
  }
  // 6-component per-atom or global virial (mutually exclusive)
  if (vflag_atom) {
    if (local_maxvatom < cur_atom_max) local_maxvatom = cur_atom_max;
    if (vatom == nullptr || static_cast<int>(k_vatom.extent(0)) < local_maxvatom)
      memoryKK->create_kokkos(k_vatom, vatom, local_maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
    Kokkos::deep_copy(d_vatom, 0.0);
    virial_per_atom_ptr = d_vatom.data();
  } else if (vflag_global) {
    if (local_maxvatom_work < cur_atom_max) {
      local_maxvatom_work = cur_atom_max;
      k_vatom_work = DAT::tdual_virial_array("pair:vatom_work", local_maxvatom_work);
    }
    d_vatom_work = k_vatom_work.view<DeviceType>();
    Kokkos::deep_copy(d_vatom_work, 0.0);
    virial_per_atom_ptr = d_vatom_work.data();
  }
  
  if (num_ff > 1) {
    if (global_nall < nall) {
      global_nall = nall;
      k_full_f = DAT::tdual_ffloat_1d("pair:full_f", num_ff * global_nall * 3);
    } else {
      Kokkos::deep_copy(k_full_f.d_view, 0.0);
    }
    h_full_f = k_full_f.h_view;

    if (global_nlocal < nlocal) {
      global_nlocal = nlocal;
      k_full_e = DAT::tdual_ffloat_1d("pair:full_e", num_ff * global_nlocal);
    } else {
      Kokkos::deep_copy(k_full_e.d_view, 0.0);
    }
    h_full_e = k_full_e.h_view;
  }

  // const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  int max_neighs = d_neighbors.extent(1);
  int num_neighs = d_neighbors.extent(0);
  // (e,v,ee,eg,ea,ve,vg,va)=(1,2,1,1,0,0,0,0) 
  // compute pe_atom all pe/atom 开启后 maxeatom 才不为0，在dump步 eflag_atom 为2，其他步为0
  // (e,v,ee,eg,ea,ve,vg,va)=(3,2,3,1,2,0,0,0)

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_eatom);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_eatom);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }
  copymode = 1;
  // EV_FLOAT ev;
  // EV_FLOAT ev_all;

  // nep_gpu_model.getGPUMemoryStats(total, used, free);
  // if (nep_gpu_model.allocate_once==0 and is_rank_0) {
  //   int result = nep_gpu_model.calculateMaxAtoms(nep_gpu_model.paramb.MN_radial, nep_gpu_model.paramb.MN_angular, max_neighs, nep_gpu_model.annmb.dim, nlocal, nprocs_total);
  // }

  // Invoke NEP computation
  if (!is_devi_step) {//增加了多模型偏差值计算，不涉及多模型则nep_gpu_models数组只有1个模型
    nep_gpu_models[0].compute(
      eflag_global,
      need_eatom_device,
      need_vflag_either_device,
      vflag_global,
      vflag_atom,
      need_cvatom_device,
      nall,                    // nall = nlocal + nghost
      inum,                    // inum = number of local atoms
      nlocal,                  // nlocal
      max_neighs,
      num_neighs,
      const_cast<int*>(reinterpret_cast<const int*>(type.data())),           // 原子类型
      const_cast<int*>(reinterpret_cast<const int*>(d_ilist.data())),        // ilist
      const_cast<int*>(reinterpret_cast<const int*>(d_numneigh.data())),     // numneigh
      const_cast<int*>(reinterpret_cast<const int*>(d_neighbors.data())),    // neighbors
      const_cast<double*>(reinterpret_cast<const double*>(x.data())),        // 位置
      // const_cast<double*>(reinterpret_cast<const double*>(d_eatom.data())),        // 位置
      // d_eatom.data(),                                 // 原子能量输出
      // const_cast<double*>(reinterpret_cast<double*>(d_eatom.data())),
      d_eatom.data(),
      nullptr,
      f.data(),                                       // 力输出 如果num_ff 大于0，只输出到full_f
      nullptr,                                        // 如果 num_ff 大于0，输出到f 和 full_f
      virial_per_atom_ptr,                           // 6-component per-atom virial or internal work buffer
      cvirial_per_atom_ptr,                          // 9-component centroid virial
      h_etot_virial_global.data(),
      neighbor_rebuilt
    );
    // Wait for NEP computation to complete
    Kokkos::fence();    
  } else {
    for (int ff_idx = num_ff - 1; ff_idx >= 0; --ff_idx) {
      auto d_full_e = k_full_e.view<DeviceType>();
      double* full_e_ptr = d_full_e.data() + ff_idx * nlocal;

      auto d_full_f = k_full_f.view<DeviceType>();
      double* full_f_ptr = d_full_f.data() + ff_idx * nall * 3;
      nep_gpu_models[ff_idx].compute(
        eflag_global,
        need_eatom_device,
        need_vflag_either_device,
        vflag_global,
        vflag_atom,
        need_cvatom_device,
        nall,                    // nall = nlocal + nghost
        inum,                    // inum = number of local atoms
        nlocal,                  // nlocal
        max_neighs,
        num_neighs,
        const_cast<int*>(reinterpret_cast<const int*>(type.data())),           // 原子类型
        const_cast<int*>(reinterpret_cast<const int*>(d_ilist.data())),        // ilist
        const_cast<int*>(reinterpret_cast<const int*>(d_numneigh.data())),     // numneigh
        const_cast<int*>(reinterpret_cast<const int*>(d_neighbors.data())),    // neighbors
        const_cast<double*>(reinterpret_cast<const double*>(x.data())),        // 位置
        // const_cast<double*>(reinterpret_cast<const double*>(d_eatom.data())),        // 位置
        // d_eatom.data(),                                 // 原子能量输出
        // const_cast<double*>(reinterpret_cast<double*>(d_eatom.data())),
        d_eatom.data(),
        full_e_ptr,
        f.data(),                                       // 力输出 如果num_ff 大于0，只输出到full_f
        full_f_ptr,                                  // 如果 num_ff 大于0，输出到f 和 full_f
        virial_per_atom_ptr,                           // 6-component per-atom virial or internal work buffer
        cvirial_per_atom_ptr,                          // 9-component centroid virial
        h_etot_virial_global.data(),
        neighbor_rebuilt
      );
      // Wait for NEP computation to complete
      Kokkos::fence();
    }
  }
  // nep_gpu_model.getGPUMemoryStats(total, used, free);
  if (is_devi_step) {
    reverse_comm_mode = ReverseCommMode::FORCE_DEVI;
    comm_reverse = 3 * num_ff;                 // 设置per-atom doubles数
    auto d_full_f = k_full_f.view<DeviceType>();
    auto d_full_e = k_full_e.view<DeviceType>();
    comm->reverse_comm(this);  // pack/unpack handle their own host/device mirrors internally
    comm_reverse = 0;                          // 重置
    reverse_comm_mode = ReverseCommMode::NONE;

    auto h_final_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_full_f);
    auto h_final_e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_full_e);
    atomKK->sync(Host, datamask_read | datamask_modify);
    auto h_ilist = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_ilist);

    double glb_avg_f_err, glb_max_f_err, glb_min_f_err;
    double glb_avg_ei_err, glb_max_ei_err, glb_min_ei_err;
    double avg_f_err, max_f_err, min_f_err, avg_ei_err, max_ei_err, min_ei_err;

    std::tuple<double, double, double, double, double, double> result = calc_max_error(
      h_ilist.data(),
      h_final_f.data(),
      h_final_e.data(),
      num_ff,
      inum,
      nlocal,
      nall,
      rank);

    avg_f_err = std::get<0>(result);
    max_f_err = std::get<1>(result);
    min_f_err = std::get<2>(result);
    avg_ei_err = std::get<3>(result);
    max_ei_err = std::get<4>(result);
    min_ei_err = std::get<5>(result);


    MPI_Allreduce(&max_f_err, &glb_max_f_err, 1, MPI_DOUBLE, MPI_MAX, world);
    MPI_Allreduce(&max_ei_err, &glb_max_ei_err, 1, MPI_DOUBLE, MPI_MAX, world);

    MPI_Allreduce(&min_f_err, &glb_min_f_err, 1, MPI_DOUBLE, MPI_MIN, world);
    MPI_Allreduce(&min_ei_err, &glb_min_ei_err, 1, MPI_DOUBLE, MPI_MIN, world);

    MPI_Allreduce(&avg_f_err, &glb_avg_f_err, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&avg_ei_err, &glb_avg_ei_err, 1, MPI_DOUBLE, MPI_SUM, world);

    if (atom->natoms > 0) {
        glb_avg_f_err /= static_cast<double>(atom->natoms);
        glb_avg_ei_err /= static_cast<double>(atom->natoms);
    } else {
        glb_avg_f_err = 0.0;
        glb_avg_ei_err = 0.0;
    }
    if (is_rank_0) {
      fprintf(explrError_fp, "%9lld %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f\n",
              update->ntimestep,
              glb_avg_f_err, glb_min_f_err, glb_max_f_err,
              glb_avg_ei_err, glb_min_ei_err, glb_max_ei_err);
      fflush(explrError_fp);
    }
  }

  if (cvflag_atom) {
    // 同步代码执行后cvatom 会自动同步到结果，只是结果是按照device中的列优先存储的。
    k_cvatom.template modify<DeviceType>();
    k_cvatom.sync<LMPHostType>();
    // auto h_cv = k_cvatom.h_view;
    // for (int i = 0; i < maxcvatom; ++i) {
    //   cvatom[i] = h_cv.data() + i * 9;
    // }

    // for (int i = 0; i < nall; ++i) {
    //   cvatom[i][0] = h_cv.data()[i + 0 * nall];  // xx
    //   cvatom[i][1] = h_cv.data()[i + 1 * nall];  // yy
    //   cvatom[i][2] = h_cv.data()[i + 2 * nall];  // zz
    //   cvatom[i][3] = h_cv.data()[i + 3 * nall];  // xy
    //   cvatom[i][4] = h_cv.data()[i + 4 * nall];  // xz
    //   cvatom[i][5] = h_cv.data()[i + 5 * nall];  // yz
    //   cvatom[i][6] = h_cv.data()[i + 6 * nall];  // yx
    //   cvatom[i][7] = h_cv.data()[i + 7 * nall];  // zx
    //   cvatom[i][8] = h_cv.data()[i + 8 * nall];  // zy
    // }

    // for (int i = 0; i < std::min(10, nlocal); ++i) {
    //          i, cvatom[i][0], cvatom[i][1], cvatom[i][2], cvatom[i][3], cvatom[i][4], cvatom[i][5], cvatom[i][6], cvatom[i][7], cvatom[i][8]);
    // }
    // for (int i = nall-11; i < nall; ++i) {
    //          i, cvatom[i][0], cvatom[i][1], cvatom[i][2], cvatom[i][3], cvatom[i][4], cvatom[i][5], cvatom[i][6], cvatom[i][7], cvatom[i][8]);
    // }

  }

  if (need_dup)
    Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += h_etot_virial_global[0];
  if (vflag_global) {
    virial[0] += h_etot_virial_global[1];
    virial[1] += h_etot_virial_global[2];
    virial[2] += h_etot_virial_global[3];
    virial[3] += h_etot_virial_global[4];
    virial[4] += h_etot_virial_global[5];
    virial[5] += h_etot_virial_global[6];    
  }

  if (eflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }



  copymode = 0;
  // free duplicated memory
  if (need_dup) {
    dup_f     = decltype(dup_f)();
    dup_eatom = decltype(dup_eatom)();
    dup_vatom = decltype(dup_vatom)();
  }  
  // Debug-only barrier: Kokkos fence + LAMMPS reverse_comm already guarantee
  // consistency. Uncomment to diagnose timing/ordering issues at scale.
  // MPI_Barrier(world);
}

template<class DeviceType>
void PairNEPKokkos<DeviceType>::update_heatflux_reverse()
{
  if (!has_device_heatflux_views())
    error->all(FLERR, "Heat flux reverse communication requires device eatom and cvatom views");

  const int nall_current = atom->nlocal + atom->nghost;
  const int required_nmax = MAX(atom->nmax, nall_current);
  if (required_nmax > heatflux_reverse_nmax) {
    heatflux_reverse_nmax = required_nmax;
    k_heatflux_reverse = DAT::tdual_xfloat_1d("pair:heatflux_reverse", heatflux_reverse_nmax * HEATFLUX_REVERSE_WIDTH);
  }

  auto reverse_view = k_heatflux_reverse.view<DeviceType>();
  auto eatom_view = d_eatom;
  auto cvatom_view = d_cvatom;
  const int nlocal_current = atom->nlocal;

  Kokkos::parallel_for("pair_nep_heatflux_pack", nall_current, KOKKOS_LAMBDA(const int i) {
    const int base = i * HEATFLUX_REVERSE_WIDTH;
    reverse_view(base + 0) = (i < nlocal_current) ? static_cast<X_FLOAT>(eatom_view(i)) : static_cast<X_FLOAT>(0.0);
    for (int j = 0; j < 9; ++j)
      reverse_view(base + 1 + j) = static_cast<X_FLOAT>(cvatom_view(i, j));
  });
  k_heatflux_reverse.template modify<DeviceType>();

  reverse_comm_mode = ReverseCommMode::HEATFLUX;
  comm_reverse = HEATFLUX_REVERSE_WIDTH;
  comm->reverse_comm(this);
  k_heatflux_reverse.template sync<DeviceType>();

  Kokkos::parallel_for(
    "pair_nep_heatflux_zero_ghost",
    Kokkos::RangePolicy<DeviceType>(atom->nlocal, nall_current),
    KOKKOS_LAMBDA(const int i) {
      const int base = i * HEATFLUX_REVERSE_WIDTH;
      for (int j = 0; j < HEATFLUX_REVERSE_WIDTH; ++j)
        reverse_view(base + j) = 0;
    });

  k_heatflux_reverse.template modify<DeviceType>();

  comm_reverse = 0;
  reverse_comm_mode = ReverseCommMode::NONE;
}

// pack_reverse_comm：使用临时 mirror 拷贝 device 数据
template<class DeviceType>
int PairNEPKokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
{
  if (reverse_comm_mode == ReverseCommMode::HEATFLUX) {
    k_heatflux_reverse.template sync<LMPHostType>();
    auto host_reverse = k_heatflux_reverse.h_view;
    int m = 0;
    int last = first + n;
    for (int i = first; i < last; ++i) {
      int base = i * HEATFLUX_REVERSE_WIDTH;
      for (int j = 0; j < HEATFLUX_REVERSE_WIDTH; ++j)
        buf[m++] = static_cast<double>(host_reverse(base + j));
    }
    return m;
  }

  // 获取 device 视图（临时）
  auto d_full_f = k_full_f.view<DeviceType>();

  // 强制拷贝 device → host（绕过 sync 问题）
  auto h_temp = Kokkos::create_mirror_view(d_full_f);
  Kokkos::deep_copy(h_temp, d_full_f);

  int m = 0;
  int last = first + n;
  for (int i = first; i < last; ++i) {
    for (int ff = 0; ff < num_ff; ++ff) {
      int base_idx = ff * nall * 3 + i * 3;
      buf[m++] = h_temp(base_idx + 0);
      buf[m++] = h_temp(base_idx + 1);
      buf[m++] = h_temp(base_idx + 2);
    }
  }

  return m;
}

template<class DeviceType>
int PairNEPKokkos<DeviceType>::pack_reverse_comm_kokkos(int n, int first, DAT::tdual_xfloat_1d &buf)
{
  auto dst = buf.view<DeviceType>();
  if (reverse_comm_mode == ReverseCommMode::HEATFLUX) {
    auto src = k_heatflux_reverse.view<DeviceType>();
    const int width = HEATFLUX_REVERSE_WIDTH;
    Kokkos::parallel_for("pair_nep_pack_heatflux_reverse_comm_kokkos", n, KOKKOS_LAMBDA(const int i) {
      const int src_base = (first + i) * width;
      const int dst_base = i * width;
      for (int j = 0; j < width; ++j)
        dst(dst_base + j) = src(src_base + j);
    });
    return n * width;
  }

  if (reverse_comm_mode == ReverseCommMode::FORCE_DEVI) {
    auto src = k_full_f.view<DeviceType>();
    const int width = 3 * num_ff;
    const int stride = nall * 3;
    const int num_ff_local = num_ff;
    Kokkos::parallel_for("pair_nep_pack_force_devi_reverse_comm_kokkos", n, KOKKOS_LAMBDA(const int i) {
      const int atom_index = first + i;
      const int dst_base = i * width;
      for (int ff = 0; ff < num_ff_local; ++ff) {
        const int src_base = ff * stride + atom_index * 3;
        const int out_base = dst_base + ff * 3;
        dst(out_base + 0) = src(src_base + 0);
        dst(out_base + 1) = src(src_base + 1);
        dst(out_base + 2) = src(src_base + 2);
      }
    });
    return n * width;
  }

  return 0;
}

// unpack_reverse_comm：先累加到临时 host mirror，再拷贝回 device
template<class DeviceType>
void PairNEPKokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
{
  if (reverse_comm_mode == ReverseCommMode::HEATFLUX) {
    k_heatflux_reverse.template modify<LMPHostType>();
    auto host_reverse = k_heatflux_reverse.h_view;
    int m = 0;
    for (int i = 0; i < n; ++i) {
      int base = list[i] * HEATFLUX_REVERSE_WIDTH;
      for (int j = 0; j < HEATFLUX_REVERSE_WIDTH; ++j)
        host_reverse(base + j) += static_cast<X_FLOAT>(buf[m++]);
    }
    return;
  }

  // 获取 device 视图
  auto d_full_f = k_full_f.view<DeviceType>();

  // 先把当前 device 数据拷贝到临时 host（保留原始值）
  auto h_temp = Kokkos::create_mirror_view(d_full_f);
  Kokkos::deep_copy(h_temp, d_full_f);

  // 累加其他 rank 贡献到临时 host
  int m = 0;
  for (int i = 0; i < n; ++i) {
    int j = list[i];
    for (int ff = 0; ff < num_ff; ++ff) {
      int base_idx = ff * nall * 3 + j * 3;
      h_temp(base_idx + 0) += buf[m++];
      h_temp(base_idx + 1) += buf[m++];
      h_temp(base_idx + 2) += buf[m++];
    }
  }

  Kokkos::deep_copy(d_full_f, h_temp);
}

template<class DeviceType>
void PairNEPKokkos<DeviceType>::unpack_reverse_comm_kokkos(int n, DAT::tdual_int_1d list, DAT::tdual_xfloat_1d &buf)
{
  auto idx = list.view<DeviceType>();
  auto src = buf.view<DeviceType>();
  if (reverse_comm_mode == ReverseCommMode::HEATFLUX) {
    auto dst = k_heatflux_reverse.view<DeviceType>();
    const int width = HEATFLUX_REVERSE_WIDTH;
    Kokkos::parallel_for("pair_nep_unpack_heatflux_reverse_comm_kokkos", n, KOKKOS_LAMBDA(const int i) {
      const int dst_base = idx(i) * width;
      const int src_base = i * width;
      for (int j = 0; j < width; ++j)
        dst(dst_base + j) += src(src_base + j);
    });
    return;
  }

  if (reverse_comm_mode == ReverseCommMode::FORCE_DEVI) {
    auto dst = k_full_f.view<DeviceType>();
    const int width = 3 * num_ff;
    const int stride = nall * 3;
    const int num_ff_local = num_ff;
    Kokkos::parallel_for("pair_nep_unpack_force_devi_reverse_comm_kokkos", n, KOKKOS_LAMBDA(const int i) {
      const int atom_index = idx(i);
      const int src_base = i * width;
      for (int ff = 0; ff < num_ff_local; ++ff) {
        const int dst_base = ff * stride + atom_index * 3;
        const int in_base = src_base + ff * 3;
        dst(dst_base + 0) += src(in_base + 0);
        dst(dst_base + 1) += src(in_base + 1);
        dst(dst_base + 2) += src(in_base + 2);
      }
    });
  }
}

template<class DeviceType>
std::tuple<double, double, double, double, double, double>
PairNEPKokkos<DeviceType>::calc_max_error(const int* ilist, const F_FLOAT* h_full_f, const F_FLOAT* h_full_e, const int num_ff, const int inum, const int nlocal, const int nall, const int rank)
{
    double num_ff_inv = 1.0 / num_ff;
    // 注意索引，要用 inum 去索引原子
    // 1. 计算所有模型的平均力 f_ave 和平均能量 ei_ave（只对 local atoms）
    std::vector<double> f_ave(inum * 3, 0.0);
    std::vector<double> ei_ave(inum, 0.0);
    int i_index = 0;
    for (int ff = 0; ff < num_ff; ++ff) {
      for (int i = 0; i < inum; ++i) {
        i_index = ilist[i];
        int base_idx_f = ff * nall * 3 + i_index * 3;
        int base_idx_e = ff * nlocal + i_index;

        f_ave[i * 3 + 0] += h_full_f[base_idx_f + 0];// 结果不用按照local存
        f_ave[i * 3 + 1] += h_full_f[base_idx_f + 1];
        f_ave[i * 3 + 2] += h_full_f[base_idx_f + 2];

        ei_ave[i] += h_full_e[base_idx_e];
      }
    }

    // 求平均
    for (int i = 0; i < inum * 3; ++i) {
        f_ave[i] *= num_ff_inv;
    }
    for (int i = 0; i < inum; ++i) {
        ei_ave[i] *= num_ff_inv;
    }

    // 2. 计算每个模型的偏差（平方和用于后续 rms）
    std::vector<double> f_err_sq_sum(inum, 0.0);   // 每个原子的力平方偏差和（用于均方根）
    std::vector<double> ei_err_sq_sum(inum, 0.0);  // 每个原子的能量平方偏差和

    double avg_f_err = 0.0;   // 用于计算平均力误差
    double avg_ei_err = 0.0;  // 用于计算平均能量误差

    double min_f_err = 1e9, max_f_err = 0.0;
    double min_ei_err = 1e9, max_ei_err = 0.0;
    double _tmp_f = 0.0;
    double _tmp_ei = 0.0;

    for (int ff = 0; ff < num_ff; ++ff) {
      for (int i = 0; i < inum; ++i) {
        i_index = ilist[i];
        int base_idx_f = ff * nall * 3 + i_index * 3;
        int base_idx_e = ff * nlocal + i_index;

        double dfx = h_full_f[base_idx_f + 0] - f_ave[i * 3 + 0];
        double dfy = h_full_f[base_idx_f + 1] - f_ave[i * 3 + 1];
        double dfz = h_full_f[base_idx_f + 2] - f_ave[i * 3 + 2];
        f_err_sq_sum[i]  += dfx*dfx + dfy*dfy + dfz*dfz;

        ei_err_sq_sum[i] += (h_full_e[base_idx_e] - ei_ave[i])*(h_full_e[base_idx_e] - ei_ave[i]);

      }
    }

    for (int i = 0; i < inum; ++i) {
      _tmp_f = sqrt(f_err_sq_sum[i] / num_ff);
      _tmp_ei  = sqrt(ei_err_sq_sum[i]/ num_ff);
      min_f_err = std::min(min_f_err, _tmp_f);
      max_f_err = std::max(max_f_err, _tmp_f);
      min_ei_err = std::min(min_ei_err, _tmp_ei);
      max_ei_err = std::max(max_ei_err, _tmp_ei);
      avg_f_err += _tmp_f;
      avg_ei_err += _tmp_ei;
    }

    return std::make_tuple(
        avg_f_err, 
        max_f_err, 
        min_f_err, 
        avg_ei_err,
        max_ei_err, 
        min_ei_err 
    );
}
/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairNEPKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairNEPKokkos<LMPHostType>;
#endif
}
