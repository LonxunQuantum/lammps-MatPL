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
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "suffix.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <algorithm>
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

  suffix_flag |= Suffix::KOKKOS;
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  me = comm->me;
  h_etot_virial_global.resize(7);
  // 这种方式可以直接输出到log.lammps中
  // if (comm->me == 0) {
  //   utils::logmesg(lmp, "=== PairNEPKokkos Constructor Finished ===\n");
  // }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairNEPKokkos<DeviceType>::~PairNEPKokkos()
{
  // printf("=====rank %d device %d doing ~pairnep start copymode %d =====\n", rank, device_id, copymode);
  if (copymode) return;

  if (allocated) {
    if(local_maxeatom > 0) memoryKK->destroy_kokkos(k_eatom, eatom);
    if(local_maxvatom > 0) memoryKK->destroy_kokkos(k_vatom, vatom);
    if(local_maxcvatom> 0) memoryKK->destroy_kokkos(k_cvatom, cvatom);
  }
  h_etot_virial_global = decltype(h_etot_virial_global)();
  // printf("=====rank %d device %d doing ~pairnep end =====\n", rank, device_id);
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
      explrError_fp = fopen(&explrError_fname[0], "w");
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
    // printf("Rank %d: Using CUDA device %d (Kokkos-managed)\n", comm->me, device_id);
    rank = comm->me;
    // if (is_rank_0) {
    //   printf("Rank %d: Using CUDA device %d (Kokkos-managed)\n", comm->me, device_id);
    // }
    // 无需 cudaSetDevice(device_id); // 已由Kokkos设置
  }
  nep_gpu_models.resize(num_ff);
  for (int i=0; i < num_ff; i++) {
    std::string model_file = potential_files[i];
    nep_gpu_models[i].read_neptxt(model_file.c_str(), is_rank_0, comm->me, device_id, i);
    if (i == 0) {
      cutoff = nep_gpu_models[i].paramb.rc_radial;
      cutoffsq = cutoff * cutoff;
    }
    if (is_rank_0) {
      utils::logmesg(lmp, "NEP Kokkos potential " + model_file + " loaded successfully\n");
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
    std::vector<int> atom_type_module = nep_gpu_models[f1].cpu_element_atomic_number_list;
    std::vector<int> atom_types;
    for (int ii = 2; ii < narg; ++ii) {
      std::string element = utils::strdup(arg[ii]);  // LAMMPS提供的安全转换
      int temp = find_atomic_number(element);
      // int temp = std::stoi(arg[ii]);
      auto iter = std::find(atom_type_module.begin(), atom_type_module.end(), temp);   
      if (iter != atom_type_module.end() || arg[ii] == 0)
      {
          int index = std::distance(atom_type_module.begin(), iter);
          // model_atom_type_idx.push_back(index); 
          // atom_types.push_back(temp);
          atom_types.push_back(index);
          // printf("=== rank %d device_id %d coeff the config atom type %d index in ff is %d\n",\
            rank, device_id, temp, index);
      }
      else
      {
          error->all(FLERR, "This element is not included in the nep file: " + potential_files[f1]);
      }
    }
    nep_gpu_models[f1].set_atom_type_map(narg-2, atom_types.data());
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
  // printf("===== DEBUG: newton_pair = %d =====\n", newton_pair);
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
      "h ","he",
      "li","be","b ","c ","n ","o ","f ","ne",
      "na","mg","al","si","p ","s ","cl","ar",
      "k ","ca","sc","ti","v ","cr","mn","fe","co","ni","cu",
      "zn","ga","ge","as","se","br","kr",
      "rb","sr","y ","zr","nb","mo","tc","ru","rh","pd","ag",
      "cd","in","sn","sb","te","i ","xe",
      "cs","ba","la","ce","pr","nd","pm","sm","eu","gd","tb","dy",
      "ho","er","tm","yb","lu","hf","ta","w ","re","os","ir","pt",
      "au","hg","tl","pb","bi","po","at","rn",
      "fr","ra","ac","th","pa","u ","np","pu"
  };

  std::vector<std::string> element_table2 = {
      "1 ","2 ",
      "3 ","4 ","5 ","6 ","7 ","8 ","9 ","10",
      "11","12","13","14","15","16","17","18",
      "19","20","21","22","23","24","25","26","27","28","29",
      "30","31","32","33","34","35","36",
      "37","38","39","40","41","42","43","44","45","46","47",
      "48","49","50","51","52","53","54",
      "55","56","57","58","59","60","61","62","63","64","65","66",
      "67","68","69","70","71","72","73","74","75","76","77","78",
      "79","80","81","82","83","84","85","86",
      "87","88","89","90","91","92","93","94",
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
void PairNEPKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;
  ev_init(eflag, vflag, 0);
  // size_t total, used, free;
  bigint ntimestep = update->ntimestep;
  bool is_devi_step = (num_ff > 1) && (ntimestep % out_freq == 0);

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK); //同步原子数据到设备
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  type = atomKK->k_type.template view<DeviceType>();

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  inum = list->inum;

  // printf("DEBUG rank=%d device_id=%d step=%d eflag=%d, vflag=%d, vflag_fdotr=%d eflag_atom=%d, vflag_atom=%d cvflag_atom=%d vflag_global=%d vflag_either %d centroidstressflag=%d maxeatom %d->%d maxvatom %d->%d maxcvatom %d->%d nlocal %d inum %d nall %d\n", \
        rank, device_id, ntimestep, eflag, vflag, vflag_fdotr, eflag_atom, vflag_atom, cvflag_atom, vflag_global, vflag_either, centroidstressflag, maxeatom, local_maxeatom, maxvatom, local_maxvatom, maxcvatom, local_maxcvatom, nlocal, inum, nall);
  // printf("======compute before k_eatom.extent(0) %d k_vatom.extent(0) %d k_cvatom.extent(0) %d ======\n",k_eatom.extent(0), k_vatom.extent(0), k_cvatom.extent(0));
  
  int cur_max = max(maxcvatom, maxvatom);
  if (eflag_atom) {
  memoryKK->destroy_kokkos(k_eatom,eatom);
  memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
  d_eatom = k_eatom.view<DeviceType>();
  }

  if (cvflag_atom) {
    if (local_maxcvatom < cur_max) local_maxcvatom = cur_max;
    memoryKK->destroy_kokkos(k_cvatom, cvatom);           // 销毁旧的
    memoryKK->create_kokkos(k_cvatom, cvatom, local_maxcvatom, "pair:cvatom");
    // printf("cvatom grow to %d\n", nmax_cvatom);
    // for (int i = 0; i < std::min(5, nlocal); ++i) {
    //   printf(" init local cvatom[%6d]: %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
    //          i, cvatom[i][0], cvatom[i][1], cvatom[i][2], cvatom[i][3], cvatom[i][4], cvatom[i][5], cvatom[i][6], cvatom[i][7], cvatom[i][8]);
    // }
    // for (int i = nall-6; i < nall; ++i) {
    //   printf(" init ghost cvatom[%6d]: %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
    //          i, cvatom[i][0], cvatom[i][1], cvatom[i][2], cvatom[i][3], cvatom[i][4], cvatom[i][5], cvatom[i][6], cvatom[i][7], cvatom[i][8]);
    // }
  }

  if (!cvflag_atom && vflag_either) {
    if (local_maxvatom < cur_max) local_maxvatom = cur_max;
      memoryKK->destroy_kokkos(k_vatom, vatom);
      memoryKK->create_kokkos(k_vatom, vatom, local_maxvatom, "pair:vatom");
      d_vatom = k_vatom.view<DeviceType>();
      // printf("vatom grow to %d\n", local_maxvatom);
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
  // printf("ntimestep %d rank=%d device_id=%d nranks=%d | (e,v,ee,eg,ea,ve,vg,va)=(%d,%d,%d,%d,%d,%d,%d,%d) | inum=%d, nall=%d | maxeatom=%d, maxvatom=%d | lmpneigh=%dX%d | xsize=%d X %d | fsize=%d X %d | vsize=%d X %d paramb-2b %d 3b %d\n", \
       ntimestep, rank, device_id, nprocs_total, eflag, vflag, eflag_either, eflag_global, eflag_atom, vflag_either, vflag_global, vflag_atom, inum, nall, maxeatom, maxvatom, num_neighs, max_neighs, x.extent(0), x.extent(1), f.extent(0), f.extent(1), d_vatom.extent(0), d_vatom.extent(1), nep_gpu_models[0].paramb.n_max_radial, nep_gpu_models[0].paramb.n_max_angular);
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
  // printf("MB-before-1 rank=%d device_id=%d step=%d Total=%f Used=%f Free=%f (GB)\n", rank, device_id, ntimestep, total / (1024.0 * 1024.0 * 1024.0), used / (1024.0 * 1024.0 * 1024.0), free / (1024.0 * 1024.0 * 1024.0));
  // if (nep_gpu_model.allocate_once==0 and is_rank_0) {
  //   int result = nep_gpu_model.calculateMaxAtoms(nep_gpu_model.paramb.MN_radial, nep_gpu_model.paramb.MN_angular, max_neighs, nep_gpu_model.annmb.dim, nlocal, nprocs_total);
  // }

  // Invoke NEP computation
  // printf("=========nep start compute force ==========\n");
    nep_gpu_models[0].compute(
      eflag_global,
      eflag_atom, 
      vflag_either,
      vflag_global,
      vflag_atom,
      cvflag_atom,
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
      d_vatom.data(),                                 // 总维里输出
      cvflag_atom ? k_cvatom.d_view.data() : nullptr,          // device 侧 9 分量指针
      h_etot_virial_global.data()
    );
    // printf("=========end start compute force  etot %f ==========\n", h_etot_virial_global[0]);
    // Wait for NEP computation to complete
    Kokkos::fence();    
  
  // nep_gpu_model.getGPUMemoryStats(total, used, free);
  // printf("MB-after-0 rank=%d device_id=%d step=%d Total=%f Used=%f Free=%f (GB)\n", rank, device_id, ntimestep, total / (1024.0 * 1024.0 * 1024.0), used / (1024.0 * 1024.0 * 1024.0), free / (1024.0 * 1024.0 * 1024.0));
  
  if (cvflag_atom) {
    // 同步代码执行后cvatom 会自动同步到结果，只是结果是按照device中的列优先存储的。
    k_cvatom.template modify<DeviceType>();
    k_cvatom.sync<LMPHostType>();

    auto h_cv = k_cvatom.h_view;
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
    //   printf(" local cvatom[%6d]: %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
    //          i, cvatom[i][0], cvatom[i][1], cvatom[i][2], cvatom[i][3], cvatom[i][4], cvatom[i][5], cvatom[i][6], cvatom[i][7], cvatom[i][8]);
    // }
    // for (int i = nall-11; i < nall; ++i) {
    //   printf(" ghost cvatom[%6d]: %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
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
    // printf("virial[0-5]=%f %f %f %f %f %f\n", virial[0], virial[1], virial[2], virial[3], virial[4], virial[5]);
  }

  if (eflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_either) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
  // free duplicated memory
  if (need_dup) {
    dup_f     = decltype(dup_f)();
    dup_eatom = decltype(dup_eatom)();
    dup_vatom = decltype(dup_vatom)();
  }  
  // 同步MPI进程，确保所有rank完成计算
  MPI_Barrier(world);
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairNEPKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairNEPKokkos<LMPHostType>;
#endif
}