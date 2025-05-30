#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include "pair_matpl.h"
#include "nep_cpu.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "domain.h"
#include <dlfcn.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <algorithm>
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMATPL::PairMATPL(LAMMPS *lmp) : Pair(lmp)
{
    me = comm->me;
	writedata = 1;
    comm_reverse = 3;

    restartinfo = 0;//  set to 0 if your pair style does not store data in restart files
    manybody_flag = 1; //set to 1 if your pair style is not pair-wise additive
    single_enable = 0; 
    // copymode = 0;
    // allocated = 0;

}

PairMATPL::~PairMATPL()
{
    // if (copymode)
    //     return;

    if (allocated) 
    {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(f_n);
        memory->destroy(e_atom_n);
    }

    if (me == 0 && explrError_fp != nullptr) {
        fclose(explrError_fp);
        explrError_fp = nullptr;
    }

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMATPL::allocate()
{
    allocated = 1;
    int np1 = atom->ntypes ;
    memory->create(setflag, np1 + 1, np1 + 1, "pair:setflag");
    for (int i = 1; i <= np1; i++)
        for (int j = i; j <= np1; j++) setflag[i][j] = 0;
    memory->create(cutsq, np1 + 1, np1 + 1, "pair:cutsq");

}

/* ----------------------------------------------------------------------
   global settings pair_style 
------------------------------------------------------------------------- */

void PairMATPL::settings(int narg, char** arg)
{
    int ff_idx;
    int iarg = 0;  // index of first forcefield file
    int rank;
    int num_devices = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef USE_CUDA
        cudaGetDeviceCount(&num_devices);
    #else
        num_devices = 0;
    #endif
    if (narg <= 0) error->all(FLERR, "Illegal pair_style command"); // numbers of args after 'pair_style matpl'
    std::vector<std::string> models;

    num_ff = 0;
    while (iarg < narg) {
        std::string arg_str(arg[iarg]);
        if (arg_str.find(".txt") != std::string::npos) {
            models.push_back(arg_str);
            num_ff++;
            iarg++;
        } else {
            break;
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

    if (me == 0 and num_ff > 1) {
        explrError_fp = fopen(&explrError_fname[0], "w");
        fprintf(explrError_fp, "%9s %16s %16s %16s %16s %16s %16s\n", 
        "#    step", "avg_devi_f", "min_devi_f", "max_devi_f", 
        "avg_devi_e", "min_devi_e", "max_devi_e");
        fflush(explrError_fp);
    }

    if (me == 0) utils::logmesg(this -> lmp, "<---- Loading model ---->");
    
    if (num_devices < 1) {
        use_nep_gpu = false;
        nep_cpu_models.resize(num_ff);
    } else {
        use_nep_gpu = true;
        #ifdef USE_CUDA
            nep_gpu_models.resize(num_ff);
        #else
            use_nep_gpu = false;
            nep_cpu_models.resize(num_ff);
        #endif
    }

    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        std::string model_file = models[ff_idx];
        // load txt format model file, it should be nep
        bool is_rank_0 = (comm->me == 0);

        // if the gpu nums > 0 and libnep.so is exsits, use gpu model
        if (ff_idx > 0) {
            is_rank_0 = false; //for print
        }
        #ifdef USE_CUDA
        if (use_nep_gpu == true) {
            int device_id = rank % num_devices;
            cudaSetDevice(device_id);

            nep_gpu_models[ff_idx].init_from_file(model_file.c_str(), is_rank_0, device_id);
            model_type = 2;
            if (device_id == 0) {
                printf("MPI rank %d rank using GPU device %d\n", rank, device_id);
            }
            // std::cout<<"load nep.txt success and the model type is 2" << std::endl;
        }
        #endif
        
        if(use_nep_gpu == false) {
            // NEP3_CPU nep_cpu_model;
            // nep_cpu_model.init_from_file(model_file, is_rank_0);
            nep_cpu_models[ff_idx].init_from_file(model_file, is_rank_0);
            model_type = 1;
        }
        if (me == 0) printf("\nLoading txt model file:   %s\n", model_file.c_str());
    }
    if (model_type == 1) {
        cutoff = nep_cpu_models[0].paramb.rc_radial;
    } else if (model_type == 2) {
        #ifdef USE_CUDA
            cutoff = nep_gpu_models[0].paramb.rc_radial;
        #else
            cutoff = 0.0;
        #endif
    }
    // since we need num_ff, so well allocate memory here
    // but not in allocate()
    // printf("========allocate f_n[ %d %d 3 ]=========\n", num_ff, atom->nmax);
    nmax = atom->nmax;
    memory->create(f_n, num_ff, atom->nmax, 3, "pair_matpl:f_n");
    memory->create(e_atom_n, num_ff, atom->natoms, "pair_matpl:e_atom_n");
} 

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs pair_coeff 
------------------------------------------------------------------------- */

void PairMATPL::coeff(int narg, char** arg)
{
    int ntypes = atom->ntypes;
    if (!allocated) { allocate(); }

    // pair_coeff * * 
    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error); // arg[0] = *
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error); // arg[1] = *

    int count = 0;
    for(int i = ilo; i <= ihi; i++) {
        for(int j = MAX(jlo,i); j <= jhi; j++) 
        {
            setflag[i][j] = 1;
            count++;
        }
    }

    if (model_type == 1) {
        std::vector<int> atom_type_module = nep_cpu_models[0].element_atomic_number_list;
        model_ntypes = atom_type_module.size();
        // if (ntypes > model_ntypes || ntypes != narg - 2)  // type numbers in strucutre file and in pair_coeff should be the same
        // {
        //     error->all(FLERR, "Element mapping is not correct, ntypes = " + std::to_string(ntypes));
        // }
        for (int ii = 2; ii < narg; ++ii) {
            int temp = std::stoi(arg[ii]);
            auto iter = std::find(atom_type_module.begin(), atom_type_module.end(), temp);   
            if (iter != atom_type_module.end() || arg[ii] == 0)
            {
                int index = std::distance(atom_type_module.begin(), iter);
                model_atom_type_idx.push_back(index); 
                atom_types.push_back(temp);
                for(int jj=0; jj < num_ff; ++jj){
                    nep_cpu_models[jj].map_atom_types.push_back(temp);
                    nep_cpu_models[jj].map_atom_type_idx.push_back(index);
                }
                // std::cout<<"=== the config atom type "<< temp << " index in ff is "  << index << std::endl;
            }
            else
            {
                error->all(FLERR, "This element is not included in the machine learning force field");
            }
        }
    } else if (model_type == 2) { // for nep_gpu
        // check or reset
        #ifdef USE_CUDA
        std::vector<int> atom_type_module = nep_gpu_models[0].element_atomic_number_list;
        model_ntypes = atom_type_module.size();
        // if (ntypes > model_ntypes || ntypes != narg - 2)  // type numbers in strucutre file and in pair_coeff should be the same
        // {
        //     error->all(FLERR, "Element mapping is not correct, ntypes = " + std::to_string(ntypes));
        // }
        for (int ii = 2; ii < narg; ++ii) {
            int temp = std::stoi(arg[ii]);
            auto iter = std::find(atom_type_module.begin(), atom_type_module.end(), temp);   
            if (iter != atom_type_module.end() || arg[ii] == 0)
            {
                int index = std::distance(atom_type_module.begin(), iter);
                model_atom_type_idx.push_back(index); 
                atom_types.push_back(temp);
                for(int jj=0; jj < num_ff; ++jj){
                    nep_gpu_models[jj].map_atom_type_idx.push_back(index);
                }
                // std::cout<<"=== the config atom type "<< temp << " index in ff is "  << index << std::endl;
            }
            else
            {
                error->all(FLERR, "This element is not included in the machine learning force field");
            }
        }
        #endif        
    }
   if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMATPL::init_one(int i, int j)
{
    //if (setflag[i][j] == 0) { error->all(FLERR, "All pair coeffs are not set"); 

    return cutoff;
}


void PairMATPL::init_style()
{
    if (force->newton_pair == 0) error->all(FLERR, "Pair style MATPL requires newton pair on");
    // Using a nearest neighbor table of type full
    neighbor->add_request(this, NeighConst::REQ_FULL);

    cutoffsq = cutoff * cutoff;
    int n = atom->ntypes;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
        cutsq[i][j] = cutoffsq;
}
/* ---------------------------------------------------------------------- */

std::tuple<double, double, double, double, double, double> PairMATPL::calc_max_error(double ***f_n, double **e_atom_n)
{
    int i, j;
    int ff_idx;
    double max_err, err, min_err, max_err_ei, err_ei, min_err_ei;
    double max_mean_err_out, max_mean_err;
    double max_mean_ei_out, max_mean_ei;
    double num_ff_inv;
    int nlocal = atom->nlocal;
    // int *tag = atom->tag;
    min_err = 10000;
    min_err_ei = 10000;
    max_err = 0.0;
    max_err_ei = 0.0;
    num_ff_inv = 1.0 / num_ff;

    max_mean_err_out = 0.0;
    max_mean_err = 0.0;
    max_mean_err_out = 0.0;
    max_mean_ei = 0.0;

    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        p_ff_idx = ff_idx;
        comm->reverse_comm(this);
    }

    std::vector<double> f_ave;
    std::vector<double> f_err[num_ff];
    std::vector<double> f_max_meanff;
    std::vector<double> ei_ave;
    std::vector<double> ei_err[num_ff];
    std::vector<double> ei_max_meanff;

    f_ave.resize(nlocal * 3);
    ei_ave.resize(nlocal);
    f_max_meanff.resize(nlocal);
    ei_max_meanff.resize(nlocal);
    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        f_err[ff_idx].resize(nlocal * 3);
        ei_err[ff_idx].resize(nlocal);
    }

    // sum over all models
    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        for (i = 0; i < nlocal; i++) {
            // std::cout << "f_n[" << ff_idx << "][" << i << "][0] = " << tag[i] << " " << f_n[ff_idx][i][0] << std::endl;
            f_ave[i * 3 + 0] += f_n[ff_idx][i][0];
            f_ave[i * 3 + 1] += f_n[ff_idx][i][1];
            f_ave[i * 3 + 2] += f_n[ff_idx][i][2];
            ei_ave[i] += e_atom_n[ff_idx][i];
            // std::cout<< "ff " << ff_idx << " i " << i << " ei " <<  e_atom_n[ff_idx][i] << " force " << f_n[ff_idx][i][0] << " "  << f_n[ff_idx][i][1] << " "  << f_n[ff_idx][i][2] << std::endl;
        }
    }

    // calc ensemble average
    for (i = 0; i < 3 * nlocal; i++) {
        f_ave[i] *= num_ff_inv;
    }
    for (i = 0; i < nlocal; i++) {
        ei_ave[i] *= num_ff_inv;
    }

    // calc error
    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        for (i = 0; i < nlocal; i++) {
            f_err[ff_idx][i * 3 + 0] = f_n[ff_idx][i][0] - f_ave[i * 3 + 0];
            f_err[ff_idx][i * 3 + 1] = f_n[ff_idx][i][1] - f_ave[i * 3 + 1];
            f_err[ff_idx][i * 3 + 2] = f_n[ff_idx][i][2] - f_ave[i * 3 + 2];
            ei_err[ff_idx][i] = e_atom_n[ff_idx][i] - ei_ave[i];
        }
    }

    // find max error
    for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
        for (j = 0; j < nlocal * 3; j += 3) {
            err = f_err[ff_idx][j] * f_err[ff_idx][j] + f_err[ff_idx][j + 1] * f_err[ff_idx][j + 1] + f_err[ff_idx][j + 2] * f_err[ff_idx][j + 2];
            f_max_meanff[j / 3] += err;
            err = sqrt(err);
            if (err > max_err) max_err = err;
            if (err < min_err) min_err = err;
        }
        for (j = 0; j < nlocal; j++) {
            err_ei = ei_err[ff_idx][j] * ei_err[ff_idx][j];
            ei_max_meanff[j] += err_ei;
            err_ei = sqrt(err_ei);
            if (err_ei > max_err_ei) max_err_ei = err_ei;
            if (err_ei < min_err_ei) min_err_ei = err_ei;
        }
    }

    // find max_mean error
    for (j = 0; j < nlocal; j++) {
        max_mean_ei  = sqrt(ei_max_meanff[j]/ num_ff);
        max_mean_err = sqrt(f_max_meanff[j] / num_ff);
        if (max_mean_err_out < max_mean_err) max_mean_err_out = max_mean_err;
        if (max_mean_ei_out < max_mean_ei) max_mean_ei_out = max_mean_ei;
    }
    return std::make_tuple(max_mean_err_out, max_err, min_err, max_mean_ei_out, max_err_ei, min_err_ei);

}

int PairMATPL::pack_reverse_comm(int n, int first, double* buf) {
    int i, m, last;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
        buf[m++] = f_n[p_ff_idx][i][0];
        buf[m++] = f_n[p_ff_idx][i][1];
        buf[m++] = f_n[p_ff_idx][i][2];
    }
    return m;
}

void PairMATPL::unpack_reverse_comm(int n, int* list, double* buf) {
    int i, j, m;

    m = 0;
    for (i = 0; i < n; i++) {
        j = list[i];
        f_n[p_ff_idx][j][0] += buf[m++];
        f_n[p_ff_idx][j][1] += buf[m++];
        f_n[p_ff_idx][j][2] += buf[m++];
    }

}

void PairMATPL::grow_memory(int nall)
{
  if (nmax < nall) {
    // printf("allocate new %7d %7d %7d\n", update->ntimestep, nmax, nall);
    nmax = nall;
    memory->grow(f_n, num_ff, nmax, 3, "pair_matpl:f_n");
    memory->grow(e_atom_n, num_ff, nmax, "pair_matpl:e_atom_n");
  }
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>PairMATPL::convert_dim(bool is_build_neighbor){
    int nlocal = atom->nlocal;
    int nghost = atom->nghost; 
    int n_all = nlocal + nghost;
    int *itype, *numneigh, **firstneigh;
    int ii, jj, inum, jnum;
    inum = list->inum;
    itype = atom->type;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    double **x = atom->x;
    if (is_build_neighbor) {
        itype_convert_map.assign(n_all, -1);
        position_cpu.assign(n_all*3, 0);
        firstneighbor_cpu.assign(inum * nep_gpu_nm, 0);
        for(ii=0; ii < n_all;ii++) {
            itype_convert_map[ii] = model_atom_type_idx[itype[ii] - 1];
            position_cpu[ii] = x[ii][0];
            position_cpu[  n_all+ii] = x[ii][1];
            position_cpu[2*n_all+ii] = x[ii][2];
            if (ii < inum){
                jnum = numneigh[ii];
                for(jj=0; jj < jnum; ++jj){
                    firstneighbor_cpu[ii*nep_gpu_nm + jj] = firstneigh[ii][jj];
                }
            }
        }
    } else {
        position_cpu.assign(n_all*3, 0);
        for(ii=0; ii < n_all;ii++) {
            position_cpu[ii] = x[ii][0];
            position_cpu[  n_all+ii] = x[ii][1];
            position_cpu[2*n_all+ii] = x[ii][2];            
        }
    }
    return std::make_tuple(std::move(itype_convert_map), std::move(firstneighbor_cpu), std::move(position_cpu));
}

void PairMATPL::compute(int eflag, int vflag)
{
    if (eflag || vflag) ev_setup(eflag, vflag);

    // int newton_pair = force->newton_pair;
    int ff_idx;
    int nlocal = atom->nlocal;
    int current_timestep = update->ntimestep;
    // int total_timestep = update->laststep;
    bool calc_virial_from_mlff = false;
    bool calc_egroup_from_mlff = false;
    int ntypes = atom->ntypes;
    int nghost = atom->nghost;
    int n_all = nlocal + nghost;
    // int inum, jnum, itype, jtype;

    bool is_build_neighbor = false;
    double max_err, global_max_err, max_err_ei, global_max_err_ei;
    double min_err, global_min_err, min_err_ei, global_min_err_ei;
    double max_mean_err_out, global_max_mean_err, max_mean_ei_out, global_max_mean_err_ei;

    double* per_atom_potential = nullptr;
    double** per_atom_virial = nullptr;
    double *virial = force->pair->virial;
    double **f = atom->f;
    // for dp and nep model from jitscript
    if (model_type == 1 and num_ff == 1) {
        double total_potential = 0.0;
        double total_virial[6] = {0.0};
        if (cvflag_atom) {
            per_atom_virial = cvatom;
        }
        if (eflag_atom) {
            per_atom_potential = eatom;
        }
        nep_cpu_models[0].compute_for_lammps(
        atom->nlocal, list->inum, list->ilist, list->numneigh, list->firstneigh, atom->type, atom->x,
        total_potential, total_virial, per_atom_potential, atom->f, per_atom_virial, ff_idx);
        if (eflag) {
            eng_vdwl += total_potential;
        }
        if (vflag) {
            for (int component = 0; component < 6; ++component) {
            virial[component] += total_virial[component];
            }
        }
    }
    else if (model_type == 1 and num_ff > 1){
        // printf("namx %d atom->nmax %d nall %d\n", nmax, atom->nmax, n_all);
        grow_memory(n_all);
        if (cvflag_atom) {
            per_atom_virial = cvatom;
        }
        if (eflag_atom) {
            per_atom_potential = eatom;
        }

        // for (ff_idx = 0; ff_idx < num_ff; ff_idx++) { 
        // This 0.0 setting method will error after the first step
        //  Caught signal 11(Segmentation fault:address not mappedto obiect ataddress 0xc0
        //1 0x00000000006fbc2C LAMMPS NS::Modify::setup()/the/path/src/Obj_mpi/../modify.cpp:308
        //2 0x00000000007d6486 LAMMPS NS::Verlet::setup()/the/path/src/Obj_mpi/../verlet.cpp:159
        //     for (int i = 0; i < n_all; i++) {
        //         f_n[ff_idx][i][0] = 0.0;
        //         f_n[ff_idx][i][1] = 0.0;
        //         f_n[ff_idx][i][2] = 0.0;
        //         e_atom_n[ff_idx][i] = 0.0;
        //     }
        // }

        for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
            if (ff_idx > 0 && (current_timestep % out_freq != 0)) continue;
            double total_potential = 0.0;
            double total_virial[6] = {0.0};
            // for multi models, the output step, should calculate deviation
            for (int i = 0; i < n_all; i++) {
                // printf(" ==== ff_idx %d atom %d num_ff %d n_all %d====\n", ff_idx, i, num_ff, n_all);
                f_n[ff_idx][i][0] = 0.0;
                f_n[ff_idx][i][1] = 0.0;
                f_n[ff_idx][i][2] = 0.0;
            }
            for (int i = 0; i < list->inum; i++) {
                e_atom_n[ff_idx][i] = 0.0;
            }
            nep_cpu_models[ff_idx].compute_for_lammps(
                atom->nlocal, list->inum, list->ilist, list->numneigh, list->firstneigh,atom->type, atom->x,
                total_potential, total_virial, e_atom_n[ff_idx], f_n[ff_idx], per_atom_virial, ff_idx);

            if (ff_idx == 0) {
                for (int i = 0; i < n_all; i++) {
                    f[i][0] = f_n[0][i][0];
                    f[i][1] = f_n[0][i][1];
                    f[i][2] = f_n[0][i][2];
                }
                if (eflag_atom) {
                    for (int i = 0; i < list->inum; i++) {
                        per_atom_potential[i] = e_atom_n[0][i];
                    }
                }
                if (eflag) {
                    eng_vdwl = total_potential;
                }
                if (vflag) {
                    for (int component = 0; component < 6; ++component) {
                        virial[component] += total_virial[component];
                    }
                }
            } // else multi models out steps
        }   // for ff_idx      
    } // model_type == 1: nep_cpu version
    //   exploration mode.
    //   calculate the error of the force
    #ifdef USE_CUDA
    else if (model_type == 2 and num_ff == 1) {

        is_build_neighbor = (current_timestep % neighbor->every == 0);

        if (pre_nlocal != nlocal or pre_nghost != nghost) {
            is_build_neighbor = true;
            pre_nlocal = nlocal;
            pre_nghost = nghost;
        }
        int global_flag;
        int local_flag = is_build_neighbor ? 1 : 0;
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, world);
        is_build_neighbor = global_flag ? true : false;

        if (current_timestep % neighbor->every == 0) {
            is_build_neighbor = true;
        }
        
        double total_potential = 0.0;
        if (cvflag_atom) {
            per_atom_virial = cvatom;
        }
        if (eflag_atom) {
            per_atom_potential = eatom;
        }
        
        // can not set the atom->type (the type set in config) to nep forcefild order, because the ghost atoms type same as the conifg
        // The atomic types corresponding to the index of neighbors are constantly changing

        std::vector<double> cpu_potential_per_atom(list->inum, 0.0);
        std::vector<double> cpu_force_per_atom(n_all * 3, 0.0);
        std::vector<double> cpu_total_virial(6, 0.0);

        // 记录 convert_dim 函数的开始时间
        // auto start_convert_dim = std::chrono::high_resolution_clock::now();
        std::tie(itype_convert_map, firstneighbor_cpu, position_cpu) = convert_dim(is_build_neighbor);
        // 记录 convert_dim 函数的结束时间
        // auto end_convert_dim = std::chrono::high_resolution_clock::now();
        // auto duration_convert_dim = std::chrono::duration_cast<std::chrono::microseconds>(end_convert_dim - start_convert_dim).count();
        
        // auto start_compute_large_box_optim = std::chrono::high_resolution_clock::now();
        if (nlocal > 0) { //If there is a vacuum layer, in a multi-core, a block of a certain core has no atoms (local atoms are 0, ghost atoms are not 0)
            nep_gpu_models[0].compute_large_box_optim(
            is_build_neighbor,
            n_all, 
            atom->nlocal,
            list->inum,
            nep_gpu_nm,
            itype_convert_map.data(),
            list->ilist,
            list->numneigh,
            firstneighbor_cpu.data(),
            position_cpu.data(),
            cpu_potential_per_atom.data(), 
            cpu_force_per_atom.data(), 
            cpu_total_virial.data());
        }
        // auto end_compute_large_box_optim = std::chrono::high_resolution_clock::now();
        // auto duration_compute_large_box_optim = std::chrono::duration_cast<std::chrono::microseconds>(end_compute_large_box_optim - start_compute_large_box_optim).count();

        // for(int tmpi=0;tmpi< 10;tmpi++) {
        //     printf("after ei [%d] = %f", tmpi, cpu_potential_per_atom[tmpi]);
        // }
        if (eflag) {
            // printf("before eng %f\n", eng_vdwl);
            double tmp_etot = 0;
            for (int i = 0; i < list->inum; ++i) {
            eng_vdwl += cpu_potential_per_atom[i];
            tmp_etot += cpu_potential_per_atom[i];
            }
            // printf("after eng %f real etot %f\n", eng_vdwl, tmp_etot);
            // std::cout<< "etot " << eng_vdwl << std::endl;
        }
        if (vflag) {
            for (int component = 0; component < 6; ++component) {
            virial[component] += cpu_total_virial[component];
            }
        }
        if (eflag_atom) {
            for (int i = 0; i < list->inum; ++i) {
                per_atom_potential[i] = cpu_potential_per_atom[i];
            }
        }
        // copy force
        for (int i = 0; i < n_all; ++i) {
            atom->f[i][0] = cpu_force_per_atom[i];
            atom->f[i][1] = cpu_force_per_atom[n_all + i];
            atom->f[i][2] = cpu_force_per_atom[2*n_all + i];
        }
    }
    else if (model_type == 2 and num_ff > 1) { // nep gpu version multi models deviation
        is_build_neighbor = (current_timestep % neighbor->every == 0);
        if (pre_nlocal != nlocal or pre_nghost != nghost) {
            is_build_neighbor = true;
            pre_nlocal = nlocal;
            pre_nghost = nghost;
        }
        int global_flag;
        int local_flag = is_build_neighbor ? 1 : 0;
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, world);
        is_build_neighbor = global_flag ? true : false;

        if (current_timestep % neighbor->every == 0) {
            is_build_neighbor = true;
        }
        
        if (cvflag_atom) {
            per_atom_virial = cvatom;
        }
        if (eflag_atom) {
            per_atom_potential = eatom;
        }
        
        for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
            if (ff_idx > 0 && (current_timestep % out_freq != 0)) continue;
            std::tie(itype_convert_map, firstneighbor_cpu, position_cpu) = convert_dim(is_build_neighbor);
            for (int i = 0; i < n_all; i++) {
            // printf("before step[%d] model[%d] f_n[%d]=%f, %f, %f, e = %f\n", current_timestep, ff_idx, i, f_n[ff_idx][i][0], f_n[ff_idx][i][1], f_n[ff_idx][i][2], e_atom_n[ff_idx][i]);
                f_n[ff_idx][i][0] = 0.0;
                f_n[ff_idx][i][1] = 0.0;
                f_n[ff_idx][i][2] = 0.0;
            }
            for (int i = 0; i < list->inum; i++) {
                e_atom_n[ff_idx][i] = 0.0;
            }
            std::vector<double> cpu_potential_per_atom(list->inum, 0.0);
            std::vector<double> cpu_force_per_atom(n_all * 3, 0.0);
            std::vector<double> cpu_total_virial(6, 0.0);
            
            if (nlocal > 0) {//If there is a vacuum layer, in a multi-core, a block of a certain core has no atoms (local atoms are 0, ghost atoms are not 0)
                nep_gpu_models[ff_idx].compute_large_box_optim(
                is_build_neighbor,
                n_all, 
                atom->nlocal,
                list->inum,
                nep_gpu_nm,
                itype_convert_map.data(),
                list->ilist,
                list->numneigh,
                firstneighbor_cpu.data(),
                position_cpu.data(),
                cpu_potential_per_atom.data(), 
                cpu_force_per_atom.data(), 
                cpu_total_virial.data());
            }

            for (int i = 0; i < list->inum; ++i) {
                e_atom_n[ff_idx][i] = cpu_potential_per_atom[i];
            }
            for (int i = 0; i < n_all; ++i) {
                f_n[ff_idx][i][0] = cpu_force_per_atom[i];
                f_n[ff_idx][i][1] = cpu_force_per_atom[n_all + i];
                f_n[ff_idx][i][2] = cpu_force_per_atom[2*n_all + i];
            }
            if (ff_idx == 0) {
                if (eflag_atom) {
                    for (int i = 0; i < list->inum; ++i) {
                        per_atom_potential[i] = cpu_potential_per_atom[i];
                    }
                }
                if (eflag) {
                    double tmp = 0;
                    for (int i = 0; i < list->inum; ++i) {
                        tmp += cpu_potential_per_atom[i];
                    }
                    eng_vdwl = tmp;
                }
                if (vflag) {
                    for (int component = 0; component < 6; ++component) {
                        virial[component] = cpu_total_virial[component];
                    }
                }
                for (int i = 0; i < n_all; ++i) {
                    f[i][0] = cpu_force_per_atom[i];
                    f[i][1] = cpu_force_per_atom[n_all + i];
                    f[i][2] = cpu_force_per_atom[2*n_all + i];
                }
            } // if  ff_idx == 0
        } // for ff_idx
    } // nep gpu version multi models deviation
    #endif 
    // for deviation of multi models
    if (num_ff > 1 && (current_timestep % out_freq == 0)) {
        // calculate model deviation with Force
        std::tuple<double, double, double, double, double, double> result = calc_max_error(f_n, e_atom_n);

        max_mean_err_out = std::get<0>(result);
        max_err = std::get<1>(result);
        min_err = std::get<2>(result);
        max_mean_ei_out = std::get<3>(result);
        max_err_ei = std::get<4>(result);
        min_err_ei = std::get<5>(result);

        // max_err = result.first;
        // max_err_ei = result.second;

        MPI_Allreduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, world);
        MPI_Allreduce(&min_err, &global_min_err, 1, MPI_DOUBLE, MPI_MIN, world);
        MPI_Allreduce(&max_mean_err_out, &global_max_mean_err, 1, MPI_DOUBLE, MPI_MAX, world);

        MPI_Allreduce(&max_err_ei, &global_max_err_ei, 1, MPI_DOUBLE, MPI_MAX, world);
        MPI_Allreduce(&min_err_ei, &global_min_err_ei, 1, MPI_DOUBLE, MPI_MIN, world);
        MPI_Allreduce(&max_mean_ei_out, &global_max_mean_err_ei, 1, MPI_DOUBLE, MPI_MAX, world);

        max_err_list.push_back(global_max_err);
        max_err_ei_list.push_back(global_max_err_ei);

        if (current_timestep % out_freq == 0) {
            if (me == 0) {
                // fprintf(explrError_fp, "%9d %16.9f %16.9f\n", (max_err_list.size()-1)*out_freq, global_max_err, global_max_err_ei);
                fprintf(explrError_fp, "%9d %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f\n", 
                            current_timestep, global_max_mean_err, global_min_err, global_max_err, 
                                global_max_mean_err_ei, global_min_err_ei, global_max_err_ei);
                fflush(explrError_fp);
            } 
        }
    }
    
    // std::cout << "t4 " << (t5 - t4).count() * 0.000001 << "\tms" << std::endl;
    // std::cout << "t5 " << (t6 - t5).count() * 0.000001 << "\tms" << std::endl;
    // std::cout << "t6 " << (t7 - t6).count() * 0.000001 << "\tms" << std::endl;
}
