/*
    Interface to PWMATMLFF
    2023.1
*/

#include "pair_pwmatmlff.h"

//#include "diy.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
//#include "respa.h"
#include "domain.h"
#include "update.h"

#include <cmath>
#include <cstring>
// tmp
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;

// declration of f2c!

extern "C" {
void f2c_calc_energy_force(int * /*imodel*/, int * /*nlocal*/, int * /*nlocal + nghost*/,
                           int * /*ntypes*/, int * /*type*/, int * /*itype*/, int * /*num_neigh*/,
                           int * /*list_neigh*/, double * /*dR_neigh*/, double * /*e_atom*/,
                           double * /*e_tot*/, double * /*f*/, double * /*virial*/,
                           int * /*ff_idx*/, int * /*max_neigh_num*/);
// ff data loading
void dp_ff_load(char * /*file name*/, int * /*ff index */, int * /*file name length*/,
                double * /*neighbor cutoff*/, int * /*max_neigh_num*/);

void li_ff_load(char * /*file name*/, int * /*ff index */, int * /*file name length*/,
                double * /*neighbor cutoff*/, int * /*max_neigh_num*/);

void nn_ff_load(char * /*file name*/, int * /*ff index */, int * /*file name length*/,
                double * /*neighbor cutoff*/, int * /*max_neigh_num*/);

// ff data destroy
// pointer of the name string
void dp_ff_deallocate(int *);
void li_ff_deallocate(int *);
void nn_ff_deallocate(int *);
}

// Function to reorder type_map based on the order of appearance of elements in type_map[type[i] - 1]
void reorderTypeMap(int *type_map, int *type, int ntypes, int nall)
{
  // Vectors to store the unique elements from type_map and type_map[type[i] - 1]
  std::set<int> uniqueValues;    // Using std::set to record appeared values
  static int original_order_initialized = 0;    // Variable to check if original_order has been initialized
  // static int original_order[3];    // To store the original order of type_map for comparison,静态存储期的数组大小必须是常量。
  static std::vector<int>original_order;        // To store the original order of type_map for comparison,动态分配创建
  int type_order[ntypes];    // ntypes is the maximum possible number of unique values
  int ttt[ntypes];           // tmp variable, ntypes is the maximum possible number of unique values

  // Initialize original_order only once
  if (original_order_initialized == 0) {
    // for (int i = 0; i < ntypes; i++) {
    // original_order[i] = type_map[i];
    // }
    original_order.assign(type_map, type_map + ntypes);
    original_order_initialized = 1;
  }

  // Extract unique elements from type_map[type[i] - 1]
  int unique_count = 0;
  for (int i = 0; i < nall; i++) {
    int value = type_map[type[i] - 1];
    // printf("@@@%5d%5d%5d%5d\n", i, type[i], value, type_map[type[i] - 1]);
    if (uniqueValues.find(value) ==
        uniqueValues.end()) {              // If this value has not appeared before
      uniqueValues.insert(value);          // Add this value to the set
      type_order[unique_count] = value;    // Assign the unique value to type_order
      ttt[type[i] - 1] =
          original_order[unique_count];    // Assign the unique value from original order to ttt
      unique_count++;
    }
  }
  // for (int i = 0; i < ntypes; i++) {
  //   printf("@@@%5d\n", type_order[i]);
  // }

  // Check if the order in type_order matches with the original_order
  bool order_matches = true;
  for (int i = 0; i < unique_count; i++) {
    if (type_order[i] != original_order[i]) {
      order_matches = false;
      break;
    }
  }

  // Update type_map based on ttt if the order doesn't match
  if (!order_matches) {
    for (int i = 0; i < ntypes; i++) { type_map[i] = ttt[i]; }
  }

}

PairPWMATMLFF::PairPWMATMLFF(LAMMPS *lmp) : Pair(lmp)
{
  nmax = 0;
  me = comm->me;
  comm_reverse = 3;
  writedata = 1;

  imodel = 0;
  max_neigh_num = 100;
  num_fail = 0;
  num_cand = 0;
  num_success = 0;
  fail_err = 0.25;
  candidate_err = 0.05;

  old_timestep = -1;
  min_dR = 1000;

  seed = 1745294809;

  if (me == 0) {
    explrError_fname = "explr.error";
    explrError_fp = fopen(&explrError_fname[0], "w");
  }
}

PairPWMATMLFF::~PairPWMATMLFF()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cut);
    memory->destroy(cutsq);
    memory->destroy(e_atom);
    memory->destroy(num_neigh);
    memory->destroy(list_neigh);
    memory->destroy(dR_neigh);

    memory->destroy(itype_atom);
    memory->destroy(e_atom_n);
    memory->destroy(f_n);
  }

  // ff paras memory
  for (int ff_idx = 1; ff_idx <= num_ff; ff_idx++) {
    if (me == 0) printf("!!!Releasing force field memory %7d\n", ff_idx);
    if (imodel == 1) li_ff_deallocate(&ff_idx);
    if (imodel == 3) nn_ff_deallocate(&ff_idx);
    if (imodel == 5) dp_ff_deallocate(&ff_idx);
    if (me == 0) printf("!!!Force field memory released\n");
  }

  if (me == 0) fclose(explrError_fp);
}

void PairPWMATMLFF::compute(int eflag, int vflag)
{
  int i, j;
  int ff_idx, tff_idx;
  double max_err, global_max_err;
  int current_timestep;
  double e_tot, e_tot_tmp;
  double virial_tmp[6];

  double *h = domain->h;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int ntypes = atom->ntypes;
  double *virial = force->pair->virial;

  current_timestep = update->ntimestep;

  grow_memory();

  ev_init(eflag, vflag);

  // get lattice
  lattice[0] = h[0];    // xx
  lattice[4] = h[1];    // yy
  lattice[8] = h[2];    // zz
  lattice[7] = h[3];    // yz
  lattice[6] = h[4];    // xz
  lattice[3] = h[5];    // xy

  // Reorder type_map based on the order of appearance of elements in type_map[type[i] - 1]
  reorderTypeMap(type_map, type, ntypes, nall);
  for (i = 0; i < nall; i++) {
    // atom->type starts from 1,
    // and type_map index starts from 0;
    itype_atom[i] = type_map[type[i] - 1];
    // printf("%5d,%5d,%5d,%5d\n", i, type[i], type_map[type[i] - 1], itype_atom[i]);
  }

  generate_neighdata();

  // call fortran subroutine energy force
  /*
      do calc_energy_force for 4 models
      1 is used for MD
      2, 3, 4 for the test
  */

  for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
    // fortran index is start from 1.
    tff_idx = ff_idx + 1;
    f2c_calc_energy_force(&imodel, &nlocal, &nall, &ntypes, &type[0], &itype_atom[0],
                          &num_neigh[0][0], &list_neigh[0][0][0], &dR_neigh[0][0][0][0],
                          &e_atom_n[ff_idx][0], &e_tot_tmp, &f_n[ff_idx][0][0], &virial_tmp[0],
                          &tff_idx, &max_neigh_num);
    if (ff_idx == 0) {
      e_tot = e_tot_tmp;
      for (i = 0; i < nall; i++)
        for (j = 0; j < 3; j++) f[i][j] = f_n[ff_idx][i][j];
      for (j = 0; j < 6; j++) virial[j] = virial_tmp[j];
    }
  }

  /*
      exploration mode.
      select candidates
      write as fractional coordinate
      Note: only write data at the very end (in the destructor)!
  */

  if (num_ff > 1) {
    // calculate model deviation with Force
    max_err = calc_max_f_err(f_n);

    MPI_Allreduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, world);

    max_err_list.push_back(global_max_err);

    if (me == 0) {
      fprintf(explrError_fp, "%9d %16.9f\n", max_err_list.size(), global_max_err);
      fflush(explrError_fp);
    }

    //$ for maximum effect explor configuration,
    //  the chose save structure should not too similar
    //$if ( (global_max_err > candidate_err &&
    //$      global_max_err < fail_err) &&
    //$     ((current_timestep - old_timestep) > 50)  ) {
    //$  old_timestep = current_timestep;
    if ((candidate_err < global_max_err) && (global_max_err < fail_err)) {
      num_cand++;
      atom_config tmp(nlocal);

      domain->x2lamda(nlocal);
      tmp.get_all(lattice, x, itype_atom);
      domain->lamda2x(nlocal);

      atom_config_list.push_back(tmp);
      config_id.push_back(current_timestep);
      //save image for SCF;
    } else if (max_err > fail_err) {
      num_fail++;
    } else if (max_err < candidate_err) {
      num_success++;
    }
  }

  // potential energy
  if (eflag_global) eng_vdwl += e_tot;

  if (update->ntimestep == update->laststep) write_info();
}

void PairPWMATMLFF::allocate()
{
  allocated = 1;

  int ntypes = atom->ntypes;
  int natoms = atom->natoms;
  int n = ntypes + 1;

  memory->create(setflag, n, n, "pair_pwmatmlff:setflag");

  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(cut, n, n, "pair_pwmatmlff:cut");
  memory->create(cutsq, n, n, "pair_pwmatmlff:cutsq");

  memory->create(e_atom, natoms, "pair_pwmatmlff:e_atom");

  memory->create(num_neigh, natoms, ntypes, "pair_pwmatmlff:num_neigh");
  memory->create(list_neigh, natoms, ntypes, max_neigh_num, "pair_pwmatmlff:list_neigh");
  memory->create(dR_neigh, natoms, ntypes, max_neigh_num, 3, "pair_pwmatmlff:dR_neigh");

  atom_config_list.reserve(500);
  config_id.reserve(500);
}

// useless but we need it present
void PairPWMATMLFF::settings(int narg, char **arg) {}

void PairPWMATMLFF::coeff(int narg, char **arg)
{
  // if (!allocated) allocate();

  int i;
  int ff_idx, tff_idx;
  int name_len;
  int ntypes = atom->ntypes;
  double temp_cut;

  temp_cut = 0.0;

  // read in nn model type and nn model number
  imodel = utils::inumeric(FLERR, arg[2], false, lmp);

  // if (imodel != 5)
  //   error->all(FLERR, "pair_pwmatmlff only support imodel=5");

  num_ff = utils::inumeric(FLERR, arg[3], false, lmp);

  if (((num_ff == 1) && (narg < (2 + 2 + num_ff + ntypes)))) {
    error->all(FLERR, "pair_coeff error:\
         * * imodel num_mlff 1.ff 2.ff ... \
             elem1 elem2 ... cut1 cut2 ...");
  } else if (((num_ff > 1) && (narg != (2 + 2 + num_ff + ntypes + 2)))) {
    error->all(FLERR, "pair_coeff error:\
         * * imodel num_mlff 1.ff 2.ff ... \
             elem1 elem2 ... \
             fail_err candidate_err");
  }

  // read in nn model file name
  for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {

    name_len = 0;
    for (i = 0; i < 500; i++) {
      if (arg[2 + 2 + ff_idx][i] == '\0') break;
      name_len++;
    }

    if (comm->me == 0)
      printf("\nStart loading force field: \n\n%s %d\n\n", arg[2 + 2 + ff_idx], name_len);

    // fortran index is start from 1
    tff_idx = ff_idx + 1;
    if (imodel == 1) li_ff_load(arg[2 + 2 + ff_idx], &tff_idx, &name_len, &temp_cut, &max_neigh_num);
    if (imodel == 3) nn_ff_load(arg[2 + 2 + ff_idx], &tff_idx, &name_len, &temp_cut, &max_neigh_num);
    if (imodel == 5) dp_ff_load(arg[2 + 2 + ff_idx], &tff_idx, &name_len, &temp_cut, &max_neigh_num);

    if (comm->me == 0) printf("Force field loaded successfully\n\n");
  }

  if (!allocated) allocate();
  // printf("@@@ allocate %7d %7d\n", imodel, max_neigh_num);
  // read in periodic table index of each atom type
  for (int i = 0; i < ntypes; i++) {
    type_map[i] = utils::inumeric(FLERR, arg[i + 2 + 2 + num_ff], false, lmp);
    // printf("@@@%5d%5d%5d%5d\n", i, type[i], type_map[type[i] - 1], type_map[i]);
  }

  // read in the neighbor cutoff for each atom type
  for (int i = 1; i <= ntypes; i++) {
    setflag[i][i] = 1;
    cut[i][i] = temp_cut;
  }

  // read in candidate error and fail error
  if (narg == (2 + 2 + num_ff + ntypes + 2)) {
    candidate_err = utils::numeric(FLERR, arg[narg - 2], false, lmp);
    fail_err = utils::numeric(FLERR, arg[narg - 1], false, lmp);
  }

  // since we need num_ff, so well allocate memory here
  // but not in allocate()
  nmax = atom->nmax;
  memory->create(f_n, num_ff, nmax, 3, "pair_pwmatmlff:f_n");
  memory->create(e_atom_n, num_ff, atom->natoms, "pair_pwmatmlff:e_atom_n");
  memory->create(itype_atom, nmax, "pair_pwmatmlff:itype_atom");

  // print information for check
  if (comm->me == 0) {
    printf("\n\n --- Read in pair Coeff --- \n\n");
    printf("  imodel %9d\n", imodel);
    printf("  num_ff %9d\n", num_ff);
    printf("  atom type ");
    for (int i = 0; i < ntypes; i++) printf(" %5d", type_map[i]);

    printf("\n  atom cut ");
    for (int i = 1; i <= ntypes; i++) printf(" %12.6f", cut[i][i]);
    printf("\n\n --- Read in pair Coeff end --- \n\n");
  }
}

void PairPWMATMLFF::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "Pair style PWMATMLFF requires newon pair on");

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairPWMATMLFF::init_one(int i, int j)
{
  if (setflag[i][j] == 0) { cut[i][j] = mix_distance(cut[i][i], cut[j][j]); }
  return cut[i][j];
}

int PairPWMATMLFF::pack_reverse_comm(int n, int first, double *buf)
{
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

void PairPWMATMLFF::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f_n[p_ff_idx][j][0] += buf[m++];
    f_n[p_ff_idx][j][1] += buf[m++];
    f_n[p_ff_idx][j][2] += buf[m++];
  }
}

void PairPWMATMLFF::grow_memory()
{
  if (atom->nmax > nmax) {
    printf("@@@ allocate new %7d %7d %7d\n", update->ntimestep, nmax, atom->nmax);
    nmax = atom->nmax;
    memory->grow(f_n, num_ff, nmax, 3, "pair_pwmatmlff:f_n");
    memory->grow(itype_atom, nmax, "pair_pwmatmlff:itype_atom");
  }
}

void PairPWMATMLFF::generate_neighdata()
{
  int i, j, k, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int etnum;

  double **x = atom->x;
  int *type = atom->type;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int ntypes = atom->ntypes;

  double min_dR_all;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (i = 0; i < nlocal; i++) {
    for (j = 0; j < ntypes; j++) {
      num_neigh[i][j] = 0;
      for (k = 0; k < max_neigh_num; k++) {
        list_neigh[i][j][k] = 0;
        dR_neigh[i][j][k][0] = 0.0;
        dR_neigh[i][j][k][1] = 0.0;
        dR_neigh[i][j][k][2] = 0.0;
      }
    }
  }

  min_dR = 1000;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][itype]) {
        etnum = num_neigh[i][jtype - 1];
        dR_neigh[i][jtype - 1][etnum][0] = delx;
        dR_neigh[i][jtype - 1][etnum][1] = dely;
        dR_neigh[i][jtype - 1][etnum][2] = delz;
        // fortran start from 1
        list_neigh[i][jtype - 1][etnum] = j + 1;
        num_neigh[i][jtype - 1] += 1;
        if (rsq < min_dR) min_dR = rsq;
      }
    }
  }

  MPI_Allreduce(&min_dR, &min_dR_all, 1, MPI_DOUBLE, MPI_MIN, world);

  if (min_dR_all < 0.81) {
    if (me == 0) printf("@@@ there are two atom too close, min_dR_all = %12.6f\n", min_dR_all);
    error->universe_all(FLERR, "@@@ pair_pwmatmlff min_dR_all.");
  }

  //printf(" --- CHECK gn --- \n");
  //for (i = 0; i < nlocal; i++) {
  //  printf("i   %7d\n", i);
  //  for (int t = 0; t < ntypes; t++) {
  //    printf("  t   %7d %7d\n", t, num_neigh[i][t]);
  //    for (j = 0; j < num_neigh[i][t]; j++)
  //      printf("    %7d %9.6f %9.6f %9.6f \n", list_neigh[i][t][j], \
  //        dR_neigh[i][t][j][0], dR_neigh[i][t][j][1], dR_neigh[i][t][j][2]);
  //  }
  //}
}

double PairPWMATMLFF::calc_max_f_err(double ***f_n)
{
  int i, j;
  double max_err, err;
  int ff_idx;
  double num_ff_inv;
  int nlocal = atom->nlocal;

  max_err = -1.0;
  num_ff_inv = 1.0 / num_ff;

  for (ff_idx = 0; ff_idx < num_ff; ff_idx++) {
    // p_ff_idx is for reverse comm
    p_ff_idx = ff_idx;
    comm->reverse_comm(this);
  }

  vector<double> f_ave;
  vector<double> f_error[num_ff];

  f_ave.resize(3 * nlocal);

  for (i = 0; i < num_ff; i++) f_error[i].resize(3 * nlocal);

  // sum over all models
  for (ff_idx = 0; ff_idx < num_ff; ff_idx++)
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < 3; j++) f_ave[i * 3 + j] += f_n[ff_idx][i][j];

  // calc ensemble average
  for (i = 0; i < 3 * nlocal; i++) f_ave[i] *= num_ff_inv;

  // clac error
  for (ff_idx = 0; ff_idx < num_ff; ff_idx++)
    for (i = 0; i < nlocal; i++)
      for (j = 0; j < 3; j++) f_error[ff_idx][i * 3 + j] = f_n[ff_idx][i][j] - f_ave[i * 3 + j];

  // find max error
  for (ff_idx = 0; ff_idx < num_ff; ff_idx++)
    for (j = 0; j < 3 * nlocal; j += 3) {
      err = f_error[ff_idx][j] * f_error[ff_idx][j] +
          f_error[ff_idx][j + 1] * f_error[ff_idx][j + 1] +
          f_error[ff_idx][j + 2] * f_error[ff_idx][j + 2];

      err = sqrt(err);

      if (err > max_err) max_err = err;
    }

  return max_err;
}

void PairPWMATMLFF::write_config(atom_config &config, int idx, int timestep)
{
  MPI_Status status;
  MPI_Request request;
  int nprocs, iproc;

  FILE *fp;
  double *buf;
  int maxbuf;
  int tmp, nlines;
  int nme, tnmax;
  int i, m;

  tmp = nme = tnmax = 0;

  nprocs = comm->nprocs;
  nme = config.num_atom;

  MPI_Allreduce(&nme, &tnmax, 1, MPI_INT, MPI_MAX, world);

  memory->create(buf, tnmax * 4, "pair_pwmatlff:write()");

  string fname = "cand_" + to_string(idx) + ".config";

  if (me == 0) fp = fopen(&fname[0], "w");

  // write header
  if (me == 0) {
    //fprintf(fp,"%7d %9d\n", atom->natoms, timestep);
    fprintf(fp, "%7d\n", atom->natoms);
    fprintf(fp, "Lattice vector\n");
    for (i = 0; i < 9; i++) {
      fprintf(fp, "%16.9f ", config.lattice[i]);
      if ((i + 1) % 3 == 0) fprintf(fp, "\n");
    }
    fprintf(fp, "Position, move_x, move_y, move_z\n");
  }

  // save data to buf
  m = 0;
  for (i = 0; i < nme; i++) {
    buf[m++] = config.type[i];
    buf[m++] = config.position[i * 3];
    buf[m++] = config.position[i * 3 + 1];
    buf[m++] = config.position[i * 3 + 2];
  }

  // write coordinate
  if (me == 0) {
    for (iproc = 0; iproc < nprocs; iproc++) {
      if (iproc) {
        MPI_Irecv(buf, tnmax * 4, MPI_DOUBLE, me + iproc, 0, world, &request);
        MPI_Send(&tmp, 0, MPI_INT, me + iproc, 0, world);
        MPI_Wait(&request, &status);
        MPI_Get_count(&status, MPI_DOUBLE, &nlines);
        nlines /= 4;
      } else
        nlines = nme;

      m = 0;
      for (i = 0; i < nlines; i++) {
        fprintf(fp, "%7d %13.12f %13.12f %13.12f 1 1 1\n", static_cast<int>(buf[m]), buf[m + 1],
                buf[m + 2], buf[m + 3]);
        m += 4;
      }
    }
    fflush(fp);
  } else {
    MPI_Recv(&tmp, 0, MPI_INT, 0, 0, world, MPI_STATUS_IGNORE);
    MPI_Rsend(buf, nme * 4, MPI_DOUBLE, 0, 0, world);
  }

  if (me == 0) fclose(fp);

  memory->destroy(buf);
}

void PairPWMATMLFF::write_info()
{
  if (num_ff > 1) {
    int num_select;
    unsigned seed, all_seed;
    int i;

    if (me == 0) {

      FILE *fp;

      /* dump stat file */
      string fname = "explr.stat";
      fp = fopen(&fname[0], "w");

      fprintf(fp, "num_succes num_cand num_fail\n");
      fprintf(fp, "%9d %9d %9d \n", num_success, num_cand, num_fail);
      fclose(fp);
    }

    /* pick out 10 config for later selection */
    num_select = MIN(10, num_cand);

    if (num_select > 0) {
      vector<int> idx(num_cand, 0);

      for (i = 0; i < num_cand; i++) idx[i] = i;

      shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));

      for (i = 0; i < num_select; i++) {
        /*
            write candidates' configs
            RANDOMLY CHOOSE 10 CONFIGS
            do padding if num_cand < num_select
        */
        if (i < num_cand) {
          write_config(atom_config_list[idx[i]], i, config_id[idx[i]]);
        } else {
          write_config(atom_config_list[idx[num_cand - 1]], i, config_id[idx[num_cand - 1]]);
        }
      }
    }
  }
}
