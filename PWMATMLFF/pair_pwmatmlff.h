/* -*- c++ -*- ----------------------------------------------------------
    liuliping, PWmat-MLFF to LAMMPS
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pwmatmlff,PairPWMATMLFF);
// clang-format on
#else

#ifndef LMP_PAIR_PWMATMLFF_H
#define LMP_PAIR_PWMATMLFF_H

#include "pair.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unordered_map>

using namespace std;

typedef unsigned long long ull; 

namespace LAMMPS_NS {

class atom_config
{
    /*
        a single atom.config 
    */
  public:
    int num_atom; 
    vector<double> lattice; 
    vector<double> position;
    vector<int> type;
    
    atom_config(int num)
    {
      num_atom = num; 
      lattice.resize(9);
      position.resize(3*num_atom);
      type.resize(num_atom);
    }
    
    void get_all(double* latt, double** x , int* itype_atom)
    {
      for (int i=0; i< 9; i++)
        lattice[i] = latt[i];
      
      for (int i=0; i< num_atom; i++) {
        position[i*3] = x[i][0];
        position[i*3+1] = x[i][1];
        position[i*3+2] = x[i][2]; 
        type[i] = itype_atom[i]; 
      }
    }
};


class PairPWMATMLFF : public Pair
{
  public:
    /*
        1) arrays that will be passed into fortran are all consequetive chunks.  
           get_fort_idx_xd functions take care of the indices, so that elements are accessed
           in the fortran style

        2) use STL containers smartly. Less stress on memory management

        3) 
    */
    
    PairPWMATMLFF(class LAMMPS *);
    ~PairPWMATMLFF() override;
    
    int nmax;
    int me;

    double** cut;
 
    double*** f_n;
    double** e_atom_n;
    int* itype_atom;

    double* e_atom;
    int** num_neigh;
    int*** list_neigh;
    double**** dR_neigh;

    double lattice[9];

    int imodel;
    int num_ff;
    int type_map[20];
    int max_neigh_num;

    int num_fail;
    int num_cand;
    int num_success; 

    double fail_err;
    double candidate_err; 

    int old_timestep;
    double min_dR;

    int p_ff_idx;

    unsigned seed;
    
    vector<double> max_err_list; 
    vector<atom_config> atom_config_list; 
    vector<int> config_id;

    string explrError_fname;
    FILE *explrError_fp;
    
    // major member functions 
    void compute(int, int) override;
    virtual void allocate();
    void settings(int, char **) override;
    void coeff(int, char **) override;
    void init_style() override;
    double init_one(int, int) override;
    int pack_reverse_comm(int, int, double* ) override;
    void unpack_reverse_comm(int, int*, double* ) override;

    void grow_memory();
    void generate_neighdata();
    double calc_max_f_err(double***);
    void write_config(class atom_config& , int , int timestep);
    void write_info();
    
};

}    // namespace LAMMPS_NS

#endif
#endif
