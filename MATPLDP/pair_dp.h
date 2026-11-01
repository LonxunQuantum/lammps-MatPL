/* -*- c++ -*- ----------------------------------------------------------
   MatPL TorchScript DP interface
------------------------------------------------------------------------- */
#ifdef PAIR_CLASS
PairStyle(matpl/dp, PairDP);
#else
#ifndef LMP_PAIR_MATPLDP_H
#define LMP_PAIR_MATPLDP_H
#include "pair.h"
#include "atom_type_mapping.h"
#include <torch/script.h>
#include <cstdio>
#include <tuple>
#include <vector>
namespace LAMMPS_NS {
class PairDP : public Pair {
 public:
  PairDP(class LAMMPS *);
  ~PairDP() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  void allocate();

 private:
  struct ModelData {
    std::vector<int> atomic_numbers;
    MatPL::AtomTypeMapping mapping;
    double cutoff;
    int max_neighbor;
  };
  struct PackedData {
    int nlocal = 0;
    int nghost = 0;
    std::vector<int> original_to_packed, packed_to_original;
    std::vector<int> center_types, neighbors;
    std::vector<double> displacements;
  };
  PackedData generate_neighdata(int);
  void add_peratom_virial(const PackedData &, int, const torch::Tensor &);
  void grow_memory(int);
  std::tuple<double, double, double, double, double, double> calc_max_error();
  int me;
  int num_ff = 0;
  int p_ff_idx = 0;
  int nmax = 0;
  double ***f_n = nullptr;
  double **e_atom_n = nullptr;
  std::vector<torch::jit::script::Module> modules;
  std::vector<ModelData> model_data;
  // Unique configured active atomic numbers, independent of rank/population.
  std::vector<int> active_species;
  torch::Device device = torch::kCPU;
  double cutoff = 0.0;
  std::string explrError_fname = "explr.error";
  std::FILE *explrError_fp = nullptr;
  int out_freq = 1;
};
}
#endif
#endif
