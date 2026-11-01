#include "pair_dp.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include <torch/torch.h>
#include <torch/csrc/autograd/autograd.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

using namespace LAMMPS_NS;

PairDP::PairDP(LAMMPS *lmp) : Pair(lmp)
{
  me = comm->me;
  writedata = 1;
  comm_reverse = 3;
  restartinfo = 0;
  manybody_flag = 1;
  single_enable = 0;
  one_coeff = 1;
  no_virial_fdotr_compute = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
}

PairDP::~PairDP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  memory->destroy(f_n);
  memory->destroy(e_atom_n);
  if (explrError_fp) fclose(explrError_fp);
}

void PairDP::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair_dp:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair_dp:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 0;
}

void PairDP::settings(int narg, char **arg)
{
  if (!narg) error->all(FLERR, "Pair style matpl/dp requires at least one model");
  memory->destroy(f_n);
  memory->destroy(e_atom_n);
  nmax = 0;
  modules.clear();
  model_data.clear();
  active_species.clear();
  cutoff = 0.0;
  if (explrError_fp) { fclose(explrError_fp); explrError_fp = nullptr; }
  out_freq = 1;
  explrError_fname = "explr.error";
  std::vector<std::string> filenames;
  int iarg = 0;
  while (iarg < narg) {
    const std::string name(arg[iarg]);
    if (name.find(".pt") == std::string::npos && name.find(".txt") == std::string::npos) break;
    filenames.push_back(name);
    ++iarg;
  }
  if (filenames.empty()) error->all(FLERR, "Pair style matpl/dp requires at least one model");
  while (iarg < narg) {
    if (iarg + 1 >= narg) error->all(FLERR, "Missing value in matpl/dp pair_style settings");
    if (strcmp(arg[iarg], "out_freq") == 0) {
      out_freq = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (out_freq <= 0) error->all(FLERR, "matpl/dp out_freq must be positive");
    } else if (strcmp(arg[iarg], "out_file") == 0) {
      explrError_fname = arg[iarg + 1];
    } else error->all(FLERR, "Unknown matpl/dp pair_style option: " + std::string(arg[iarg]));
    iarg += 2;
  }
  const int devices = torch::cuda::device_count();
  device = torch::cuda::is_available() && devices > 0 ?
      torch::Device(torch::kCUDA, me % devices) : torch::Device(torch::kCPU);
  for (const auto &filename : filenames) {
    try {
      auto model = torch::jit::load(filename, device);
      model.to(torch::kFloat64);
      if (model.hasattr("model_type") && model.attr("model_type").toStringRef() != "DP")
        throw std::invalid_argument("model_type must be DP");
      ModelData data;
      data.cutoff = model.attr("Rmax").toDouble();
      const auto maximum = model.attr("maxNeighborNum").toInt();
      const auto species = model.attr("atom_type").toList();
      if (!std::isfinite(data.cutoff) || data.cutoff <= 0.0 || maximum <= 0 ||
          maximum > std::numeric_limits<int>::max() || species.size() == 0 ||
          species.size() > static_cast<size_t>(std::numeric_limits<int>::max() / maximum))
        throw std::invalid_argument("Rmax, maxNeighborNum and atom_type must be valid");
      data.max_neighbor = static_cast<int>(maximum);
      for (size_t t = 0; t < species.size(); ++t) {
        const auto z = species.get(t).toInt();
        if (z < 1 || z > 94 || std::find(data.atomic_numbers.begin(), data.atomic_numbers.end(), z) != data.atomic_numbers.end())
          throw std::invalid_argument("atom_type must contain unique supported atomic numbers");
        data.atomic_numbers.push_back(static_cast<int>(z));
      }
      cutoff = std::max(cutoff, data.cutoff);
      modules.push_back(std::move(model));
      model_data.push_back(std::move(data));
    } catch (const std::exception &e) {
      error->all(FLERR, "Invalid DP model metadata or load failure in " + filename + ": " + e.what());
    }
  }
  num_ff = static_cast<int>(modules.size());
  if (me == 0) {
    utils::logmesg(lmp, "Loaded {} DP model(s) on {}, union cutoff {}\n", num_ff,
                  device.is_cuda() ? "GPU" : "CPU", cutoff);
    if (num_ff > 1) {
      explrError_fp = fopen(explrError_fname.c_str(), "w");
      if (!explrError_fp) error->one(FLERR, "Cannot open DP model deviation output: " + explrError_fname);
      fprintf(explrError_fp, "%9s %16s %16s %16s %16s %16s %16s\n", "#    step",
              "avg_devi_f", "min_devi_f", "max_devi_f", "avg_devi_e", "min_devi_e", "max_devi_e");
      fflush(explrError_fp);
    }
  }
}

void PairDP::coeff(int narg, char **arg)
{
  if (narg != atom->ntypes + 2)
    error->all(FLERR, "Expected one mapping per LAMMPS atom type in matpl/dp pair_coeff");
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  if (ilo != 1 || ihi != atom->ntypes || jlo != 1 || jhi != atom->ntypes)
    error->all(FLERR, "Pair matpl/dp coefficients require the full * * type range");
  std::vector<std::string> symbols;
  for (int i = 2; i < narg; ++i) symbols.emplace_back(arg[i]);
  std::vector<MatPL::AtomTypeMapping> mappings;
  try {
    for (const auto &data : model_data)
      mappings.push_back(MatPL::build_atom_type_mapping(atom->ntypes, symbols, data.atomic_numbers));
    MatPL::validate_active_masks(mappings);
  } catch (const std::exception &e) {
    error->all(FLERR, "DP atom type mapping: " + std::string(e.what()));
  }
  if (mappings.empty()) error->all(FLERR, "DP models must be loaded before pair_coeff");
  active_species.clear();
  for (int z : mappings[0].atomic_number)
    if (z && std::find(active_species.begin(), active_species.end(), z) == active_species.end())
      active_species.push_back(z);
  for (int i = 0; i < num_ff; ++i) model_data[i].mapping = std::move(mappings[i]);
  if (!allocated) allocate();
  // Register every pair independently of activity: hybrid dispatch needs these flags.
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) setflag[i][j] = 1;
}

double PairDP::init_one(int, int) { return cutoff; }

void PairDP::init_style()
{
  if (!force->newton_pair) error->all(FLERR, "Pair style matpl/dp requires newton pair on");
  if (active_species.empty()) error->all(FLERR, "Pair style matpl/dp requires atom type mappings");
  neighbor->add_request(this, NeighConst::REQ_FULL);
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = 1; j <= atom->ntypes; ++j) cutsq[i][j] = cutoff * cutoff;
}

PairDP::PackedData PairDP::generate_neighdata(int model_index)
{
  const auto &data = model_data[model_index];
  const auto &mapping = data.mapping;
  const int nlocal = atom->nlocal, nall = nlocal + atom->nghost;
  const int width = static_cast<int>(data.atomic_numbers.size()) * data.max_neighbor;
  // Hybrid skip lists omit entire centers. Full mapping requires every active
  // owned center to be covered before any firstneigh/numneigh access.
  if (list->inum != nlocal) {
    std::vector<unsigned char> listed(nlocal, 0);
    for (int ii = 0; ii < list->inum; ++ii) {
      const int i = list->ilist[ii];
      if (i >= 0 && i < nlocal) listed[i] = 1;
    }
    for (int i = 0; i < nlocal; ++i)
      if (mapping.active[atom->type[i] - 1] && !listed[i])
        error->one(FLERR, "DP active center missing from neighbor list; hybrid assignments must include every active DP atom type");
  }
  PackedData packed;
  packed.original_to_packed.assign(nall, -1);
  // TorchScript concatenates Ei by the supplied configured species order.
  for (int z : active_species)
    for (int i = 0; i < nlocal; ++i)
      if (mapping.atomic_number[atom->type[i] - 1] == z) {
        packed.original_to_packed[i] = static_cast<int>(packed.packed_to_original.size());
        packed.packed_to_original.push_back(i);
        packed.center_types.push_back(mapping.model_index[atom->type[i] - 1]);
      }
  packed.nlocal = static_cast<int>(packed.packed_to_original.size());
  for (int i = nlocal; i < nall; ++i)
    if (mapping.active[atom->type[i] - 1]) {
      packed.original_to_packed[i] = static_cast<int>(packed.packed_to_original.size());
      packed.packed_to_original.push_back(i);
    }
  packed.nghost = static_cast<int>(packed.packed_to_original.size()) - packed.nlocal;
  const size_t edges = static_cast<size_t>(packed.nlocal) * width;
  // Padding must be zero on every call, including after migration or population changes.
  packed.neighbors.assign(edges, 0);
  packed.displacements.assign(edges * 4, 0.0);
  double min_rsq = std::numeric_limits<double>::infinity();
  for (int center = 0; center < packed.nlocal; ++center) {
    const int i = packed.packed_to_original[center];
    std::vector<int> counts(data.atomic_numbers.size(), 0);
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      const int j = list->firstneigh[i][jj] & NEIGHMASK;
      // Mask before distances: even a coincident NULL atom is outside the model.
      if (!mapping.active[atom->type[j] - 1]) continue;
      const double dx = atom->x[j][0] - atom->x[i][0];
      const double dy = atom->x[j][1] - atom->x[i][1];
      const double dz = atom->x[j][2] - atom->x[i][2];
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq > data.cutoff * data.cutoff) continue;
      const int species = mapping.model_index[atom->type[j] - 1];
      if (counts[species] >= data.max_neighbor)
        error->one(FLERR, "DP neighbor capacity exceeded for model " + std::to_string(model_index) +
                   ", atom " + std::to_string(atom->tag[i]) + ", species " + std::to_string(data.atomic_numbers[species]));
      const size_t edge = static_cast<size_t>(center) * width + species * data.max_neighbor + counts[species]++;
      packed.neighbors[edge] = packed.original_to_packed[j] + 1;
      packed.displacements[edge * 4] = std::sqrt(rsq);
      packed.displacements[edge * 4 + 1] = dx;
      packed.displacements[edge * 4 + 2] = dy;
      packed.displacements[edge * 4 + 3] = dz;
      min_rsq = std::min(min_rsq, rsq);
    }
  }
  double global_min;
  MPI_Allreduce(&min_rsq, &global_min, 1, MPI_DOUBLE, MPI_MIN, world);
  if (global_min < 0.61) error->all(FLERR, "DP: there are two active atoms too close");
  return packed;
}

void PairDP::add_peratom_virial(const PackedData &packed, int width, const torch::Tensor &gradient)
{
  const double *g = gradient.data_ptr<double>();
  const int a[6] = {0, 1, 2, 0, 0, 1};
  const int b[6] = {0, 1, 2, 1, 2, 2};
  for (size_t edge = 0; edge < packed.neighbors.size(); ++edge) {
    if (!packed.neighbors[edge]) continue;
    const int i = packed.packed_to_original[edge / width];
    const int j = packed.packed_to_original[packed.neighbors[edge] - 1];
    const double *d = &packed.displacements[edge * 4];
    const double *ge = &g[edge * 4];
    double q[3];
    // Radius and Cartesian channels are independent model inputs. Chain the radius
    // derivative back to physical Cartesian displacements before tallying virial.
    for (int k = 0; k < 3; ++k) q[k] = ge[k + 1] + ge[0] * d[k + 1] / d[0];
    for (int k = 0; k < 6; ++k) {
      const double value = -0.5 * d[a[k] + 1] * q[b[k]];
      vatom[i][k] += value;
      vatom[j][k] += value;
    }
  }
  // ComputeStressAtom reverses ghost virial contributions; do not reverse here.
}

void PairDP::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  const int nall = atom->nlocal + atom->nghost;
  const bool deviation = num_ff > 1 && update->ntimestep % out_freq == 0;
  if (deviation) {
    grow_memory(nall);
    for (int ff = 0; ff < num_ff; ++ff)
      for (int i = 0; i < nall; ++i) {
        e_atom_n[ff][i] = 0.0;
        for (int k = 0; k < 3; ++k) f_n[ff][i][k] = 0.0;
      }
  }
  for (int ff = 0; ff < num_ff; ++ff) {
    if (ff && !deviation) continue;
    const auto packed = generate_neighdata(ff);
    // Empty ranks still perform neighbor checks and all deviation collectives.
    if (!packed.nlocal) continue;
    const int width = static_cast<int>(model_data[ff].atomic_numbers.size()) * model_data[ff].max_neighbor;
    auto ints = torch::TensorOptions().dtype(torch::kInt);
    auto doubles = torch::TensorOptions().dtype(torch::kFloat64);
    auto neighbors = torch::from_blob(const_cast<int *>(packed.neighbors.data()), {1, packed.nlocal, width}, ints).to(device);
    auto types = torch::from_blob(const_cast<int *>(packed.center_types.data()), {packed.nlocal}, ints).to(device);
    auto species = torch::from_blob(active_species.data(), {static_cast<int64_t>(active_species.size())}, ints).to(device);
    auto dr = torch::from_blob(const_cast<double *>(packed.displacements.data()), {1, packed.nlocal, width, 4}, doubles).to(device);
    if (ff == 0 && vflag_atom) dr.set_requires_grad(true);
    try {
      auto output = modules[ff].forward({neighbors, types, species, dr, packed.nghost}).toTuple();
      if (output->elements().size() < 5) throw std::runtime_error("DP output must contain Etot, Ei, Force, Egroup and Virial");
      auto energy = output->elements()[0].toTensor();
      auto ei = output->elements()[1].toTensor().to(torch::kCPU).contiguous();
      auto force_tensor = output->elements()[2].toTensor().to(torch::kCPU).contiguous();
      auto model_virial = output->elements()[4].toTensor().to(torch::kCPU).contiguous();
      if (energy.numel() != 1 || ei.dim() != 2 || ei.size(0) != 1 || ei.size(1) != packed.nlocal ||
          force_tensor.dim() != 3 || force_tensor.size(0) != 1 ||
          force_tensor.size(1) != static_cast<int64_t>(packed.packed_to_original.size()) || force_tensor.size(2) != 3 ||
          model_virial.numel() != 9)
        throw std::runtime_error("Invalid DP output tensor dimensions");
      const double *fp = force_tensor.data_ptr<double>();
      const double *ep = ei.data_ptr<double>();
      for (size_t p = 0; p < packed.packed_to_original.size(); ++p) {
        const int i = packed.packed_to_original[p];
        for (int k = 0; k < 3; ++k) {
          if (ff == 0) atom->f[i][k] += fp[p * 3 + k];
          if (deviation) f_n[ff][i][k] = fp[p * 3 + k];
        }
        if (static_cast<int>(p) < packed.nlocal) {
          if (ff == 0 && eflag_atom) eatom[i] += ep[p];
          if (deviation) e_atom_n[ff][i] = ep[p];
        }
      }
      if (ff == 0) {
        if (eflag_global) eng_vdwl += energy.item<double>();
        if (vflag_global) {
          const double *vp = model_virial.data_ptr<double>();
          const int components[6] = {0, 4, 8, 1, 2, 5};
          for (int k = 0; k < 6; ++k) virial[k] += vp[components[k]];
        }
        if (vflag_atom) {
          try {
            auto gradients = torch::autograd::grad({energy.sum()}, {dr});
            if (gradients.empty() || !gradients[0].defined()) throw std::runtime_error("undefined ImageDR gradient");
            auto gradient = gradients[0].to(torch::kCPU).contiguous();
            add_peratom_virial(packed, width, gradient);
          } catch (const std::exception &e) {
            error->one(FLERR, "DP per-atom virial requires a differentiable Etot/ImageDR graph: " + std::string(e.what()));
          }
        }
      }
    } catch (const std::exception &e) {
      error->one(FLERR, "DP model evaluation failed: " + std::string(e.what()));
    }
  }
  if (deviation) {
    const auto result = calc_max_error();
    const double local_sum[2] = {std::get<0>(result), std::get<3>(result)};
    const double local_max[2] = {std::get<1>(result), std::get<4>(result)};
    const double local_min[2] = {std::get<2>(result), std::get<5>(result)};
    double sum[2], maximum[2], minimum[2];
    MPI_Allreduce(local_sum, sum, 2, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(local_max, maximum, 2, MPI_DOUBLE, MPI_MAX, world);
    MPI_Allreduce(local_min, minimum, 2, MPI_DOUBLE, MPI_MIN, world);
    bigint active_local = 0, active_global;
    for (int i = 0; i < atom->nlocal; ++i)
      if (model_data[0].mapping.active[atom->type[i] - 1]) ++active_local;
    MPI_Allreduce(&active_local, &active_global, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    for (int k = 0; k < 2; ++k) {
      if (active_global) sum[k] /= static_cast<double>(active_global);
      else sum[k] = maximum[k] = minimum[k] = 0.0;
    }
    if (me == 0) {
      fprintf(explrError_fp, "%9lld %16.9f %16.9f %16.9f %16.9f %16.9f %16.9f\n",
              static_cast<long long>(update->ntimestep), sum[0], minimum[0], maximum[0], sum[1], minimum[1], maximum[1]);
      fflush(explrError_fp);
    }
  }
}

void PairDP::grow_memory(int nall)
{
  if (nall <= nmax) return;
  nmax = std::max(nall, atom->nmax);
  memory->grow(f_n, num_ff, nmax, 3, "pair_dp:f_n");
  memory->grow(e_atom_n, num_ff, nmax, "pair_dp:e_atom_n");
}

int PairDP::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  for (int i = first; i < first + n; ++i)
    for (int k = 0; k < 3; ++k) buf[m++] = f_n[p_ff_idx][i][k];
  return m;
}

void PairDP::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) f_n[p_ff_idx][list[i]][k] += buf[m++];
}

std::tuple<double, double, double, double, double, double> PairDP::calc_max_error()
{
  for (int ff = 0; ff < num_ff; ++ff) {
    p_ff_idx = ff;
    comm->reverse_comm(this);
  }
  double sum_f = 0.0, max_f = 0.0, min_f = std::numeric_limits<double>::infinity();
  double sum_e = 0.0, max_e = 0.0, min_e = std::numeric_limits<double>::infinity();
  for (int i = 0; i < atom->nlocal; ++i) {
    if (!model_data[0].mapping.active[atom->type[i] - 1]) continue;
    double mean_f[3] = {0, 0, 0}, mean_e = 0.0;
    for (int ff = 0; ff < num_ff; ++ff) {
      for (int k = 0; k < 3; ++k) mean_f[k] += f_n[ff][i][k] / num_ff;
      mean_e += e_atom_n[ff][i] / num_ff;
    }
    double variance_f = 0.0, variance_e = 0.0;
    for (int ff = 0; ff < num_ff; ++ff) {
      for (int k = 0; k < 3; ++k) variance_f += std::pow(f_n[ff][i][k] - mean_f[k], 2) / num_ff;
      variance_e += std::pow(e_atom_n[ff][i] - mean_e, 2) / num_ff;
    }
    const double df = std::sqrt(variance_f), de = std::sqrt(variance_e);
    sum_f += df; max_f = std::max(max_f, df); min_f = std::min(min_f, df);
    sum_e += de; max_e = std::max(max_e, de); min_e = std::min(min_e, de);
  }
  return std::make_tuple(sum_f, max_f, min_f, sum_e, max_e, min_e);
}
