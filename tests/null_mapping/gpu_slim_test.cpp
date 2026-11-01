#include "nep_type_slim.h"
#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>
int main() {
  const auto map = nep_type_slim::build_layout(4, {3, -1, 1, 3});
  assert((map.sim_to_nep == std::vector<int>{1, 3}));
  assert((map.nep_to_sim == std::vector<int>{-1, 0, -1, 1}));
  for (const auto& invalid : std::vector<std::vector<int>>{{-2, 1}, {4, 1}, {-1, -1}, {}}) {
    bool rejected = false;
    try { nep_type_slim::build_layout(4, invalid); }
    catch (const std::exception&) { rejected = true; }
    assert(rejected);
  }
  std::vector<double> full(16); std::iota(full.begin(), full.end(), 0.0);
  const auto compact = nep_type_slim::extract_coefficients(full, 4, map.sim_to_nep, 1, 1);
  assert((compact == std::vector<double>{5, 7, 13, 15}));
  std::cout << "PASS -1 omission, original model indices, duplicates, invalid bounds, active-set requirement, compact coefficient identity\n";
}
