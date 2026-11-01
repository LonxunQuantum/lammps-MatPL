#include "atom_type_mapping.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using MatPL::AtomTypeMapping;
using MatPL::build_atom_type_mapping;
static void require(bool ok, const char *msg) { if (!ok) throw std::runtime_error(msg); }
static void expect_error(const std::vector<std::string>& symbols, const std::vector<int>& model, int ntypes) {
  bool caught=false;
  try { build_atom_type_mapping(ntypes,symbols,model); } catch (const std::invalid_argument&) { caught=true; }
  require(caught,"expected invalid mapping error");
}
int main() {
 try {
  auto a=build_atom_type_mapping(2,{"Hf","O"},{8,72});
  require(a.model_index==std::vector<int>({1,0}),"M01 all mapped order");
  require(a.atomic_number==std::vector<int>({72,8}),"M01 atomic numbers");
  auto b=build_atom_type_mapping(3,{"Hf","O","NULL"},{8,72});
  require(b.model_index==std::vector<int>({1,0,-1}) && b.atomic_number[2]==0 && !b.active[2],"M02 tail NULL");
  require(build_atom_type_mapping(3,{"null","Hf","O"},{8,72}).model_index==std::vector<int>({-1,1,0}),"M03 head NULL");
  require(build_atom_type_mapping(3,{"Hf","NuLl","O"},{8,72}).model_index==std::vector<int>({1,-1,0}),"M04 middle NULL stable indices");
  require(build_atom_type_mapping(4,{"Hf","NULL","NULL","O"},{8,72}).model_index==std::vector<int>({1,-1,-1,0}),"M05 consecutive NULL");
  expect_error({"NULL","null"},{8,72},2); // M06
  expect_error({"Si"},{8,72},1); // M07
  a=build_atom_type_mapping(2,{"NULL","O"},{8,72});
  require(a.model_index==std::vector<int>({-1,0}) && a.atomic_number==std::vector<int>({0,8}),"M08 repeated mapping rebuild");
  auto c=build_atom_type_mapping(3,{"Hf","NULL","O"},{72,8});
  auto d=build_atom_type_mapping(3,{"Hf","NULL","O"},{8,72});
  require(c.model_index==std::vector<int>({0,-1,1}) && d.model_index==std::vector<int>({1,-1,0}),"M09 per-model order");
  MatPL::validate_active_masks({c,d});
  bool caught=false;
  try { MatPL::validate_active_masks({c,build_atom_type_mapping(3,{"Hf","O","NULL"},{8,72})}); } catch (const std::invalid_argument&) { caught=true; }
  require(caught,"M10 unequal active masks rejected");
  expect_error({"O"},{8,72},2);
  expect_error({"O","Hf","O"},{8,72},2);
  expect_error({"Nullium"},{8,72},1);
  require(build_atom_type_mapping(2,{"72","8"},{8,72}).model_index==std::vector<int>({1,0}),"numeric compatibility");
  std::cout << "PASS M01-M10 and invalid argument counts\n";
 } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
