// Shared LAMMPS atom-type mapping. Entries always use LAMMPS type - 1.
#ifndef MATPL_ATOM_TYPE_MAPPING_H
#define MATPL_ATOM_TYPE_MAPPING_H
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace MatPL {
struct AtomTypeMapping {
  std::vector<int> model_index;       // -1 for NULL
  std::vector<int> atomic_number;     // 0 for NULL
  std::vector<unsigned char> active;  // false for NULL
};

inline std::string lowercase(std::string text)
{
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

inline int atomic_number(const std::string& symbol)
{
  static const char* elements[] = {
    "h","he","li","be","b","c","n","o","f","ne","na","mg","al","si","p","s","cl","ar",
    "k","ca","sc","ti","v","cr","mn","fe","co","ni","cu","zn","ga","ge","as","se","br","kr",
    "rb","sr","y","zr","nb","mo","tc","ru","rh","pd","ag","cd","in","sn","sb","te","i","xe",
    "cs","ba","la","ce","pr","nd","pm","sm","eu","gd","tb","dy","ho","er","tm","yb","lu",
    "hf","ta","w","re","os","ir","pt","au","hg","tl","pb","bi","po","at","rn","fr","ra",
    "ac","th","pa","u","np","pu"
  };
  const std::string key = lowercase(symbol);
  for (int i = 0; i < 94; ++i)
    if (key == elements[i] || key == std::to_string(i + 1)) return i + 1;
  return -1;
}

inline AtomTypeMapping build_atom_type_mapping(
    int ntypes, const std::vector<std::string>& symbols, const std::vector<int>& model_elements)
{
  if (ntypes <= 0 || symbols.size() != static_cast<std::size_t>(ntypes))
    throw std::invalid_argument("Expected one mapping per LAMMPS atom type");
  AtomTypeMapping result;
  result.model_index.assign(ntypes, -1);
  result.atomic_number.assign(ntypes, 0);
  result.active.assign(ntypes, 0);
  bool any_active = false;
  for (int i = 0; i < ntypes; ++i) {
    if (lowercase(symbols[i]) == "null") continue;
    const int z = atomic_number(symbols[i]);
    if (z < 0) throw std::invalid_argument("Unknown element in atom type mapping: " + symbols[i]);
    const auto it = std::find(model_elements.begin(), model_elements.end(), z);
    if (it == model_elements.end())
      throw std::invalid_argument("Element " + symbols[i] + " is not included in model");
    result.model_index[i] = static_cast<int>(it - model_elements.begin());
    result.atomic_number[i] = z;
    result.active[i] = 1;
    any_active = true;
  }
  if (!any_active) throw std::invalid_argument("At least one atom type must be active (mapping cannot be all NULL)");
  return result;
}

inline void validate_active_masks(const std::vector<AtomTypeMapping>& mappings)
{
  for (std::size_t i = 1; i < mappings.size(); ++i)
    if (mappings[i].active != mappings[0].active)
      throw std::invalid_argument("Models used for deviation must have identical active atom type masks");
}
} // namespace MatPL
#endif
