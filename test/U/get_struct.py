from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor as AAA 
from ase.build import make_supercell
import os

mpr = MPRester()
struct = mpr.get_structure_by_material_id("mp-44",True,True)
atoms = AAA.get_atoms(struct)
atoms.cell[0,0] = 2.854
atoms.cell[1,1] = 5.869
atoms.cell[2,2] = 4.955
atoms = make_supercell(atoms,[[8,0,0],[0,4,0],[0,0,5]])
atoms.write("model.data",format="lammps-data",atom_style="atomic",specorder=["U"])
