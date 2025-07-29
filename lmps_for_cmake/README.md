# how to install

# load env
1. activate python env（pytorch）
2. load cuda/11.8-share intel/2020 gcc11
3. load cmake/3. # if the cmake in python-env, skip 

# compile
4. copy CMakeLists.txt to lammps_root_dir/cmake/
   copy basic.cmake to  lammps_root_dir/cmake/presets/
   copy pair_d3.cu  pair_d3.h  pair_d3_pars.h to lammps_root_dir/src/
   copy NEP_GPU nep_cpu.cpp nep_cpu.h pair_matpl.cpp pair_matpl.h dftd3para.h under the MATPL to lammps_root_dir/src/

5. make build dir under lammps_root_dir/

6. in build path
  cmake ../cmake -C ../cmake/presets/basic.cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
7. make -j4

8. configure lmp to envs

    export PATH=/the_lammps_root_dir/buid:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c "import torch; print(torch.__path__[0])")/lib:$(dirname $(dirname $(which python3)))/lib

