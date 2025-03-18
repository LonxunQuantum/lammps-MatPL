

# get package
``` bash
$  git clone https://github.com/LonxunQuantum/Lammps_for_PWMLFF.git
```

# install

## step1. load compiler
Load Intel compiler suite

``` bash
$ module load intel/2020
```
We used the `ifort` compiler to compile Fortran code, the `MKL library`, and `Intel MPI`. 

We recommend using `Intel/2020` here. Compiler versions lower than `Intel/2020` may have compilation errors when compiling lammps.

## step2 compile code

``` bash
# Firstly, compile fortran code that provides MLFF force field inference.
$ cd lammps-mlff/lammps_neigh_mlff_20230508/src/PWMATMLFF/fortran_code
$ make clean | make

# compile lammps
# The MAKE/Makempi.mpi file has already been written, there is no need to execute make yes PWMATMLFF
# 6 is the number of cores used for multi-core parallel compilation, please adjust according to the actual machine situation. 
$ cd lammps-mlff/lammps_neigh_mlff_20230508/src
$ make clean-all | make mpi -j6
```

### Possible error issues

   If the following error occurs when executing this command:

``` bash
(base) [wuxing@login src]$ make clean-all | make mpi -j 6
Gathering installed package information (may take a little while)
diff: style_angle.h: No such file or directory
make[1]: Entering directory /data/home/wuxing/codespace/lammps_fortran/lammps_neigh_mlff_20230508/src'
Gathering git version information
make[1]: Leaving directory /data/home/wuxing/codespace/lammps_fortran/lammps_neigh_mlff_20230508/src'
Compiling LAMMPS for machine mpi
cp: cannot create regular file ‘Obj_mpi/Makefile’: No such file or directory
```
Encountering this error, please re execute the compilation command. It is possible that using the - j option for parallel compilation triggered a race condition, resulting in file creation failure.


 
