# ugly makefile, but do not change this makefile unless you have full test it.
.PHONY:all
#all: cmain.x mlff_predict.x main_MD.x cmain.a
#all: cmain.x mlff_predict.x main_MD.x pair_qcad.a
all: f2c_calc_energy_force.a

MPIFC = mpiifort
MPICXX = mpiicpc

is_mpiifort := $(shell which mpiifort 1>&2 2> /dev/null; echo $$?)
ifneq ($(is_mpiifort),0)
	MPIFC=mpif90
endif

FFLAG = -O3 -I/opt/iotk/src/ -mkl
# FFLAG = -g -traceback -C -I/opt/iotk/src/ -mkl -debug all
CFLAG = -O3 --std=c++11 -mkl -I../
FLINKFLAG = -cxxlib -nofor_main
#SRCS = $(wildcard *.f90)
#OBJ = $(patsubst %.f90,%.o,$(SRCS))
# priority-sensitive
OBJ =   dp_ff_mod.o li_ff_mod.o nn_ff_mod.o mod_data.o mod_m_neigh.o \
        gen_deepMD_feature.o calc_deepMD_forceM.o \
        gen_2b_feature.o gen_3b_feature.o \
		gen_2bgauss_feature.o gen_3bcos_feature.o \
		gen_MTP_feature.o gen_SNAP_feature.o \
		gen_deepMD1_feature.o gen_deepMD2_feature.o \
		calc_lin_forceM.o calc_NN_forceM.o \
		find_feature_2b_type3.o find_feature_3b_type3.o \
		find_feature_2bgauss.o find_feature_3bcos.o \
		find_feature_SNAP.o find_feature_deepMD1.o find_feature_deepMD2.o \
		get_expand_MT.o find_feature_MTP.o calc_loop_ind.o calc_polynomial.o \
		calc_clebsch_gordan.o calc_U_JM1M2.o factorial.o \
		calc_chebyshev.o \
        ML_FF_EF.o \
        f2c_calc_energy_force.o  

# priority-sensitive
MODS =	calc_lin.mod calc_NN.mod calc_deepmd_f.mod calc_deepMD.mod calc_ftype1.mod calc_ftype2.mod calc_2bgauss_feature.mod calc_3bcos_feature.mod calc_mtp_feature.mod calc_snap_feature.mod calc_deepmd1_feature.mod calc_deepmd2_feature.mod mod_data.mod mod_m_neigh.mod
	   
%.o:  %.f90
	$(MPIFC) $(FFLAG) -c $<
%.o:  %.f
	$(MPIFC) $(FFLAG) -c $<
%.o: %.cpp
	$(MPICXX) $(CFLAG) -c $<

calc_lin.mod: calc_lin_forceM.f90
	$(MPIFC) $(FFLAG) -c $^
calc_NN.mod: calc_NN_forceM.f90
	$(MPIFC) $(FFLAG) -c $^
calc_deepMD.mod: calc_deepMD_forceM.f90
	$(MPIFC) $(FFLAG) -c $^
calc_deepMD_f.mod: gen_deepMD_feature.f90
	$(MPIFC) $(FFLAG) -c $^
calc_ftype1.mod: gen_2b_feature.f90
	$(MPIFC) $(FFLAG) -c $^
calc_ftype2.mod: gen_3b_feature.f90
	$(MPIFC) $(FFLAG) -c $^
calc_2bgauss_feature.mod: gen_2bgauss_feature.f90
	$(MPIFC) $(FFLAG) -c $^
calc_3bcos_feature.mod: gen_3bcos_feature.f90
	$(MPIFC) $(FFLAG) -c $^
calc_mtp_feature.mod: gen_MTP_feature.f90
	$(MPIFC) $(FFLAG) -c $^
calc_snap_feature.mod: gen_SNAP_feature.f90
	$(MPIFC) $(FFLAG) -c $^
calc_deepmd1_feature.mod: gen_deepMD1_feature.f90
	$(MPIFC) $(FFLAG) -c $^
calc_deepmd2_feature.mod: gen_deepMD2_feature.f90
	$(MPIFC) $(FFLAG) -c $^
mod_data.mod: mod_data.f90
	$(MPIFC) $(FFLAG) -c $^
mod_m_neigh.mod: mod_m_neigh.f90
	$(MPIFC) $(FFLAG) -c $^

mods: calc_lin.mod calc_NN.mod calc_deepMD.mod \
      mod_data.mod mod_m_neigh.mod mod_md.mod calc_deepmd_f.mod \
      calc_ftype1.mod calc_ftype2.mod \
	  calc_2bgauss_feature.mod calc_3bcos_feature.mod \
	  calc_mtp_feature.mod calc_snap_feature.mod \
	  calc_deepmd1_feature.mod calc_deepmd2_feature.mod

#mlff_predict.x: mods $(OBJ) mlff_predict.o
#	make -C iotk
#	\cp iotk/src/libiotk.a libiotk.a
#	$(MPIFC) $(FFLAG) libiotk.a -o $@ $(OBJ) mlff_predict.o

#cmain.x: mods $(OBJ) cmain.o f2c_calc_energy_force.a
#	\cp iotk/src/libiotk.a libiotk.a
#	$(MPIFC) $(FFLAG) $(FLINKFLAG) libiotk.a -o $@ $(OBJ) cmain.o
#cmain.x: cmain.o f2c_calc_energy_force.a
#	$(MPIFC) $(FFLAG) $(FLINKFLAG) f2c_calc_energy_force.a -o $@ $(OBJ) cmain.o

libiotk.a:
	make -C iotk/
	\cp iotk/src/libiotk.a libiotk.a

# the $(OBJ) contains f2c_calc_energy_force.o
# this f2c_calc_energy_force.a is used in pair_qcad.cpp
f2c_calc_energy_force.a: $(mods) $(OBJ) libiotk.a
	ar ruv $@ $(OBJ) libiotk.a
	\cp f2c_calc_energy_force.a ../
	
# this the fortran MD/MD100(inference) main program
#main_MD.x: $(mods) $(OBJ) libiotk.a main_MD.o
#	$(MPIFC) $(FFLAG) $(FLINKFLAG) libiotk.a -o $@ $(OBJ) main_MD.o

clean:
	#make clean -C iotk
	rm -f *.a *.o *.mod *.x 

