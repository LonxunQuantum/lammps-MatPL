subroutine f2c_calc_energy_force(i_model_lvn, &
   inatoms, &
   inall, &
   intypes, &
   icatype, &
   iatype, &
   inum_neigh, &
   ilist_neigh, &
   idR_neigh, &
   e_atom_out, &
   e_tot_out, &
   f_atom_out, &
   virial_out, &
   ff_idx, &
   m_neigh ) bind(c,name="f2c_calc_energy_force")

   !**************************************************
   !  Inference routine run on a SINGLE thread
   !  with neighbor list AND network paras passed-in
   !  L. Wang 2023.1
   !**************************************************
   use iso_c_binding
   use mod_data  ! natoms, nall, ntypes, catype, atype, iflag_model
   ! use for feature type
   use calc_ftype1, only: load_model_type1, set_image_info_type1
   use calc_ftype2, only: load_model_type2, set_image_info_type2
   use calc_2bgauss_feature, only: load_model_type3, set_image_info_type3
   use calc_3bcos_feature, only: load_model_type4, set_image_info_type4
   use calc_MTP_feature, only: load_model_type5, set_image_info_type5
   use calc_SNAP_feature, only: load_model_type6, set_image_info_type6
   use calc_deepMD1_feature, only: load_model_type7, set_image_info_type7
   use calc_deepMD2_feature, only: load_model_type8, set_image_info_type8
   ! use for model
   use calc_lin, only: load_model_lin, set_image_info_lin, nfeat_type_l, ifeat_type_l
   use calc_NN, only: load_model_NN, set_image_info_NN, nfeat_type_n, ifeat_type_n
   use calc_deepMD, only: load_model_deepMD, set_image_info_deepMD
   use calc_deepMD_f,only: load_model_deepMD_f,set_image_info_deepMD_f

   implicit none

   integer, intent(in) :: i_model_lvn, inatoms, inall, intypes, m_neigh
   integer, dimension(inall), intent(in) :: icatype
   integer, dimension(inall), intent(in) :: iatype
   integer, dimension(intypes, inatoms), intent(in) :: inum_neigh
   integer, dimension(m_neigh, intypes, inatoms), intent(in) :: ilist_neigh
   real*8, dimension(3, m_neigh, intypes, inatoms), intent(in) :: idR_neigh

   !! $ spend almost two days to change inall to inatoms.
   !! $ inum_neigh value reset to 0.
   real*8, dimension(inatoms), intent(out) :: e_atom_out
   real*8, intent(out) :: e_tot_out
   real*8, dimension(3,inall), intent(out) :: f_atom_out
   real*8, dimension(6), intent(out) :: virial_out
   integer, intent(in) :: ff_idx

   logical :: atype_check
   integer :: i, j, t, kk
   integer :: nfeat_type
   integer, dimension(10) :: ifeat_type

   ! write(*,*) " --- f2c CHECK --- "
   ! write(*,*) "imodel ", i_model_lvn
   ! write(*,*) "nlocal ", inatoms
   ! write(*,*) "nall   ", inall
   ! write(*,*) "ntypes ", intypes
   ! write(*,*) "ff_idx ", ff_idx

   ! do i = 1, inatoms
   !  write(*,*) i, icatype(i), iatype(i)
   ! enddo

   ! do i = 1, inatoms
   !  write(*,*) "inatoms", i
   !  do t = 1, intypes
   !    write(*,*) "  ", t, inum_neigh(t,i)
   !    do j = 1, inum_neigh(t,i)
   !     write(*,"('     ', A, 2(I5, 1X), 3(F12.6, 3X))") &
   !               "     ", j, ilist_neigh(j,t,i), &
   !       idR_neigh(1,j,t,i), idR_neigh(2,j,t,i), idR_neigh(3,j,t,i)
   !    enddo
   !  enddo
   ! enddo

   natoms = inatoms  ! number of atoms in the local region
   nall = inall      ! number of atoms in the local region + ghost atoms
   ntypes = intypes
   catype(1:nall) = icatype(1:nall)   ! lammps atom type
   atype(1:nall) = iatype(1:nall)     ! periodic table index
   iflag_model = i_model_lvn

   ! write(*,*) 'Dimensions of ilist_neigh:'
   ! write(*,*) 'm_neigh:', size(ilist_neigh, 1)
   ! write(*,*) 'intypes:', size(ilist_neigh, 2)
   ! write(*,*) 'inatoms:', size(ilist_neigh, 3)
   ! ********************************************
   !   Initialization: load control & net paras
   ! ********************************************

   if(iflag_model.eq.1) then
      call load_model_lin(ff_idx)
      ! check atom type
      atype_check = .True.
      call set_image_info_lin(atype_check)
      nfeat_type = nfeat_type_l     ! maybe define as global variable
      ifeat_type = ifeat_type_l
   endif

   if(iflag_model.eq.3) then
      call load_model_NN(ff_idx)
      ! check atom type
      atype_check = .True.
      call set_image_info_NN(atype_check)
      nfeat_type = nfeat_type_n     ! maybe define as global variable
      ifeat_type = ifeat_type_n
   endif

   if((iflag_model.eq.1).or.(iflag_model.eq.3)) then
      do kk = 1, nfeat_type
         if(ifeat_type(kk).eq.1) then
            ! load feature cutoff, shift and norm
            call load_model_type1(ff_idx)          ! load up the parameter etc
            ! allocate memroy for features
            call set_image_info_type1(ff_idx)
         endif
         if(ifeat_type(kk).eq.2) then
            call load_model_type2(ff_idx)          ! load up the parameter etc
            call set_image_info_type2(ff_idx)
         endif
         if(ifeat_type(kk).eq.3) then
            call load_model_type3(ff_idx)          ! load up the parameter etc
            call set_image_info_type3(ff_idx)
         endif
         if(ifeat_type(kk).eq.4) then
            call load_model_type4(ff_idx)          ! load up the parameter etc
            call set_image_info_type4(ff_idx)
         endif
         if(ifeat_type(kk).eq.5) then
            call load_model_type5(ff_idx)          ! load up the parameter etc
            call set_image_info_type5(ff_idx)
         endif
         if(ifeat_type(kk).eq.6) then
            call load_model_type6(ff_idx)          ! load up the parameter etc
            call set_image_info_type6(ff_idx)
         endif
         if(ifeat_type(kk).eq.7) then
            call load_model_type7(ff_idx)          ! load up the parameter etc
            call set_image_info_type7(ff_idx)
         endif
         if(ifeat_type(kk).eq.8) then
            call load_model_type8(ff_idx)          ! load up the parameter etc
            call set_image_info_type8(ff_idx)
         endif
      enddo
   endif

   if(iflag_model.eq.5) then

      ! load feature cutoff, shift and norm
      call load_model_deepMD_f(ff_idx)
      ! allocate memroy for features
      call set_image_info_deepMD_f(ff_idx)

      ! load the network parameters
      call load_model_deepMD(ff_idx)
      ! check atom type
      atype_check = .True.
      call set_image_info_deepMD(atype_check)

   endif


   ! ***************************************
   !           FF calculation
   ! ***************************************

   e_atom_out = 0.0
   e_tot_out = 0.0
   f_atom_out = 0.0
   virial_out = 0.0

   call ML_FF_EF(inum_neigh, ilist_neigh, idR_neigh, &
      e_atom_out, e_tot_out, f_atom_out, virial_out)

   ! a) the force calculated by ML_FF_EF is \partial E / \partial x,
   !    lacking a minus sign.
   ! b) the same as virial.
   f_atom_out(1:3,1:nall) = -1.0*f_atom_out(1:3,1:nall)
   virial_out(1:6) = -1.0*virial_out(1:6)

end


