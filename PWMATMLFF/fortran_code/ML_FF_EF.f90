subroutine ML_FF_EF(num_neigh,list_neigh,dR_neigh,&
   e_atom,Etot,fatom,virial)
   use IFPORT
   use mod_data, only : natoms, nall, ntypes, iflag_model
   use calc_ftype1, only : feat_M1,dfeat_M1,nfeat0M1,gen_feature_type1, &
      num_neigh_alltypeM1,list_neigh_alltypeM1,natom1,dR_neigh_alltypeM1
   use calc_ftype2, only : feat_M2,dfeat_M2,nfeat0M2,gen_feature_type2, &
      num_neigh_alltypeM2,list_neigh_alltypeM2,natom2,dR_neigh_alltypeM2
   use calc_2bgauss_feature, only : feat_M3,dfeat_M3,nfeat0M3,gen_feature_2bgauss, &
      num_neigh_alltypeM3,list_neigh_alltypeM3,natom3,dR_neigh_alltypeM3
   use calc_3bcos_feature, only : feat_M4,dfeat_M4,nfeat0M4,gen_3bcos_feature, &
      num_neigh_alltypeM4,list_neigh_alltypeM4,natom4,dR_neigh_alltypeM4
   use calc_MTP_feature, only : feat_M5,dfeat_M5,nfeat0M5,gen_MTP_feature, &
      num_neigh_alltypeM5,list_neigh_alltypeM5,natom5,dR_neigh_alltypeM5
   use calc_SNAP_feature, only : feat_M6,dfeat_M6,nfeat0M6,gen_SNAP_feature, &
      num_neigh_alltypeM6,list_neigh_alltypeM6,natom6,dR_neigh_alltypeM6
   use calc_deepMD1_feature, only : feat_M7,dfeat_M7,nfeat0M7,gen_deepMD1_feature, &
      num_neigh_alltypeM7,list_neigh_alltypeM7,natom7,dR_neigh_alltypeM7
   use calc_deepMD2_feature, only : feat_M8,dfeat_M8,nfeat0M8,gen_deepMD2_feature,  &
      num_neigh_alltypeM8,list_neigh_alltypeM8,natom8,dR_neigh_alltypeM8
   use calc_lin, only : cal_energy_force_lin, nfeat_type_l,ifeat_type_l
   use calc_NN, only : cal_energy_force_NN, nfeat_type_n,ifeat_type_n
   use calc_deepMD, only : cal_energy_force_deepMD
   use mod_m_neigh, only : load_m_neigh,m_neigh

   implicit none

   integer, dimension(ntypes,natoms), intent(in) :: num_neigh
   integer, dimension(m_neigh,ntypes,natoms), intent(in) :: list_neigh
   real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh
   real(8), intent(out) :: Etot
   real(8), dimension(natoms), intent(out) :: e_atom
   real(8), dimension(3, nall), intent(out) :: fatom
   real(8), dimension(6), intent(out) :: virial

   real(8),allocatable,dimension (:,:) :: feat
   real(8),allocatable,dimension (:,:,:,:) :: dfeat
   integer :: natom
   integer :: i, j, t, kk
   integer :: iat, ii, jj, count
   integer :: nfeat0
   integer :: nfeat_type
   integer :: ifeat_type(10)
   integer :: num_neigh_alltypeM_use(natoms)
   real(8),allocatable,dimension (:,:,:) :: dR_neigh_alltypeM_use
   integer, allocatable, dimension (:,:) :: list_neigh_alltypeM_use

   call load_m_neigh()

   ! write(*,*) "address 2 ", LOC(num_neigh)
   ! write(*,*) "address 2 ", LOC(list_neigh)
   ! write(*,*) " --- ML CHECK --- "
   ! write(*,*) "iflag ", iflag_model, m_neigh
   ! write(*,*) "ntypes ", ntypes, natoms
   ! do i = 1, natoms
   !  write(*,*) "natom ", i
   !  do t = 1, ntypes
   !    write(*,*) " ", t, num_neigh(t,i)
   !    do j = 1, m_neigh
   !      write(*,"('     ', A, 1(I10, 1X), 3(F12.6, 3X))") " l ", list_neigh(j,t,i), dR_neigh(1,j,t,i), dR_neigh(2,j,t,i), dR_neigh(3,j,t,i)
   !    enddo
   !  enddo
   ! enddo

   ! ***************************************
   !              flow1
   ! ***************************************
   if ((iflag_model.eq.1).or.(iflag_model.eq.3)) then

      if(iflag_model.eq.1) then
         nfeat_type=nfeat_type_l
         ifeat_type=ifeat_type_l
      endif

      if(iflag_model.eq.3) then
         nfeat_type=nfeat_type_n
         ifeat_type=ifeat_type_n
      endif

      nfeat0=0
      do kk = 1, nfeat_type
         if (ifeat_type(kk).eq.1) then
            call gen_feature_type1(num_neigh,list_neigh,dR_neigh)
            nfeat0=nfeat0+nfeat0M1
         endif
         if (ifeat_type(kk).eq.2) then
            call gen_feature_type2(num_neigh,list_neigh,dR_neigh)
            nfeat0=nfeat0+nfeat0M2
         endif
         if (ifeat_type(kk).eq.3) then
            call gen_feature_2bgauss(num_neigh,list_neigh,dR_neigh)
            nfeat0=nfeat0+nfeat0M3
         endif
         if (ifeat_type(kk).eq.4) then
            call gen_3bcos_feature(num_neigh,list_neigh,dR_neigh)
            nfeat0=nfeat0+nfeat0M4
         endif
         if (ifeat_type(kk).eq.5) then
            call gen_MTP_feature(num_neigh,list_neigh,dR_neigh)
            nfeat0=nfeat0+nfeat0M5
         endif
         if (ifeat_type(kk).eq.6) then
            call gen_SNAP_feature(num_neigh,list_neigh,dR_neigh)
            nfeat0=nfeat0+nfeat0M6
         endif
         if (ifeat_type(kk).eq.7) then
            call gen_deepMD1_feature(num_neigh,list_neigh,dR_neigh)
            nfeat0=nfeat0+nfeat0M7
         endif
         if (ifeat_type(kk).eq.8) then
            call gen_deepMD2_feature(num_neigh,list_neigh,dR_neigh)
            nfeat0=nfeat0+nfeat0M8
         endif
      enddo

      !*****************************************
      !         passing feature params
      !*****************************************
      do kk = 1, nfeat_type
         if (ifeat_type(kk).eq.1) then
            natom=natom1
            ! m_neigh=m_neigh1
            num_neigh_alltypeM_use = num_neigh_alltypeM1
            if(allocated(list_neigh_alltypeM_use)) then
               deallocate(list_neigh_alltypeM_use)
            endif
            if(allocated(dR_neigh_alltypeM_use)) then
               deallocate(dR_neigh_alltypeM_use)
            endif
            allocate(dR_neigh_alltypeM_use(3,m_neigh,natoms))
            allocate(list_neigh_alltypeM_use(m_neigh, natom))
            dR_neigh_alltypeM_use = dR_neigh_alltypeM1
            list_neigh_alltypeM_use = list_neigh_alltypeM1
         endif
         if (ifeat_type(kk).eq.2) then
            natom=natom2
            ! m_neigh=m_neigh2
            num_neigh_alltypeM_use = num_neigh_alltypeM2
            if(allocated(list_neigh_alltypeM_use)) then
               deallocate(list_neigh_alltypeM_use)
            endif
            if(allocated(dR_neigh_alltypeM_use)) then
               deallocate(dR_neigh_alltypeM_use)
            endif
            allocate(dR_neigh_alltypeM_use(3,m_neigh,natoms))
            allocate(list_neigh_alltypeM_use(m_neigh, natom))
            dR_neigh_alltypeM_use = dR_neigh_alltypeM2
            list_neigh_alltypeM_use = list_neigh_alltypeM2
         endif
         if (ifeat_type(kk).eq.3) then
            natom=natom3
            ! m_neigh=m_neigh3
            num_neigh_alltypeM_use = num_neigh_alltypeM3
            if(allocated(list_neigh_alltypeM_use)) then
               deallocate(list_neigh_alltypeM_use)
            endif
            if(allocated(dR_neigh_alltypeM_use)) then
               deallocate(dR_neigh_alltypeM_use)
            endif
            allocate(dR_neigh_alltypeM_use(3,m_neigh,natoms))
            allocate(list_neigh_alltypeM_use(m_neigh, natom))
            dR_neigh_alltypeM_use = dR_neigh_alltypeM3
            list_neigh_alltypeM_use = list_neigh_alltypeM3
         endif
         if (ifeat_type(kk).eq.4) then
            natom=natom4
            ! m_neigh=m_neigh4
            num_neigh_alltypeM_use = num_neigh_alltypeM4
            if(allocated(list_neigh_alltypeM_use)) then
               deallocate(list_neigh_alltypeM_use)
            endif
            if(allocated(dR_neigh_alltypeM_use)) then
               deallocate(dR_neigh_alltypeM_use)
            endif
            allocate(dR_neigh_alltypeM_use(3,m_neigh,natoms))
            allocate(list_neigh_alltypeM_use(m_neigh, natom))
            dR_neigh_alltypeM_use = dR_neigh_alltypeM4
            list_neigh_alltypeM_use = list_neigh_alltypeM4
         endif
         if (ifeat_type(kk).eq.5) then
            natom=natom5
            ! m_neigh=m_neigh5
            num_neigh_alltypeM_use = num_neigh_alltypeM5
            if(allocated(list_neigh_alltypeM_use)) then
               deallocate(list_neigh_alltypeM_use)
            endif
            if(allocated(dR_neigh_alltypeM_use)) then
               deallocate(dR_neigh_alltypeM_use)
            endif
            allocate(dR_neigh_alltypeM_use(3,m_neigh,natoms))
            allocate(list_neigh_alltypeM_use(m_neigh, natom))
            dR_neigh_alltypeM_use = dR_neigh_alltypeM5
            list_neigh_alltypeM_use = list_neigh_alltypeM5
         endif
         if (ifeat_type(kk).eq.6) then
            natom=natom6
            ! m_neigh=m_neigh6
            num_neigh_alltypeM_use = num_neigh_alltypeM6
            if(allocated(list_neigh_alltypeM_use)) then
               deallocate(list_neigh_alltypeM_use)
            endif
            if(allocated(dR_neigh_alltypeM_use)) then
               deallocate(dR_neigh_alltypeM_use)
            endif
            allocate(dR_neigh_alltypeM_use(3,m_neigh,natoms))
            allocate(list_neigh_alltypeM_use(m_neigh, natom))
            dR_neigh_alltypeM_use = dR_neigh_alltypeM6
            list_neigh_alltypeM_use = list_neigh_alltypeM6
         endif
         if (ifeat_type(kk).eq.7) then
            natom=natom7
            ! m_neigh=m_neigh7
            num_neigh_alltypeM_use = num_neigh_alltypeM7
            if(allocated(list_neigh_alltypeM_use)) then
               deallocate(list_neigh_alltypeM_use)
            endif
            if(allocated(dR_neigh_alltypeM_use)) then
               deallocate(dR_neigh_alltypeM_use)
            endif
            allocate(dR_neigh_alltypeM_use(3,m_neigh,natoms))
            allocate(list_neigh_alltypeM_use(m_neigh, natom))
            dR_neigh_alltypeM_use = dR_neigh_alltypeM7
            list_neigh_alltypeM_use = list_neigh_alltypeM7
         endif
         if (ifeat_type(kk).eq.8) then
            natom=natom8
            ! m_neigh=m_neigh8
            num_neigh_alltypeM_use = num_neigh_alltypeM8
            if(allocated(list_neigh_alltypeM_use)) then
               deallocate(list_neigh_alltypeM_use)
            endif
            if(allocated(dR_neigh_alltypeM_use)) then
               deallocate(dR_neigh_alltypeM_use)
            endif
            allocate(dR_neigh_alltypeM_use(3,m_neigh,natoms))
            allocate(list_neigh_alltypeM_use(m_neigh, natom))
            dR_neigh_alltypeM_use = dR_neigh_alltypeM8
            list_neigh_alltypeM_use = list_neigh_alltypeM8
         endif
      enddo

      !*******************************************
      !    Assemble different feature types
      !*******************************************

      ! natom_n is the divided number of atom. Defined in mod_mpi.f90

      ! allocate(feat(nfeat0,natom_n))
      ! allocate(dfeat(nfeat0,natom_n,m_neigh,3))
      allocate(feat(nfeat0,natoms))
      allocate(dfeat(nfeat0,natoms,m_neigh,3))

      count =0
      ! write(*,*) "feat_M1 ", feat_M1
      do kk = 1, nfeat_type
         ! features that are passed into the NN
         if (ifeat_type(kk).eq.1) then
            do iat=1,natoms
               do ii=1,nfeat0M1
                  feat(ii+count,iat)=feat_M1(ii,iat)
               enddo
            enddo
            count=count+nfeat0M1
         endif

         if (ifeat_type(kk).eq.2) then
            do iat=1,natoms
               do ii=1,nfeat0M2
                  feat(ii+count,iat)=feat_M2(ii,iat)
               enddo
            enddo
            count=count+nfeat0M2
         endif

         if (ifeat_type(kk).eq.3) then
            do iat=1,natoms
               do ii=1,nfeat0M3
                  feat(ii+count,iat)=feat_M3(ii,iat)
               enddo
            enddo
            count=count+nfeat0M3
         endif

         if (ifeat_type(kk).eq.4) then
            do iat=1,natoms
               do ii=1,nfeat0M4
                  feat(ii+count,iat)=feat_M4(ii,iat)
               enddo
            enddo
            count=count+nfeat0M4
         endif

         if (ifeat_type(kk).eq.5) then
            do iat=1,natoms
               do ii=1,nfeat0M5
                  feat(ii+count,iat)=feat_M5(ii,iat)
               enddo
            enddo
            count=count+nfeat0M5
         endif

         if (ifeat_type(kk).eq.6) then
            do iat=1,natoms
               do ii=1,nfeat0M6
                  feat(ii+count,iat)=feat_M6(ii,iat)
               enddo
            enddo
            count=count+nfeat0M6
         endif

         if (ifeat_type(kk).eq.7) then
            do iat=1,natoms
               do ii=1,nfeat0M7
                  feat(ii+count,iat)=feat_M7(ii,iat)
               enddo
            enddo
            count=count+nfeat0M7
         endif

         if (ifeat_type(kk).eq.8) then
            do iat=1,natoms
               do ii=1,nfeat0M8
                  feat(ii+count,iat)=feat_M8(ii,iat)
               enddo
            enddo
            count=count+nfeat0M8
         endif
      enddo

      count=0
      do kk = 1, nfeat_type
         if (ifeat_type(kk).eq.1) then
            do jj=1,m_neigh
               do iat=1,natoms
                  do ii=1,nfeat0M1
                     dfeat(ii+count,iat,jj,1)=dfeat_M1(ii,iat,jj,1)
                     dfeat(ii+count,iat,jj,2)=dfeat_M1(ii,iat,jj,2)
                     dfeat(ii+count,iat,jj,3)=dfeat_M1(ii,iat,jj,3)
                  enddo
               enddo
            enddo
            count=count+nfeat0M1
         endif

         if (ifeat_type(kk).eq.2) then
            do jj=1,m_neigh
               do iat=1,natoms
                  do ii=1,nfeat0M2
                     dfeat(ii+count,iat,jj,1)=dfeat_M2(ii,iat,jj,1)
                     dfeat(ii+count,iat,jj,2)=dfeat_M2(ii,iat,jj,2)
                     dfeat(ii+count,iat,jj,3)=dfeat_M2(ii,iat,jj,3)
                  enddo
               enddo
            enddo
            count=count+nfeat0M2
         endif

         if (ifeat_type(kk).eq.3) then
            do jj=1,m_neigh
               do iat=1,natoms
                  do ii=1,nfeat0M3
                     dfeat(ii+count,iat,jj,1)=dfeat_M3(ii,iat,jj,1)
                     dfeat(ii+count,iat,jj,2)=dfeat_M3(ii,iat,jj,2)
                     dfeat(ii+count,iat,jj,3)=dfeat_M3(ii,iat,jj,3)
                  enddo
               enddo
            enddo
            count=count+nfeat0M3
         endif

         if (ifeat_type(kk).eq.4) then
            do jj=1,m_neigh
               do iat=1,natoms
                  do ii=1,nfeat0M4
                     dfeat(ii+count,iat,jj,1)=dfeat_M4(ii,iat,jj,1)
                     dfeat(ii+count,iat,jj,2)=dfeat_M4(ii,iat,jj,2)
                     dfeat(ii+count,iat,jj,3)=dfeat_M4(ii,iat,jj,3)
                  enddo
               enddo
            enddo
            count=count+nfeat0M4
         endif

         if (ifeat_type(kk).eq.5) then
            do jj=1,m_neigh
               do iat=1,natoms
                  do ii=1,nfeat0M5
                     dfeat(ii+count,iat,jj,1)=dfeat_M5(ii,iat,jj,1)
                     dfeat(ii+count,iat,jj,2)=dfeat_M5(ii,iat,jj,2)
                     dfeat(ii+count,iat,jj,3)=dfeat_M5(ii,iat,jj,3)
                  enddo
               enddo
            enddo
            count=count+nfeat0M5
         endif

         if (ifeat_type(kk).eq.6) then
            do jj=1,m_neigh
               do iat=1,natoms
                  do ii=1,nfeat0M6
                     dfeat(ii+count,iat,jj,1)=dfeat_M6(ii,iat,jj,1)
                     dfeat(ii+count,iat,jj,2)=dfeat_M6(ii,iat,jj,2)
                     dfeat(ii+count,iat,jj,3)=dfeat_M6(ii,iat,jj,3)
                  enddo
               enddo
            enddo
            count=count+nfeat0M6
         endif

         if (ifeat_type(kk).eq.7) then
            do jj=1,m_neigh
               do iat=1,natoms
                  do ii=1,nfeat0M7
                     dfeat(ii+count,iat,jj,1)=dfeat_M7(ii,iat,jj,1)
                     dfeat(ii+count,iat,jj,2)=dfeat_M7(ii,iat,jj,2)
                     dfeat(ii+count,iat,jj,3)=dfeat_M7(ii,iat,jj,3)
                  enddo
               enddo
            enddo
            count=count+nfeat0M7
         endif

         if (ifeat_type(kk).eq.8) then
            do jj=1,m_neigh
               do iat=1,natoms
                  do ii=1,nfeat0M8
                     dfeat(ii+count,iat,jj,1)=dfeat_M8(ii,iat,jj,1)
                     dfeat(ii+count,iat,jj,2)=dfeat_M8(ii,iat,jj,2)
                     dfeat(ii+count,iat,jj,3)=dfeat_M8(ii,iat,jj,3)
                  enddo
               enddo
            enddo
            count=count+nfeat0M8
         endif
      enddo
   endif


   ! ***************************************
   !              inference
   ! ***************************************
   if(iflag_model.eq.1) then
      ! linear model
      ! write(*,*) "before call cal_energy_force_lin"
      ! write(*,*) "feat: ", feat
      call cal_energy_force_lin(feat,dfeat,num_neigh_alltypeM_use,list_neigh_alltypeM_use,dR_neigh_alltypeM_use,e_atom,Etot,fatom,virial,natom,nfeat0,m_neigh)
      ! write(*,*) "MLFF predict with linear"
      ! write(*,*) "Etot: ", Etot
      ! write(*,*) "e_atom: ", e_atom(1:natoms)
      ! stop
      ! write(*,*) "fatom: ", fatom(:,1:natoms)
   endif

   if(iflag_model.eq.3) then
      ! NN model
      call cal_energy_force_NN(feat,dfeat,num_neigh_alltypeM_use,list_neigh_alltypeM_use,dR_neigh_alltypeM_use,e_atom,Etot,fatom,virial,natom,nfeat0,m_neigh)
      ! write(*,*) "MLFF predict with NN"
      ! write(*,*) "Etot: ", Etot
      ! write(*,*) "e_atom: ", e_atom(1:natoms)
      ! stop
      ! write(*,*) "fatom: ", fatom(:,1:natoms)
   endif

   if(iflag_model.eq.5) then
      ! DP model
      call cal_energy_force_deepMD(num_neigh,list_neigh,dR_neigh,Etot,fatom,virial)
      ! write(*,*) "Etot: ", Etot
   endif

   if ((iflag_model.eq.1).or.(iflag_model.eq.3)) then

      deallocate(feat)
      deallocate(dfeat)
      deallocate(list_neigh_alltypeM_use)

   endif

! if(iflag_born_charge_ewald .eq. 1) then
!     !write(*,*) "MLFF predict with ewald"
!     allocate(ewald_atom(natom))
!     allocate(fatom_ewald(3,natom))
!     ewald = 0.0d0
!     ewald_atom = 0.0d0
!     fatom_ewald = 0.0d0
!     AL_bohr = AL*Angstrom2Bohr  ! Angstrom2Bohr; to atmoic units
!     vol = dabs(AL_bohr(3, 1)*(AL_bohr(1, 2)*AL_bohr(2, 3) - AL_bohr(1, 3)*AL_bohr(2, 2)) &
!        + AL_bohr(3, 2)*(AL_bohr(1, 3)*AL_bohr(2, 1) - AL_bohr(1, 1)*AL_bohr(2, 3)) &
!        + AL_bohr(3, 3)*(AL_bohr(1, 1)*AL_bohr(2, 2) - AL_bohr(1, 2)*AL_bohr(2, 1)))

!     call get_ewald(natom, AL_bohr, iatom, xatom, zatom_ewald, ewald, ewald_atom, fatom_ewald)
!     !write(*,*) "MD, before ewald, eatom(1:3): ", e_atom(1:3)
!     !write(*,*) "fit-lin-ewald, ewald(1:3) hartree:", ewald_atom(1:3)
!     !write(*,*) "Hartree2eV: ", Hartree2eV
!     !write(*,*) "ewald(1:3): ", ewald_atom(1:3)*Hartree2eV
!     e_atom(1:natom) = e_atom(1:natom) + ewald_atom(1:natom)*Hartree2eV
!     fatom(1:3, 1:natom) = fatom(1:3, 1:natom) + fatom_ewald(1:3, 1:natom)*Hartree2eV*Angstrom2Bohr
!     deallocate(ewald_atom)
!     deallocate(fatom_ewald)
! endif

   return

end subroutine ML_FF_EF


