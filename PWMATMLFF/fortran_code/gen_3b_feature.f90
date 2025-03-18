!**************************************************
!     3b feature calculation on a single core
!**************************************************
module calc_ftype2

   use li_ff_mod, only: li_ff
   use nn_ff_mod, only: nn_ff

   use mod_data, only : natoms, ntypes, catype, iflag_model

   IMPLICIT NONE
   integer :: i,itype,itype1,itype2
   integer :: max_neigh,max_neigh_M
   real(8) :: E_tolerance
   integer :: recalc_grid
   real(8),allocatable,dimension (:,:,:) :: grid31_2,grid32_2
   integer :: n3b1_type(100),n3b2_type(100),n3b1m,n3b2m
   real(8) :: Rc_M

   integer :: m_neigh,m_neigh2
   integer :: num,num_M,natom2
   integer :: n3b1,n3b2,nfeat0m
   integer :: nfeat0(100)       ! ?
   integer :: iat_type(100)
   integer :: iflag_grid,iflag_ftype,iflag_grid_type(100)
   real(8) :: Rc_type(100), Rc2_type(100), Rm_type(100),fact_grid_type(100),dR_grid1_type(100),dR_grid2_type(100)

   !cccccccccccccccccccc the variable to be used in global feature type
   real(8),allocatable,dimension (:,:) :: feat_M2
   real(8),allocatable,dimension (:,:,:,:) :: dfeat_M2
   integer,allocatable,dimension (:,:) :: list_neigh_alltypeM2
   integer,allocatable,dimension (:) :: num_neigh_alltypeM2
   integer :: nfeat0M2
   real(8),allocatable,dimension (:,:,:) :: dR_neigh_alltypeM2
   !cccccccccccccccccccc the variable to be used in global feature type

contains

   subroutine load_model_type2(ff_idx)
      integer, intent(in) :: ff_idx
      integer :: kkk,k1,k2,k12,ii_f

      ! gen_3b_feature.in
      if (iflag_model.eq.1) then
         Rc_M=li_ff(ff_idx)%ff_Rc_M
         m_neigh=li_ff(ff_idx)%ff_max_neigh
         iat_type=li_ff(ff_idx)%ff_iat_type
         Rc_type=li_ff(ff_idx)%ff_Rc_type
         Rc2_type=li_ff(ff_idx)%ff_Rc2_type
         Rm_type=li_ff(ff_idx)%ff_Rm_type
         iflag_grid_type=li_ff(ff_idx)%ff_iflag_grid_type
         fact_grid_type=li_ff(ff_idx)%ff_fact_grid_type
         dR_grid1_type=li_ff(ff_idx)%ff_dR_grid1_type
         dR_grid2_type=li_ff(ff_idx)%ff_dR_grid2_type
         n3b1_type=li_ff(ff_idx)%ff_n3b1_type
         n3b2_type=li_ff(ff_idx)%ff_n3b2_type
         E_tolerance=li_ff(ff_idx)%ff_E_tolerance
         iflag_ftype=li_ff(ff_idx)%ff_iflag_ftype
         recalc_grid=li_ff(ff_idx)%ff_recalc_grid
      else if (iflag_model.eq.3) then
         Rc_M=nn_ff(ff_idx)%nn_feat_2_para%Rc_M
         m_neigh=nn_ff(ff_idx)%ff_max_neigh
         iat_type=nn_ff(ff_idx)%nn_feat_2_para%iat_type
         Rc_type=nn_ff(ff_idx)%nn_feat_2_para%Rc_type
         Rc2_type=nn_ff(ff_idx)%nn_feat_2_para%Rc2_type
         Rm_type=nn_ff(ff_idx)%nn_feat_2_para%Rm_type
         iflag_grid_type=nn_ff(ff_idx)%nn_feat_2_para%iflag_grid_type
         fact_grid_type=nn_ff(ff_idx)%nn_feat_2_para%fact_grid_type
         dR_grid1_type=nn_ff(ff_idx)%nn_feat_2_para%dR_grid1_type
         dR_grid2_type=nn_ff(ff_idx)%nn_feat_2_para%dR_grid2_type
         n3b1_type=nn_ff(ff_idx)%nn_feat_2_para%n3b1_type
         n3b2_type=nn_ff(ff_idx)%nn_feat_2_para%n3b2_type
         E_tolerance=nn_ff(ff_idx)%nn_feat_2_para%E_tolerance
         iflag_ftype=nn_ff(ff_idx)%nn_feat_2_para%iflag_ftype
         recalc_grid=nn_ff(ff_idx)%nn_feat_2_para%recalc_grid
      endif

      ! ccccccccccccccccccccccccccccccccccccccccccccccccc
      ! calculate features of all types
      n3b1m=0
      n3b2m=0
      do i=1,ntypes
         if(n3b1_type(i).gt.n3b1m) then
            n3b1m=n3b1_type(i)
         endif
         if(n3b2_type(i).gt.n3b2m) then
            n3b2m=n3b2_type(i)
         endif
      enddo
      !cccccccccccccccccccccccccccccccccccccccccccccccc
      num=0
      do itype2=1,ntypes
         do itype1=1,itype2
            do k1=1,n3b1m
               do k2=1,n3b1m
                  do k12=1,n3b2m
                     ii_f=0
                     if(itype1.ne.itype2) then
                        ii_f=1
                     endif
                     if(itype1.eq.itype2.and.k1.le.k2) then
                        ii_f=1
                     endif
                     if(ii_f.gt.0) then
                        num=num+1
                     endif
                  enddo
               enddo
            enddo
         enddo
      enddo
      nfeat0m=num

      do itype=1,ntypes
         num=0
         do itype2=1,ntypes
            do itype1=1,itype2
               do k1=1,n3b1_type(itype)
                  do k2=1,n3b1_type(itype)
                     do k12=1,n3b2_type(itype)
                        ii_f=0
                        if(itype1.ne.itype2) then
                           ii_f=1
                        endif
                        if(itype1.eq.itype2.and.k1.le.k2) then
                           ii_f=1
                        endif
                        if(ii_f.gt.0) then
                           num=num+1
                        endif
                     enddo
                  enddo
               enddo
            enddo
         enddo
         nfeat0(itype)=num
      enddo
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (.not.allocated(grid31_2)) then
         allocate(grid31_2(2,n3b1m,ntypes))
         allocate(grid32_2(2,n3b2m,ntypes))
      endif
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc
      do kkk=1,ntypes    ! center atom
         iflag_grid=iflag_grid_type(kkk)
         n3b1=n3b1_type(kkk)
         n3b2=n3b2_type(kkk)

         if(iflag_grid.eq.3) then
            ! for iflag_grid.eq.3, the grid is just read in.
            ! read grid3b_cb12_type3, grid3b_b1b2_type3
            ! For each point, it just have two numbers, r1,r2, indicating the region of the sin peak function.
            if (iflag_model.eq.1) then
               do i=1,n3b1
                  grid31_2(1,i,kkk)=li_ff(ff_idx)%ff_grid31_2(1,i,kkk)
                  grid31_2(2,i,kkk)=li_ff(ff_idx)%ff_grid31_2(2,i,kkk)
               enddo

               do i=1,n3b2
                  grid32_2(1,i,kkk)=li_ff(ff_idx)%ff_grid32_2(1,i,kkk)
                  grid32_2(2,i,kkk)=li_ff(ff_idx)%ff_grid32_2(2,i,kkk)
               enddo
            else if (iflag_model.eq.3) then
               do i=1,n3b1
                  grid31_2(1,i,kkk)=nn_ff(ff_idx)%nn_feat_2_para%grid31_2(1,i,kkk)
                  grid31_2(2,i,kkk)=nn_ff(ff_idx)%nn_feat_2_para%grid31_2(2,i,kkk)
               enddo

               do i=1,n3b2
                  grid32_2(1,i,kkk)=nn_ff(ff_idx)%nn_feat_2_para%grid32_2(1,i,kkk)
                  grid32_2(2,i,kkk)=nn_ff(ff_idx)%nn_feat_2_para%grid32_2(2,i,kkk)
               enddo
            endif

         endif
      enddo     ! kkk=1,ntypes

!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!  FInish the initial grid treatment
   end subroutine load_model_type2


   subroutine set_image_info_type2(ff_idx)
      integer, intent(in) :: ff_idx

      ! if (iflag_model.eq.1) then
      !    m_neigh=li_ff(ff_idx)%ff_max_neigh
      ! else if (iflag_model.eq.3) then
      !    m_neigh=nn_ff(ff_idx)%ff_max_neigh
      ! endif
      ! m_neigh2=m_neigh
      natom2=natoms

   end subroutine set_image_info_type2


   subroutine gen_feature_type2(num_neigh,list_neigh,dR_neigh)
      integer, dimension(ntypes,natoms), intent(in) :: num_neigh
      integer, dimension(m_neigh,ntypes,natoms), intent(in) :: list_neigh
      real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh

      integer :: j
      integer :: ii,jj,jjm,iat,iat1

      integer,allocatable,dimension (:,:) :: map2neigh_alltypeM
      integer,allocatable,dimension (:,:) :: list_tmp
      integer,allocatable,dimension (:,:,:) :: map2neigh_M
      integer,allocatable,dimension (:,:,:) :: list_neigh_M
      integer,allocatable,dimension (:,:) :: num_neigh_M
      integer,allocatable,dimension (:,:) :: list_neigh_alltype
      integer,allocatable,dimension (:) :: num_neigh_alltype
      real(8),allocatable,dimension (:,:) :: feat
      real(8),allocatable,dimension (:,:,:,:) :: dfeat
      real(8),allocatable,dimension (:,:,:) :: dR_neigh_alltype
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc

      if (allocated(feat_M2)) then
         deallocate(feat_M2)
      endif
      if (allocated(dfeat_M2)) then
         deallocate(dfeat_M2)
      endif
      if (allocated(list_neigh_alltypeM2)) then
         deallocate(list_neigh_alltypeM2)
      endif
      if (allocated(num_neigh_alltypeM2)) then
         deallocate(num_neigh_alltypeM2)
      endif
      if (allocated(dR_neigh_alltypeM2)) then
         deallocate(dR_neigh_alltypeM2)
      endif

      allocate(feat_M2(nfeat0m,natoms))
      allocate(dfeat_M2(nfeat0m,natoms,m_neigh,3))
      allocate(list_neigh_alltypeM2(m_neigh,natoms))
      allocate(num_neigh_alltypeM2(natoms))
      allocate(dR_neigh_alltypeM2(3,m_neigh,natoms))
      ! allocate(list_neigh(m_neigh,ntypes,natoms))
      ! allocate(dR_neigh(3,m_neigh,ntypes,natoms))   ! d(neighbore)-d(center) in xyz
      ! allocate(num_neigh(ntypes,natoms))
      allocate(map2neigh_M(m_neigh,ntypes,natoms)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
      allocate(list_neigh_M(m_neigh,ntypes,natoms)) ! the neigh list of Rc_M
      allocate(num_neigh_M(ntypes,natoms))
      allocate(list_neigh_alltype(m_neigh,natoms))
      allocate(num_neigh_alltype(natoms))
      allocate(dR_neigh_alltype(3,m_neigh,natoms))
      allocate(map2neigh_alltypeM(m_neigh,natoms)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
      allocate(list_tmp(m_neigh,ntypes))
      allocate(feat(nfeat0m,natoms))
      allocate(dfeat(nfeat0m,natoms,m_neigh,3))

      feat = 0.d0
      dfeat = 0.d0
      feat_M2 = 0.d0
      dfeat_M2 = 0.d0

      max_neigh=-1
      num_neigh_alltype=0
      max_neigh_M=-1
      num_neigh_alltypeM2=0
      list_neigh_alltypeM2=0

      do iat=1,natoms
         list_neigh_alltype(1,iat)=iat
         list_neigh_alltypeM2(1,iat)=iat
         dR_neigh_alltypeM2(:,1,iat)=0.d0
         num_M=1
         do itype=1,ntypes
            ! do j=1,num_neigh_M(itype,iat)
            do j=1,num_neigh(itype,iat)   ! need to check
               num_M=num_M+1
               if(num_M.gt.m_neigh) then
                  write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
                  stop
               endif
               ! list_neigh_alltypeM2(num_M,iat)=list_neigh_M(j,itype,iat)
               list_neigh_alltypeM2(num_M,iat)=list_neigh(j,itype,iat)
               list_tmp(j,itype)=num_M
               dR_neigh_alltypeM2(:,num_M,iat)=dR_neigh(:,j,itype,iat)
            enddo
         enddo

         num=1
         map2neigh_alltypeM(1,iat)=1
         do itype=1,ntypes
            do j=1,num_neigh(itype,iat)
               num=num+1
               list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
               ! map2neigh_alltypeM(num,iat)=list_tmp(map2neigh_M(j,itype,iat),itype)
               ! map2neigh_M(j,itype,iat), maps the jth neigh in list_neigh(Rc) to jth' neigh in list_neigh_M(Rc_M)
               map2neigh_alltypeM(num,iat)=num
            enddo
         enddo

         num_neigh_alltype(iat)=num
         num_neigh_alltypeM2(iat)=num_M
         if(num.gt.max_neigh) max_neigh=num
         if(num_M.gt.max_neigh_M) max_neigh_M=num_M
      enddo ! iat=1,natoms

      !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      ! This num_neigh_alltype(iat) include itself !
      !    dfeat=0.d0
      !    feat=0.d0
      !ccccc
      if(iflag_ftype.eq.3) then
         !  iflag_ftype.eq.3, the sin peak span over the two ends specified by grid31_2,grid32_2
         !  So, there could be many overlaps between different sin peaks
         call find_feature_3b_type3(num_neigh,dR_neigh,m_neigh,list_neigh,&
            n3b1_type,n3b2_type,n3b1m,n3b2m,&
            Rc2_type,grid31_2,grid32_2,&
            feat,dfeat,nfeat0m)
      endif
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc

      do iat=1,natoms
         do ii=1,nfeat0m
            feat_M2(ii,iat)=feat(ii,iat)
         enddo
      enddo

      iat1=0
      do iat=1,natoms
         iat1=iat1+1
         do jj=1,num_neigh_alltype(iat)
            jjm=map2neigh_alltypeM(jj,iat)
            do ii=1,nfeat0m
               dfeat_M2(ii,iat1,jjm,1)=dfeat(ii,iat1,jj,1)  ! this is the feature stored in neigh list of Rc_M
               dfeat_M2(ii,iat1,jjm,2)=dfeat(ii,iat1,jj,2)
               dfeat_M2(ii,iat1,jjm,3)=dfeat(ii,iat1,jj,3)
            enddo
         enddo
      enddo

      nfeat0M2=nfeat0m    ! the number of features for feature type 2

      ! deallocate(list_neigh)
      ! deallocate(dR_neigh)
      ! deallocate(num_neigh)
      deallocate(list_neigh_alltype)
      deallocate(num_neigh_alltype)
      deallocate(dR_neigh_alltype)
      deallocate(list_neigh_M)
      deallocate(num_neigh_M)
      deallocate(map2neigh_M)
      ! deallocate(feat_M2)
      ! deallocate(dfeat_M2)
      ! deallocate(list_neigh_alltypeM2) !
      ! deallocate(num_neigh_alltypeM2)  !
      deallocate(map2neigh_alltypeM)
      deallocate(list_tmp)
      deallocate(feat)  !
      deallocate(dfeat) !
      ! mem leak
      ! deallocate(grid31_2)  !
      ! deallocate(grid32_2)  !

   end subroutine gen_feature_type2

end module calc_ftype2
