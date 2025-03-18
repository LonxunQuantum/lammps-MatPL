module calc_deepMD2_feature

   use li_ff_mod, only: li_ff
   use nn_ff_mod, only: nn_ff

   use mod_data, only : natoms, ntypes, catype, iflag_model

   IMPLICIT NONE

   integer :: itype
   integer :: max_neigh,max_neigh_M
   integer :: iat_type(100)
   real(8) :: Rc_M,Rc_type(100),weight_rterm(100)
   real(8) :: grid2(200,50),wgauss(200,50)
   integer :: n2b_type(100)
   real(8) :: E_tolerance

   integer :: m_neigh,m_neigh8
   integer :: num,num_M,natom8
   integer :: nfeat0m,nfeat0(100),M1

   real(8),allocatable,dimension (:,:) :: feat_M8
   real(8),allocatable,dimension (:,:,:,:) :: dfeat_M8
   integer,allocatable,dimension (:,:) :: list_neigh_alltypeM8
   integer,allocatable,dimension (:) :: num_neigh_alltypeM8
   integer :: nfeat0M8
   real(8),allocatable,dimension (:,:,:) :: dR_neigh_alltypeM8

contains
   subroutine load_model_type8(ff_idx)
      integer, intent(in) :: ff_idx

      ! gen_deepMD2_feature.in
      if (iflag_model.eq.1) then
         Rc_M=li_ff(ff_idx)%ff_Rc_M
         m_neigh=li_ff(ff_idx)%ff_max_neigh
         iat_type=li_ff(ff_idx)%ff_iat_type
         Rc_type=li_ff(ff_idx)%ff_Rc_type
         n2b_type=li_ff(ff_idx)%ff_n2b_type
         weight_rterm=li_ff(ff_idx)%ff_weight_rterm
         grid2=li_ff(ff_idx)%ff_grid2
         wgauss=li_ff(ff_idx)%ff_wgauss
         E_tolerance=li_ff(ff_idx)%ff_E_tolerance
      else if (iflag_model.eq.3) then
         Rc_M=nn_ff(ff_idx)%nn_feat_8_para%Rc_M
         m_neigh=nn_ff(ff_idx)%ff_max_neigh
         iat_type=nn_ff(ff_idx)%nn_feat_8_para%iat_type
         Rc_type=nn_ff(ff_idx)%nn_feat_8_para%Rc_type
         n2b_type=nn_ff(ff_idx)%nn_feat_8_para%n2b_type
         weight_rterm=nn_ff(ff_idx)%nn_feat_8_para%weight_rterm
         grid2=nn_ff(ff_idx)%nn_feat_8_para%grid2
         wgauss=nn_ff(ff_idx)%nn_feat_8_para%wgauss
         E_tolerance=nn_ff(ff_idx)%nn_feat_8_para%E_tolerance
      endif

      nfeat0m=0
      do itype=1,ntypes
         M1=n2b_type(itype)*ntypes
         nfeat0(itype)=M1*(M1+1)/2
         if(nfeat0(itype).gt.nfeat0m) nfeat0m=nfeat0(itype)
      enddo
      !   write(6,*) "itype,nfeat0=",(nfeat0(itype),itype=1,ntypes)
      !cccccccccccccccccccccccccccccccccccccccc
      !  FInish the initial grid treatment
   end subroutine load_model_type8

   subroutine set_image_info_type8(ff_idx)
      integer, intent(in) :: ff_idx

      ! m_neigh=li_ff(ff_idx)%ff_max_neigh
      ! m_neigh8=m_neigh
      natom8 = natoms

   end subroutine set_image_info_type8

   subroutine gen_deepMD2_feature(num_neigh,list_neigh,dR_neigh)
      integer, dimension(ntypes,natoms), intent(in) :: num_neigh
      integer, dimension(m_neigh,ntypes,natoms), intent(in) :: list_neigh
      real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh

      integer(4) :: j
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
      if (allocated(dfeat_M8)) then
         deallocate(feat_M8)
         deallocate(dfeat_M8)
         deallocate(list_neigh_alltypeM8)
         deallocate(num_neigh_alltypeM8)
         deallocate(dR_neigh_alltypeM8)
      endif

      ! the dimension of these array, should be changed to natoms
      ! instead of natoms. Each process only needs to know its own natoms

      allocate(map2neigh_M(m_neigh,ntypes,natoms)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
      allocate(list_neigh_M(m_neigh,ntypes,natoms)) ! the neigh list of Rc_M
      allocate(num_neigh_M(ntypes,natoms))
      !   allocate(list_neigh(m_neigh,ntypes,natoms))
      !   allocate(dR_neigh(3,m_neigh,ntypes,natoms))   ! d(neighbore)-d(center) in xyz
      !   allocate(num_neigh(ntypes,natoms))
      allocate(list_neigh_alltype(m_neigh,natoms))
      allocate(num_neigh_alltype(natoms))
      allocate(dR_neigh_alltype(3,m_neigh,natoms))

      allocate(list_neigh_alltypeM8(m_neigh,natoms))
      allocate(num_neigh_alltypeM8(natoms))
      allocate(dR_neigh_alltypeM8(3,m_neigh,natoms))
      allocate(map2neigh_alltypeM(m_neigh,natoms)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
      allocate(list_tmp(m_neigh,ntypes))

      allocate(feat(nfeat0m,natoms))         ! each note, only know its own feat
      allocate(dfeat(nfeat0m,natoms,m_neigh,3))  ! dfeat is the derivative from the neighboring dR,
      allocate(feat_M8(nfeat0m,natoms))
      allocate(dfeat_M8(nfeat0m,natoms,m_neigh,3))

      feat = 0.d0
      dfeat = 0.d0
      feat_M8 = 0.d0
      dfeat_M8 = 0.d0

      max_neigh=-1
      num_neigh_alltype=0
      max_neigh_M=-1
      num_neigh_alltypeM8=0
      list_neigh_alltypeM8=0

      do iat=1,natoms
         list_neigh_alltype(1,iat)=iat
         list_neigh_alltypeM8(1,iat)=iat

         num_M=1
         do itype=1,ntypes
            ! do j=1,num_neigh_M(itype,iat)
            do j=1,num_neigh(itype,iat)
               num_M=num_M+1
               if(num_M.gt.m_neigh) then
                  write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
                  stop
               endif
               list_neigh_alltypeM8(num_M,iat)=list_neigh(j,itype,iat)
               ! list_neigh_alltypeM8(num_M,iat)=list_neigh_M(j,itype,iat)
               list_tmp(j,itype)=num_M
               dR_neigh_alltypeM8(:,num_M,iat)=dR_neigh(:,j,itype,iat)
            enddo
         enddo

         num=1
         map2neigh_alltypeM(1,iat)=1
         do itype=1,ntypes
            do j=1,num_neigh(itype,iat)
               num=num+1
               list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
               map2neigh_alltypeM(num,iat)=num
               ! map2neigh_alltypeM(num,iat)=list_tmp(map2neigh_M(j,itype,iat),itype)
               ! map2neigh_M(j,itype,iat), maps the jth neigh in list_neigh(Rc) to jth' neigh in list_neigh_M(Rc_M)
            enddo
         enddo

         num_neigh_alltype(iat)=num
         num_neigh_alltypeM8(iat)=num_M
         if(num.gt.max_neigh) max_neigh=num
         if(num_M.gt.max_neigh_M) max_neigh_M=num_M
      enddo  ! iat

      !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      ! This num_neigh_alltype(iat) include itself !
      !    dfeat=0.d0
      !    feat=0.d0
      call find_feature_deepMD2(Rc_type,n2b_type,weight_rterm,&
         num_neigh,list_neigh,m_neigh,dR_neigh,&
         grid2,wgauss,nfeat0m,feat,dfeat)
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc

      do iat1=1,natoms
         do ii=1,nfeat0m
            feat_M8(ii,iat1)=feat(ii,iat1)
         enddo
      enddo

      iat1=0
      do iat=1,natoms
         iat1=iat1+1
         do jj=1,num_neigh_alltype(iat)
            jjm=map2neigh_alltypeM(jj,iat)
            do ii=1,nfeat0m
               dfeat_M8(ii,iat1,jjm,1)=dfeat(ii,iat1,jj,1)  ! this is the feature stored in neigh list of Rc_M
               dfeat_M8(ii,iat1,jjm,2)=dfeat(ii,iat1,jj,2)
               dfeat_M8(ii,iat1,jjm,3)=dfeat(ii,iat1,jj,3)
            enddo
         enddo
      enddo

      nfeat0M8=nfeat0m    ! the number of features for feature type 1
      !ccccccccccccccccccccccccccccccccccccccccccccc
      !   deallocate(list_neigh)
      !   deallocate(dR_neigh)
      !   deallocate(num_neigh)
      deallocate(list_neigh_alltype)
      deallocate(num_neigh_alltype)
      deallocate(list_neigh_M)
      deallocate(num_neigh_M)
      deallocate(map2neigh_M)
      ! deallocate(list_neigh_alltypeM8)
      ! deallocate(num_neigh_alltypeM8)
      deallocate(map2neigh_alltypeM)
      deallocate(list_tmp)
      deallocate(feat)
      deallocate(dfeat)

   end subroutine gen_deepMD2_feature

end module calc_deepMD2_feature
