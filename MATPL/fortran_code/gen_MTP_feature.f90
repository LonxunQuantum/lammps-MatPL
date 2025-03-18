module calc_MTP_feature

   use li_ff_mod, only: li_ff,li_numCC_type,li_numT_all,li_rank_all,li_mu_all,li_jmu_b_all,li_itype_b_all,li_indi_all,li_indj_all
   use nn_ff_mod, only: nn_ff,nn_numCC_type,nn_numT_all,nn_rank_all,nn_mu_all,nn_jmu_b_all,nn_itype_b_all,nn_indi_all,nn_indj_all

   use mod_data, only : natoms, ntypes, catype, iflag_model

   IMPLICIT NONE
   integer :: i,j,itype
   integer :: max_neigh,max_neigh_M
   integer :: iat_type(100)
   real(8) :: Rc_M,Rc_type(100),Rm_type(100)
   real(8) :: E_tolerance

   integer :: m_neigh,m_neigh5
   integer :: num,num_M,natom5
   integer :: nfeat0m,nfeat0(100)

   real(8),allocatable,dimension (:,:) :: feat_M5
   real(8),allocatable,dimension (:,:,:,:) :: dfeat_M5
   integer,allocatable,dimension (:,:) :: list_neigh_alltypeM5
   integer,allocatable,dimension (:) :: num_neigh_alltypeM5
   integer :: nfeat0M5
   real(8),allocatable,dimension (:,:,:) :: dR_neigh_alltypeM5
!cccccccccccccccccccccccccccccccccccccccccccccccc
   integer :: numT_all(20000,10)
   integer :: rank_all(4,20000,10),mu_all(4,20000,10),jmu_b_all(4,20000,10),itype_b_all(4,20000,10)
   integer :: indi_all(5,4,20000,10),indj_all(5,4,20000,10)
   integer :: numCC_type(10)
   ! integer :: jmu_b(10,5000),itype_b(10,5000)
   ! integer :: indi(10,10),indj(10,10),ind(10,10)
   ! integer :: mu(10),rank(10)
   ! integer :: iflag_ti(10,10),iflag_tj(10,10)
   ! integer :: numCC,kk,numC,kkc
   ! integer :: numc0

contains
   subroutine load_model_type5(ff_idx)
      integer, intent(in) :: ff_idx

      ! gen_MTP_feature.in
      if (iflag_model.eq.1) then
         Rc_M=li_ff(ff_idx)%ff_Rc_M
         m_neigh=li_ff(ff_idx)%ff_max_neigh
         iat_type=li_ff(ff_idx)%ff_iat_type
         Rc_type=li_ff(ff_idx)%ff_Rc_type
         Rm_type=li_ff(ff_idx)%ff_Rm_type
         E_tolerance=li_ff(ff_idx)%ff_E_tolerance

         numCC_type=li_numCC_type
         numT_all=li_numT_all
         rank_all=li_rank_all
         mu_all=li_mu_all
         jmu_b_all=li_jmu_b_all
         itype_b_all=li_itype_b_all
         indi_all=li_indi_all
         indj_all=li_indj_all
      else if (iflag_model.eq.3) then
         Rc_M=nn_ff(ff_idx)%nn_feat_5_para%Rc_M
         m_neigh=nn_ff(ff_idx)%ff_max_neigh
         iat_type=nn_ff(ff_idx)%nn_feat_5_para%iat_type
         Rc_type=nn_ff(ff_idx)%nn_feat_5_para%Rc_type
         Rm_type=nn_ff(ff_idx)%nn_feat_5_para%Rm_type
         E_tolerance=nn_ff(ff_idx)%nn_feat_5_para%E_tolerance

         numCC_type=nn_numCC_type
         numT_all=nn_numT_all
         rank_all=nn_rank_all
         mu_all=nn_mu_all
         jmu_b_all=nn_jmu_b_all
         itype_b_all=nn_itype_b_all
         indi_all=nn_indi_all
         indj_all=nn_indj_all
      endif

      nfeat0m=0
      do itype=1,ntypes
         if(numCC_type(itype).gt.nfeat0m) nfeat0m=numCC_type(itype)
         nfeat0(itype)=numCC_type(itype)
      enddo
      !cccccccccccccccccccccccccccccccccccccccc
      !  FInish the initial grid treatment
   end subroutine load_model_type5

   subroutine set_image_info_type5(ff_idx)
      integer, intent(in) :: ff_idx

      ! m_neigh=li_ff(ff_idx)%ff_max_neigh
      ! m_neigh5=m_neigh
      natom5=natoms

   end subroutine set_image_info_type5

   subroutine gen_MTP_feature(num_neigh,list_neigh,dR_neigh)
      integer, dimension(ntypes,natoms), intent(in) :: num_neigh
      integer, dimension(m_neigh,ntypes,natoms), intent(in) :: list_neigh
      real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh

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
      if (allocated(dfeat_M5)) then
         deallocate(feat_M5)
         deallocate(dfeat_M5)
         deallocate(list_neigh_alltypeM5)
         deallocate(num_neigh_alltypeM5)
         deallocate(dR_neigh_alltypeM5)
      endif

      ! the dimension of these array, should be changed to natom_n
      ! instead of natoms. Each process only needs to know its own natom_n
      allocate(map2neigh_M(m_neigh,ntypes,natoms)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
      allocate(list_neigh_M(m_neigh,ntypes,natoms)) ! the neigh list of Rc_M
      allocate(num_neigh_M(ntypes,natoms))
      ! allocate(list_neigh(m_neigh,ntypes,natoms))
      ! allocate(dR_neigh(3,m_neigh,ntypes,natoms))   ! d(neighbore)-d(center) in xyz
      ! allocate(num_neigh(ntypes,natoms))
      allocate(list_neigh_alltype(m_neigh,natoms))
      allocate(num_neigh_alltype(natoms))
      allocate(dR_neigh_alltype(3,m_neigh,natoms))

      allocate(list_neigh_alltypeM5(m_neigh,natoms))
      allocate(num_neigh_alltypeM5(natoms))
      allocate(dR_neigh_alltypeM5(3,m_neigh,natoms))
      allocate(map2neigh_alltypeM(m_neigh,natoms)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
      allocate(list_tmp(m_neigh,ntypes))

      allocate(feat(nfeat0m,natoms))         ! each note, only know its own feat
      allocate(dfeat(nfeat0m,natoms,m_neigh,3))  ! dfeat is the derivative from the neighboring dR,
      allocate(feat_M5(nfeat0m,natoms))
      allocate(dfeat_M5(nfeat0m,natoms,m_neigh,3))
      ! allocate(feat(nfeat0m,natom_n))         ! each note, only know its own feat
      ! allocate(dfeat(nfeat0m,natom_n,m_neigh,3))  ! dfeat is the derivative from the neighboring dR,
      ! allocate(feat_M5(nfeat0m,natom_n))
      ! allocate(dfeat_M5(nfeat0m,natom_n,m_neigh,3))

      feat = 0.d0
      dfeat = 0.d0
      feat_M5 = 0.d0
      dfeat_M5 = 0.d0

      max_neigh=-1
      num_neigh_alltype=0
      max_neigh_M=-1
      num_neigh_alltypeM5=0
      list_neigh_alltypeM5=0

      do iat=1,natoms
         list_neigh_alltype(1,iat)=iat
         list_neigh_alltypeM5(1,iat)=iat

         num_M=1
         do itype=1,ntypes
            ! do j=1,num_neigh_M(itype,iat)
            do j=1,num_neigh(itype,iat)
               num_M=num_M+1
               if(num_M.gt.m_neigh) then
                  write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
                  stop
               endif
               ! list_neigh_alltypeM5(num_M,iat)=list_neigh_M(j,itype,iat)
               list_neigh_alltypeM5(num_M,iat)=list_neigh(j,itype,iat)
               list_tmp(j,itype)=num_M
               dR_neigh_alltypeM5(:,num_M,iat)=dR_neigh(:,j,itype,iat)
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
         num_neigh_alltypeM5(iat)=num_M
         if(num.gt.max_neigh) max_neigh=num
         if(num_M.gt.max_neigh_M) max_neigh_M=num_M
      enddo  ! iat

      !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      ! This num_neigh_alltype(iat) include itself !
      !    dfeat=0.d0
      !    feat=0.d0

      call find_feature_MTP(num_neigh,dR_neigh,m_neigh,list_neigh, &
         numCC_type,numT_all,mu_all,rank_all,jmu_b_all,itype_b_all,indi_all,indj_all, &
         feat,dfeat,nfeat0m,Rc_type,Rm_type)
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc

      do iat1=1,natoms
         do ii=1,nfeat0m
            feat_M5(ii,iat1)=feat(ii,iat1)
         enddo
      enddo

      iat1=0
      do iat=1,natoms
         iat1=iat1+1
         do jj=1,num_neigh_alltype(iat)
            jjm=map2neigh_alltypeM(jj,iat)
            do ii=1,nfeat0m
               dfeat_M5(ii,iat1,jjm,1)=dfeat(ii,iat1,jj,1)  ! this is the feature stored in neigh list of Rc_M
               dfeat_M5(ii,iat1,jjm,2)=dfeat(ii,iat1,jj,2)
               dfeat_M5(ii,iat1,jjm,3)=dfeat(ii,iat1,jj,3)
            enddo
         enddo
      enddo

      nfeat0M5=nfeat0m    ! the number of features for feature type 1
      !ccccccccccccccccccccccccccccccccccccccccccccc
      ! deallocate(list_neigh)
      ! deallocate(dR_neigh)
      ! deallocate(num_neigh)
      deallocate(list_neigh_alltype)
      deallocate(num_neigh_alltype)

      deallocate(list_neigh_M)
      deallocate(num_neigh_M)
      deallocate(map2neigh_M)
      ! deallocate(list_neigh_alltypeM5)
      ! deallocate(num_neigh_alltypeM5)
      deallocate(map2neigh_alltypeM)
      deallocate(list_tmp)
      deallocate(feat)
      deallocate(dfeat)

   end subroutine gen_MTP_feature

end module calc_MTP_feature
