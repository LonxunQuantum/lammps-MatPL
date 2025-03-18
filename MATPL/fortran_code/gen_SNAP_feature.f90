module calc_SNAP_feature

   use li_ff_mod, only: li_ff
   use nn_ff_mod, only: nn_ff

   use mod_data, only : natoms, ntypes, catype, iflag_model

   IMPLICIT NONE
   integer :: j,itype,ii,k
   integer :: max_neigh,max_neigh_M
   integer :: iat_type(100)
   real(8) :: Rc_M,Rc_type(100),Rm_type(100)
   real(8) :: E_tolerance

   integer :: m_neigh,m_neigh6
   integer :: num,num_M,natom6
   integer :: nfeat0m,nfeat0(100)

   real(8),allocatable,dimension (:,:) :: feat_M6
   real(8),allocatable,dimension (:,:,:,:) :: dfeat_M6
   integer,allocatable,dimension (:,:) :: list_neigh_alltypeM6
   integer,allocatable,dimension (:) :: num_neigh_alltypeM6
   integer :: nfeat0M6
   real(8),allocatable,dimension (:,:,:) :: dR_neigh_alltypeM6

   integer :: nsnapw_type(50)    ! the number of weight combination
   real(8) :: snapj_type(50)      !  the J  (can be integer, or hald integer, 1, 1.5)
   real(8) :: wsnap_type(50,10,50)  ! the weight
   integer :: nBB(50)
   integer :: nBBm,jmm,jm,j1,j2,m1,m2,ms,is
   real(8) :: prod,prod0
   real(8),external :: factorial
   real(8),allocatable,dimension (:,:,:,:) :: CC_func
   integer,allocatable,dimension (:,:,:) :: jjj123
   real(8),allocatable, dimension (:,:,:,:,:,:) :: Clebsch_Gordan

contains
   subroutine load_model_type6(ff_idx)
      integer, intent(in) :: ff_idx

      ! gen_SNAP_feature.in
      if (iflag_model.eq.1) then
         Rc_M=li_ff(ff_idx)%ff_Rc_M
         m_neigh=li_ff(ff_idx)%ff_max_neigh
         iat_type=li_ff(ff_idx)%ff_iat_type
         Rc_type=li_ff(ff_idx)%ff_Rc_type
         snapj_type=li_ff(ff_idx)%ff_snapj_type
         nsnapw_type=li_ff(ff_idx)%ff_nsnapw_type
         wsnap_type=li_ff(ff_idx)%ff_wsnap_type
         E_tolerance=li_ff(ff_idx)%ff_E_tolerance
      else if (iflag_model.eq.3) then
         Rc_M=nn_ff(ff_idx)%nn_feat_6_para%Rc_M
         m_neigh=nn_ff(ff_idx)%ff_max_neigh
         iat_type=nn_ff(ff_idx)%nn_feat_6_para%iat_type
         Rc_type=nn_ff(ff_idx)%nn_feat_6_para%Rc_type
         snapj_type=nn_ff(ff_idx)%nn_feat_6_para%snapj_type
         nsnapw_type=nn_ff(ff_idx)%nn_feat_6_para%nsnapw_type
         wsnap_type=nn_ff(ff_idx)%nn_feat_6_para%wsnap_type
         E_tolerance=nn_ff(ff_idx)%nn_feat_6_para%E_tolerance
      endif

      nfeat0m=0
      nBBm=0
      do itype=1,ntypes
         jm=snapj_type(itype)*2*1.0001     ! double index
         num=0
         do j=0,jm     ! jm is the double index, the real j = j/2
            do j1=0,j     ! double index, original: 0,0.5,1,1.5
               do j2=0,j1    ! double index
                  if(abs(j1-j2).le.j.and.j.le.j1+j2.and.mod(j1+j2-j+100,2).eq.0) then
                     num=num+1       ! num os the index of the feature, and its corresponding j1,j2,j
                  endif
               enddo
            enddo
         enddo
         nBB(itype)=num
         nfeat0(itype)=num*nsnapw_type(itype)
         if(nBB(itype).gt.nBBm) nBBm=nBB(itype)
         if(nfeat0(itype).gt.nfeat0m) nfeat0m=nfeat0(itype)
      enddo

      allocate(jjj123(3,nBBm,ntypes))

      jmm=0
      do itype=1,ntypes
         jm=snapj_type(itype)*2*1.0001     ! double index
         if(jm.gt.jmm) jmm=jm
         num=0
         do j=0,jm     ! jm is the double index
            do j1=0,j     ! double index, original: 0,0.5,1,1.5
               do j2=0,j1    ! double index
                  if(abs(j1-j2).le.j.and.j.le.j1+j2.and.mod(j1+j2-j+100,2).eq.0) then
                     num=num+1       ! num os the index of the feature, and its corresponding j1,j2,j
                     jjj123(1,num,itype)=j1
                     jjj123(2,num,itype)=j2
                     jjj123(3,num,itype)=j
                  endif
               enddo
            enddo
         enddo

         !  write(6,*) "itype,num,nBB(itype),nfeat0",itype,nBB(itype),nfeat0(itype)
      enddo

      !   write(6,*) "jmm=",jmm

      !-------------------------------------------
      allocate(CC_func(0:jmm,-jmm:jmm,-jmm:jmm,0:jmm))
      CC_func=0.d0

      do j=0,jmm   ! double index
         do m2=-j,j,2
            do m1=-j,j,2
               if(m1+m2.ge.0) then
                  prod0=factorial((j+m1)/2)*factorial((j-m1)/2)*factorial((j+m2)/2)*factorial((j-m2)/2)
                  prod0=dsqrt(1.d0*prod0)

                  ms=j-m1
                  if(j-m2.lt.ms) ms=j-m2
                  do is=0,ms/2
                     prod=prod0/(factorial(is)*factorial(is+(m1+m2)/2)*factorial((j-m1)/2-is)*factorial((j-m2)/2-is))
                     CC_func(is,m1,m2,j)=prod
                  enddo
               endif
            enddo
         enddo
      enddo
      !-------------------------------------------
      allocate(Clebsch_Gordan(-jmm:jmm,-jmm:jmm,-jmm:jmm,0:jmm,0:jmm,0:jmm))
      call calc_Clebsch_Gordan(Clebsch_Gordan,jmm)

      !cccccccccccccccccccccccccccccccccccccccc
      !  FInish the initial grid treatment
   end subroutine load_model_type6

   subroutine set_image_info_type6(ff_idx)
      integer, intent(in) :: ff_idx

      ! m_neigh=li_ff(ff_idx)%ff_max_neigh
      ! m_neigh6=m_neigh
      natom6=natoms

   end subroutine set_image_info_type6

   subroutine gen_SNAP_feature(num_neigh,list_neigh,dR_neigh)
      integer, dimension(ntypes,natoms), intent(in) :: num_neigh
      integer, dimension(m_neigh,ntypes,natoms), intent(in) :: list_neigh
      real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh

      integer(4) :: i,ii,jj,jjm,iat,iat1

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
      if (allocated(dfeat_M6)) then
         deallocate(feat_M6)
         deallocate(dfeat_M6)
         deallocate(list_neigh_alltypeM6)
         deallocate(num_neigh_alltypeM6)
         deallocate(dR_neigh_alltypeM6)
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

      allocate(list_neigh_alltypeM6(m_neigh,natoms))
      allocate(num_neigh_alltypeM6(natoms))
      allocate(dR_neigh_alltypeM6(3,m_neigh,natoms))
      allocate(map2neigh_alltypeM(m_neigh,natoms)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
      allocate(list_tmp(m_neigh,ntypes))

      allocate(feat(nfeat0m,natoms))         ! each note, only know its own feat
      allocate(dfeat(nfeat0m,natoms,m_neigh,3))  ! dfeat is the derivative from the neighboring dR,
      allocate(feat_M6(nfeat0m,natoms))
      allocate(dfeat_M6(nfeat0m,natoms,m_neigh,3))

      feat = 0.d0
      dfeat = 0.d0
      feat_M6 = 0.d0
      dfeat_M6 = 0.d0

      max_neigh=-1
      num_neigh_alltype=0
      max_neigh_M=-1
      num_neigh_alltypeM6=0
      list_neigh_alltypeM6=0

      do iat=1,natoms
         list_neigh_alltype(1,iat)=iat
         list_neigh_alltypeM6(1,iat)=iat

         num_M=1
         do itype=1,ntypes
            do j=1,num_neigh(itype,iat)
               ! do j=1,num_neigh_M(itype,iat)
               num_M=num_M+1
               if(num_M.gt.m_neigh) then
                  write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
                  stop
               endif
               !    list_neigh_alltypeM6(num_M,iat)=list_neigh_M(j,itype,iat)
               list_neigh_alltypeM6(num_M,iat)=list_neigh(j,itype,iat)
               list_tmp(j,itype)=num_M
               dR_neigh_alltypeM6(:,num_M,iat)=dR_neigh(:,j,itype,iat)
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
         num_neigh_alltypeM6(iat)=num_M
         if(num.gt.max_neigh) max_neigh=num
         if(num_M.gt.max_neigh_M) max_neigh_M=num_M
      enddo  ! iat

      !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      ! This num_neigh_alltype(iat) include itself !
      !    dfeat=0.d0
      !    feat=0.d0
      call find_feature_snap(Rc_type,nsnapw_type,snapj_type,wsnap_type,&
         num_neigh,list_neigh,dR_neigh,m_neigh,&
         nBB,nBBm,jjj123,CC_func,Clebsch_Gordan,jmm,&
         nfeat0m,feat,dfeat)
      !cccccccccccccccccccccccccccccccccccccccccccccccccccc

      do iat1=1,natoms
         do ii=1,nfeat0m
            feat_M6(ii,iat1)=feat(ii,iat1)
         enddo
      enddo

      iat1=0
      do iat=1,natoms
         iat1=iat1+1
         do jj=1,num_neigh_alltype(iat)
            jjm=map2neigh_alltypeM(jj,iat)
            do ii=1,nfeat0m
               dfeat_M6(ii,iat1,jjm,1)=dfeat(ii,iat1,jj,1)  ! this is the feature stored in neigh list of Rc_M
               dfeat_M6(ii,iat1,jjm,2)=dfeat(ii,iat1,jj,2)
               dfeat_M6(ii,iat1,jjm,3)=dfeat(ii,iat1,jj,3)
            enddo
         enddo
      enddo

      nfeat0M6=nfeat0m    ! the number of features for feature type 1
      !ccccccccccccccccccccccccccccccccccccccccccccc
      ! deallocate(list_neigh)
      ! deallocate(dR_neigh)
      ! deallocate(num_neigh)
      deallocate(list_neigh_alltype)
      deallocate(num_neigh_alltype)

      deallocate(list_neigh_M)
      deallocate(num_neigh_M)
      deallocate(map2neigh_M)
      ! deallocate(list_neigh_alltypeM6)
      ! deallocate(num_neigh_alltypeM6)
      deallocate(map2neigh_alltypeM)
      deallocate(list_tmp)
      deallocate(feat)
      deallocate(dfeat)
      deallocate(jjj123)
      deallocate(CC_func)
      deallocate(Clebsch_Gordan)

   end subroutine gen_SNAP_feature

end module calc_SNAP_feature
