module nn_ff_mod

   ! **************************************************************
   !           Overarching data module for forece field
   !   Will be called from LAMMPS pair constructor and destructor
   ! **************************************************************
   use iso_c_binding
   implicit none

   !!!!!!!! gen_xxx_feature.in part
   type feat_1_para
      ! parameters for 2b feature
      real(8) :: Rc_M, E_tolerance
      integer :: iflag_ftype, recalc_grid
      integer :: iat_type(100), iflag_grid_type(100), n2b_type(100)
      real(8) :: Rc_type(100), Rm_type(100), fact_grid_type(100), dR_grid1_type(100)
      real(8),allocatable,dimension (:,:,:) :: grid2_2
      integer :: n2b_tmp,n2b_tmp_idx
   end type feat_1_para

   type feat_2_para
      ! parameters for 3b feature
      real(8) :: Rc_M, E_tolerance
      integer :: iflag_ftype, recalc_grid
      integer :: iat_type(100), iflag_grid_type(100), n3b1_type(100),n3b2_type(100)
      real(8) :: Rc_type(100), Rc2_type(100), Rm_type(100), fact_grid_type(100), dR_grid1_type(100), dR_grid2_type(100)
      real(8),allocatable,dimension (:,:,:) :: grid31_2,grid32_2
      integer :: n3b1_tmp,n3b2_tmp,n3b1_tmp_idx,n3b2_tmp_idx
   end type feat_2_para

   type feat_3_para
      ! parameters for 2bgauss feature
      real(8) :: Rc_M, E_tolerance
      integer :: iat_type(100), n2b_type(100)
      real(8) :: Rc_type(100), grid2(200,50), wgauss(200,50)
   end type feat_3_para

   type feat_4_para
      ! parameters for 2bgauss feature
      real(8) :: Rc_M, E_tolerance
      integer :: iat_type(100), n3b_type(50)
      real(8) :: Rc_type(100), eta_type(200,50), w_type(100,50), alamda_type(100,50)
   end type feat_4_para

   type feat_5_para
      ! parameters for MTP feature
      real(8) :: Rc_M, E_tolerance
      integer :: iat_type(100)
      real(8) :: Rc_type(100), Rm_type(100)
      integer :: nMTP_type(100),MTP_num
      integer :: mu(10),rank(10),ind(10,10)
   end type feat_5_para

   type feat_6_para
      ! parameters for SNAP feature
      real(8) :: Rc_M, E_tolerance
      integer :: iat_type(100)
      integer :: nsnapw_type(50)
      real(8) :: Rc_type(100), snapj_type(50), wsnap_type(50,10,50)
   end type feat_6_para

   type feat_7_para
      ! parameters for deepmd1 feature
      real(8) :: Rc_M, E_tolerance
      integer :: iat_type(100)
      real(8) :: Rc_type(100), Rc2_type(100), Rm_type(100)
      real(8) :: weight_rterm(100),w_dummy
      integer :: M_type(100),M2_type(100)
   end type feat_7_para

   type feat_8_para
      ! parameters for deepmd2 feature
      real(8) :: Rc_M, E_tolerance
      integer :: iat_type(100), n2b_type(100)
      real(8) :: Rc_type(100), weight_rterm(100), grid2(200,50), wgauss(200,50)
   end type feat_8_para

   !!!!!!!! end of gen_xxx_feature.in part

   !!!!!!!! wrap up a ff with an struct
   type single_nn_ff
      ! force field paras
      integer :: ff_model_idx
      integer :: ff_max_neigh

      ! feat info part
      integer :: ff_iflag_PCA
      integer :: ff_nfeat_type     ! number of feat type
      integer :: ff_ifeat_type(10)        ! feat type list
      integer :: ff_num_type
      integer,allocatable,dimension(:,:) :: ff_itype_atom_sumfe ! a list with atomtype and sum of feats, atomtype,feat1,feat2

      ! Wij part
      integer :: ff_nodeMM,ff_nlayer
      integer,allocatable,dimension(:,:) :: ff_nodeNN
      real(8),allocatable,dimension(:,:) :: ff_a_scaler,ff_b_scaler
      real*8,allocatable,dimension(:,:,:,:) :: ff_Wij_nn
      real*8,allocatable,dimension(:,:,:) :: ff_B_nn
      ! vdw part
      integer :: nterm
      real(8),allocatable,dimension(:) :: rad_atom,E_ave_vdw
      real(8),allocatable,dimension(:,:,:) :: wp_atom

      ! input/gen_feature_xxx.in
      type(feat_1_para) nn_feat_1_para
      type(feat_2_para) nn_feat_2_para
      type(feat_3_para) nn_feat_3_para
      type(feat_4_para) nn_feat_4_para
      type(feat_5_para) nn_feat_5_para
      type(feat_6_para) nn_feat_6_para
      type(feat_7_para) nn_feat_7_para
      type(feat_8_para) nn_feat_8_para

   end type single_nn_ff

   ! structs for ff data
   type(single_nn_ff) nn_ff(4)

   ! loop index
   integer :: i,ii,j,jj,k,kk,kkk,skip
   integer :: itype,itype1,itype2,num,k1,k2,k12,ii_f

   ! others tmps
   logical(2) :: alive_nn
   integer :: n2bm      ! numOf2bfeat
   integer :: n3b1m,n3b2m
   ! integer nfeat0m_1, nfeat0m_2   ! _1 for feature1, _2 for feature2
   integer :: iflag_grid1,iflag_grid2,iflag_ftype1,iflag_ftype2
   integer :: tmp_nfeat1
   integer :: ntype_vdw,itype_vdw
   character(20) :: txt
   integer :: nlayer_tmp,i1,i2,ntmp,j1,j2
   real(8) :: w_tmp,b_tmp

   logical :: f12_called,f34_called  ! 声明一个逻辑变量����������������用于跟踪是否已经调用了f12,f34

   INTEGER :: ierr
   !    character*1 :: txt
   integer :: jmu_b(10,5000),itype_b(10,5000)
   integer :: indi(10,10),indj(10,10)
   integer :: iflag_ti(10,10),iflag_tj(10,10)
   integer :: numCC,numC,kkc
   integer :: numc0
   integer :: nn_numT_all(20000,10)
   integer :: nn_rank_all(4,20000,10),nn_mu_all(4,20000,10),nn_jmu_b_all(4,20000,10),nn_itype_b_all(4,20000,10)
   integer :: nn_indi_all(5,4,20000,10),nn_indj_all(5,4,20000,10)
   integer :: nn_numCC_type(10)

contains

   subroutine nn_ff_load(name_ptr, ff_idx, slen, ocut, m_neigh) bind(c,name="nn_ff_load")
      character, dimension(300), intent(in) :: name_ptr    ! name string pointer
      integer, intent(in) :: slen            ! length of name string
      integer, intent(in) :: ff_idx          ! index of ff to be loaded
      real(8), intent(out) :: ocut
      integer, intent(out) :: m_neigh
      character(300) temp

      temp = trim(name_ptr(1))
      do i=2,slen
         temp = trim(temp)//trim(name_ptr(i))
      enddo

      inquire(file=temp,exist=alive_nn)
      if (alive_nn.ne..true.) then
         write(*,*) "force field named ", temp, " not found. Terminate."
         stop
      endif

      ! *********************************************
      !      Allocate the ff pointers and load
      ! *********************************************

      ! cccccccccccccccccccccccccccccccccccccccccccccccc
      ! this part can be used for all features
      ! reading feat.info part
      open(10, file=temp)
      read(10,*) nn_ff(ff_idx)%ff_model_idx
      if (nn_ff(ff_idx)%ff_model_idx.ne.3) then
         write(*,*) "Model type error. Should be 3"
         stop
      endif

      read(10,*) nn_ff(ff_idx)%ff_iflag_PCA   ! this can be used to turn off degmm part
      read(10,*) nn_ff(ff_idx)%ff_nfeat_type  ! number of feat type
      do i=1,nn_ff(ff_idx)%ff_nfeat_type
         read(10,*) nn_ff(ff_idx)%ff_ifeat_type(i)
      enddo
      read(10,*) nn_ff(ff_idx)%ff_num_type
      allocate(nn_ff(ff_idx)%ff_itype_atom_sumfe(nn_ff(ff_idx)%ff_num_type,2))
      tmp_nfeat1=0
      do i=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) (nn_ff(ff_idx)%ff_itype_atom_sumfe(i,k),k=1,3)
         if (nn_ff(ff_idx)%ff_itype_atom_sumfe(i,2).gt.tmp_nfeat1) then
            tmp_nfeat1=nn_ff(ff_idx)%ff_itype_atom_sumfe(i,2)
         endif
      enddo

      f12_called = .false.  ! Initialize to false
      f34_called = .false.
      do kk = 1, nn_ff(ff_idx)%ff_nfeat_type
         ! for feature1 & feature2
         if ((nn_ff(ff_idx)%ff_ifeat_type(kk).eq.1) .or. (nn_ff(ff_idx)%ff_ifeat_type(kk).eq.2)) then
            if (.not. f12_called) then    ! 只在f12未被调用时调用一次
               call f12(ff_idx,ocut,m_neigh)
               f12_called = .true.        ! 设置f12_called为真，表示已经调用了f12
            endif
         endif
         ! for feature3 & feature4
         if ((nn_ff(ff_idx)%ff_ifeat_type(kk).eq.3) .or. (nn_ff(ff_idx)%ff_ifeat_type(kk).eq.4)) then
            if (.not. f34_called) then
               call f34(ff_idx,ocut,m_neigh)
               f34_called = .true.
            endif
         endif
         ! for feature5
         if (nn_ff(ff_idx)%ff_ifeat_type(kk).eq.5) then
            call f5(ff_idx,ocut,m_neigh)
         endif
         ! for feature6
         if (nn_ff(ff_idx)%ff_ifeat_type(kk).eq.6) then
            call f6(ff_idx,ocut,m_neigh)
         endif
         ! for feature7
         if (nn_ff(ff_idx)%ff_ifeat_type(kk).eq.7) then
            call f7(ff_idx,ocut,m_neigh)
         endif
         ! for feature8
         if (nn_ff(ff_idx)%ff_ifeat_type(kk).eq.8) then
            call f8(ff_idx,ocut,m_neigh)
         endif
      enddo

      ! reading Wij part
      do skip=1,5
         read(10,*)
      enddo
      read(10,*) txt,nlayer_tmp

      nn_ff(ff_idx)%ff_nlayer = nlayer_tmp/2/nn_ff(ff_idx)%ff_num_type

      if(nlayer_tmp.ne.nn_ff(ff_idx)%ff_nlayer*2*nn_ff(ff_idx)%ff_num_type) then
         write(6,*) "ERROR! nlayer is wrong",nlayer_tmp
         stop
      endif

      if(.not.allocated(nn_ff(ff_idx)%ff_nodeNN)) then
         allocate(nn_ff(ff_idx)%ff_nodeNN(nn_ff(ff_idx)%ff_nlayer+1,nn_ff(ff_idx)%ff_num_type))
      endif
      if(.not.allocated(nn_ff(ff_idx)%ff_a_scaler)) then
         allocate(nn_ff(ff_idx)%ff_a_scaler(tmp_nfeat1,nn_ff(ff_idx)%ff_num_type))
         allocate(nn_ff(ff_idx)%ff_b_scaler(tmp_nfeat1,nn_ff(ff_idx)%ff_num_type))
         allocate(nn_ff(ff_idx)%ff_Wij_nn(tmp_nfeat1,tmp_nfeat1,nn_ff(ff_idx)%ff_nlayer,nn_ff(ff_idx)%ff_num_type))
         allocate(nn_ff(ff_idx)%ff_B_nn(tmp_nfeat1,nn_ff(ff_idx)%ff_nlayer,nn_ff(ff_idx)%ff_num_type))
      endif

      do itype=1,nn_ff(ff_idx)%ff_num_type
         do ii=1,nn_ff(ff_idx)%ff_nlayer
            read(10,*) txt,nn_ff(ff_idx)%ff_nodeNN(ii,itype),nn_ff(ff_idx)%ff_nodeNN(ii+1,itype)

            if(ii.eq.1.and.nn_ff(ff_idx)%ff_nodeNN(1,itype).ne.nn_ff(ff_idx)%ff_itype_atom_sumfe(itype,2)) then
               write(6,*) "nodeNN in Wij.txt not correct",nn_ff(ff_idx)%ff_nodeNN(1,itype),nn_ff(ff_idx)%ff_itype_atom_sumfe(itype,2)
               stop
            endif

            do j=1,nn_ff(ff_idx)%ff_nodeNN(ii,itype)*nn_ff(ff_idx)%ff_nodeNN(ii+1,itype)
               read(10,*) i1,i2,w_tmp
               nn_ff(ff_idx)%ff_Wij_nn(i1+1,i2+1,ii,itype)=w_tmp
            enddo

            read(10,*)

            do j=1,nn_ff(ff_idx)%ff_nodeNN(ii+1,itype)
               read(10,*) i1,i2,b_tmp
               nn_ff(ff_idx)%ff_B_nn(j,ii,itype)=b_tmp
            enddo
         enddo
      enddo

      nn_ff(ff_idx)%ff_nodeMM=0            ! max number of nodes = max number of features, can be deleted
      do itype=1,nn_ff(ff_idx)%ff_num_type
         do ii=1,nn_ff(ff_idx)%ff_nlayer+1
            if(nn_ff(ff_idx)%ff_nodeNN(ii,itype).gt.nn_ff(ff_idx)%ff_nodeMM) then
               nn_ff(ff_idx)%ff_nodeMM = nn_ff(ff_idx)%ff_nodeNN(ii,itype)
            endif
         enddo
      enddo
      ! ****************** read Wij ends************************

      ! ****************** read scaler starts************************
      do skip=1,3
         read(10,*)
      enddo
      read(10,*) txt,ntmp

      if(ntmp.ne.nn_ff(ff_idx)%ff_num_type*2) then
         write(6,*) "size not right in data_scale.txt",ii
         stop
      endif

      do itype=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) txt,ntmp

         if(ntmp.ne.nn_ff(ff_idx)%ff_itype_atom_sumfe(itype,2)) then
            write(6,*) "nfeat size not correct in data_scale.txt",ntmp
            stop
         endif

         do j=1,ntmp
            read(10,*) j1,j2, nn_ff(ff_idx)%ff_a_scaler(j,itype)
         enddo

         read(10,*) txt,ntmp

         if(ntmp.ne.nn_ff(ff_idx)%ff_itype_atom_sumfe(itype,2)) then
            write(6,*) "nfeat size not correct in data_scale.txt",ntmp
            stop
         endif

         do j=1,ntmp
            read(10,*) j1,j2, nn_ff(ff_idx)%ff_b_scaler(j,itype)
         enddo
      enddo
      ! ****************** read scaler ends************************

      ! reading vdw_fitB.ntype
      ! ************* NOT IN USE NOW **************
      read(10,*) ntype_vdw,nn_ff(ff_idx)%nterm
      if(nn_ff(ff_idx)%nterm.gt.2) then
         write(6,*) "nterm.gt.2,stop"
         stop
      endif
      if(ntype_vdw.ne.nn_ff(ff_idx)%ff_num_type) then
         write(6,*) "ntypes not same in vdw_fitB.ntype,something wrong",ntype_vdw,nn_ff(ff_idx)%ff_num_type
         stop
      endif
      allocate(nn_ff(ff_idx)%rad_atom(nn_ff(ff_idx)%ff_num_type))
      allocate(nn_ff(ff_idx)%E_ave_vdw(nn_ff(ff_idx)%ff_num_type))
      allocate(nn_ff(ff_idx)%wp_atom(nn_ff(ff_idx)%ff_num_type,nn_ff(ff_idx)%ff_num_type,2))
      do itype=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) itype_vdw,nn_ff(ff_idx)%rad_atom(itype),nn_ff(ff_idx)%E_ave_vdw(itype),&
            ((nn_ff(ff_idx)%wp_atom(i,itype,j),i=1,nn_ff(ff_idx)%ff_num_type),j=1,nn_ff(ff_idx)%nterm)
      enddo

      close(10)

   end subroutine nn_ff_load

   subroutine f12(ff_idx,ocut,m_neigh)
      integer, intent(in) :: ff_idx          ! index of ff to be loaded
      real(8), intent(out) :: ocut           ! cutoff radius
      integer, intent(out) :: m_neigh        ! max number of neighbors

      ! reading gen_2b_feature.in part
      read(10,*) nn_ff(ff_idx)%nn_feat_1_para%Rc_M, nn_ff(ff_idx)%ff_max_neigh
      
      m_neigh = nn_ff(1)%ff_max_neigh

      read(10,*) nn_ff(ff_idx)%ff_num_type

      do i=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) nn_ff(ff_idx)%nn_feat_1_para%iat_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_1_para%Rc_type(i), &
            nn_ff(ff_idx)%nn_feat_1_para%Rm_type(i), &
            nn_ff(ff_idx)%nn_feat_1_para%iflag_grid_type(i), &
            nn_ff(ff_idx)%nn_feat_1_para%fact_grid_type(i), &
            nn_ff(ff_idx)%nn_feat_1_para%dR_grid1_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_1_para%n2b_type(i)

         if(nn_ff(ff_idx)%nn_feat_1_para%Rc_type(i).gt.nn_ff(ff_idx)%nn_feat_1_para%Rc_M) then
            write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in", &
               i,nn_ff(ff_idx)%nn_feat_1_para%Rc_type(i),nn_ff(ff_idx)%nn_feat_1_para%Rc_M
            stop
         endif
      enddo

      read(10,*) nn_ff(ff_idx)%nn_feat_1_para%E_tolerance
      read(10,*) nn_ff(ff_idx)%nn_feat_1_para%iflag_ftype
      read(10,*) nn_ff(ff_idx)%nn_feat_1_para%recalc_grid

      ! reading gen_3b_feature.in part
      read(10,*) nn_ff(ff_idx)%nn_feat_2_para%Rc_M, nn_ff(ff_idx)%ff_max_neigh

      m_neigh = nn_ff(1)%ff_max_neigh

      read(10,*) nn_ff(ff_idx)%ff_num_type

      do i=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) nn_ff(ff_idx)%nn_feat_2_para%iat_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_2_para%Rc_type(i), &
            nn_ff(ff_idx)%nn_feat_2_para%Rc2_type(i), &
            nn_ff(ff_idx)%nn_feat_2_para%Rm_type(i), &
            nn_ff(ff_idx)%nn_feat_2_para%iflag_grid_type(i), &
            nn_ff(ff_idx)%nn_feat_2_para%fact_grid_type(i), &
            nn_ff(ff_idx)%nn_feat_2_para%dR_grid1_type(i), &
            nn_ff(ff_idx)%nn_feat_2_para%dR_grid2_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_2_para%n3b1_type(i), nn_ff(ff_idx)%nn_feat_2_para%n3b2_type(i)

         if(nn_ff(ff_idx)%nn_feat_2_para%Rc_type(i).gt.nn_ff(ff_idx)%nn_feat_2_para%Rc_M) then
            write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in", &
               i,nn_ff(ff_idx)%nn_feat_2_para%Rc_type(i),nn_ff(ff_idx)%nn_feat_2_para%Rc_M
            stop
         endif

         if(nn_ff(ff_idx)%nn_feat_2_para%Rc2_type(i).gt.2*nn_ff(ff_idx)%nn_feat_2_para%Rc_type(i)) then
            write(6,*) "Rc2_type must be smaller than 2*Rc_type, gen_3b_feature.in", &
               i,nn_ff(ff_idx)%nn_feat_2_para%Rc_type(i),nn_ff(ff_idx)%nn_feat_2_para%Rc2_type(i)
            stop
         endif
      enddo

      read(10,*) nn_ff(ff_idx)%nn_feat_2_para%E_tolerance
      read(10,*) nn_ff(ff_idx)%nn_feat_2_para%iflag_ftype
      read(10,*) nn_ff(ff_idx)%nn_feat_2_para%recalc_grid

      !cccccccccccccccccccccccccccccccccccccccccccccccccccc
      ! calculate features of all types
      n2bm=0
      n3b1m=0
      n3b2m=0

      iflag_ftype1=nn_ff(ff_idx)%nn_feat_1_para%iflag_ftype
      iflag_ftype2=nn_ff(ff_idx)%nn_feat_2_para%iflag_ftype

      do i=1,nn_ff(ff_idx)%ff_num_type
         if(iflag_ftype1.eq.3.and.nn_ff(ff_idx)%nn_feat_1_para%iflag_grid_type(i).ne.3) then
            write(6,*) "if iflag_ftype.eq.3, iflag_grid must equal 3, stop"
            stop
         endif
         if(iflag_ftype2.eq.3.and.nn_ff(ff_idx)%nn_feat_2_para%iflag_grid_type(i).ne.3) then
            write(6,*) "if iflag_ftype.eq.3, iflag_grid must equal 3, stop"
            stop
         endif
         ! for 2b
         if(nn_ff(ff_idx)%nn_feat_1_para%n2b_type(i).gt.n2bm) then
            n2bm=nn_ff(ff_idx)%nn_feat_1_para%n2b_type(i)
         endif
         ! for 3b
         if(nn_ff(ff_idx)%nn_feat_2_para%n3b1_type(i).gt.n3b1m) then
            n3b1m=nn_ff(ff_idx)%nn_feat_2_para%n3b1_type(i)
         endif
         if(nn_ff(ff_idx)%nn_feat_2_para%n3b2_type(i).gt.n3b2m) then
            n3b2m=nn_ff(ff_idx)%nn_feat_2_para%n3b2_type(i)
         endif
      enddo

      ! nfeat0m_1=nn_ff(ff_idx)%ff_num_type*n2bm

      num=0
      do itype2=1,nn_ff(ff_idx)%ff_num_type
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

      ! nfeat0m_2=num

      ! cccccccccccccccccccccccccccccccc
      ! reading grid part
      allocate(nn_ff(ff_idx)%nn_feat_1_para%grid2_2(2,n2bm+1,nn_ff(ff_idx)%ff_num_type))
      allocate(nn_ff(ff_idx)%nn_feat_2_para%grid31_2(2,n3b1m,nn_ff(ff_idx)%ff_num_type))
      allocate(nn_ff(ff_idx)%nn_feat_2_para%grid32_2(2,n3b2m,nn_ff(ff_idx)%ff_num_type))

      ! for 2b
      do kkk=1,nn_ff(ff_idx)%ff_num_type  ! center atom???
         iflag_grid1=nn_ff(ff_idx)%nn_feat_1_para%iflag_grid_type(kkk)

         if(iflag_grid1.eq.3) then
            ! read grid2b_type3
            read(10,*) nn_ff(ff_idx)%nn_feat_1_para%n2b_tmp

            if(nn_ff(ff_idx)%nn_feat_1_para%n2b_tmp.ne.nn_ff(ff_idx)%nn_feat_1_para%n2b_type(kkk)) then
               write(6,*) "n2b_tmp.ne.ff_n2b_type,in grid2b_type3", nn_ff(ff_idx)%nn_feat_1_para%n2b_tmp,nn_ff(ff_idx)%nn_feat_1_para%n2b_type(kkk)
               stop
            endif

            do i=1,nn_ff(ff_idx)%nn_feat_1_para%n2b_type(kkk)
               read(10,*) nn_ff(ff_idx)%nn_feat_1_para%n2b_tmp_idx, &
                  nn_ff(ff_idx)%nn_feat_1_para%grid2_2(1,i,kkk), &
                  nn_ff(ff_idx)%nn_feat_1_para%grid2_2(2,i,kkk)
               if(nn_ff(ff_idx)%nn_feat_1_para%grid2_2(2,i,kkk).gt.nn_ff(ff_idx)%nn_feat_1_para%Rc_type(kkk)) then
                  write(6,*) "grid2_2.gt.Rc",nn_ff(ff_idx)%nn_feat_1_para%grid2_2(2,i,kkk),nn_ff(ff_idx)%nn_feat_1_para%Rc_type(kkk)
               endif
            enddo
         endif
      enddo

      ! for 3b
      do kkk=1,nn_ff(ff_idx)%ff_num_type
         iflag_grid2=nn_ff(ff_idx)%nn_feat_2_para%iflag_grid_type(kkk)

         if(iflag_grid2.eq.3) then
            ! read grid3b_b1b2_type3
            read(10,*) nn_ff(ff_idx)%nn_feat_2_para%n3b2_tmp

            if(nn_ff(ff_idx)%nn_feat_2_para%n3b2_tmp.ne.nn_ff(ff_idx)%nn_feat_2_para%n3b2_type(kkk)) then
               write(6,*) "n3b2_tmp.ne.ff_n3b2_type,in grid2b_type3", nn_ff(ff_idx)%nn_feat_2_para%n3b2_tmp,nn_ff(ff_idx)%nn_feat_2_para%n3b2_type(kkk)
               stop
            endif

            do i=1,nn_ff(ff_idx)%nn_feat_2_para%n3b2_type(kkk)
               read(10,*) nn_ff(ff_idx)%nn_feat_2_para%n3b2_tmp_idx, &
                  nn_ff(ff_idx)%nn_feat_2_para%grid32_2(1,i,kkk), &
                  nn_ff(ff_idx)%nn_feat_2_para%grid32_2(2,i,kkk)
               if(nn_ff(ff_idx)%nn_feat_2_para%grid32_2(2,i,kkk).gt.nn_ff(ff_idx)%nn_feat_2_para%Rc_type(kkk)) then
                  write(6,*) "grid32_2.gt.Rc",nn_ff(ff_idx)%nn_feat_2_para%grid32_2(2,i,kkk),nn_ff(ff_idx)%nn_feat_2_para%Rc_type(kkk)
               endif
            enddo
         endif
      enddo

      do kkk=1,nn_ff(ff_idx)%ff_num_type
         iflag_grid2=nn_ff(ff_idx)%nn_feat_2_para%iflag_grid_type(kkk)

         if(iflag_grid2.eq.3) then
            ! read grid3b_cb12_type3
            read(10,*) nn_ff(ff_idx)%nn_feat_2_para%n3b1_tmp

            if(nn_ff(ff_idx)%nn_feat_2_para%n3b1_tmp.ne.nn_ff(ff_idx)%nn_feat_2_para%n3b1_type(kkk)) then
               write(6,*) "n3b1_tmp.ne.ff_n3b1_type,in grid2b_type3", nn_ff(ff_idx)%nn_feat_2_para%n3b1_tmp,nn_ff(ff_idx)%nn_feat_2_para%n3b1_type(kkk)
               stop
            endif

            do i=1,nn_ff(ff_idx)%nn_feat_2_para%n3b1_type(kkk)
               read(10,*) nn_ff(ff_idx)%nn_feat_2_para%n3b1_tmp_idx, &
                  nn_ff(ff_idx)%nn_feat_2_para%grid31_2(1,i,kkk), &
                  nn_ff(ff_idx)%nn_feat_2_para%grid31_2(2,i,kkk)
               if(nn_ff(ff_idx)%nn_feat_2_para%grid31_2(2,i,kkk).gt.nn_ff(ff_idx)%nn_feat_2_para%Rc_type(kkk)) then
                  write(6,*) "grid31_2.gt.Rc",nn_ff(ff_idx)%nn_feat_2_para%grid31_2(2,i,kkk),nn_ff(ff_idx)%nn_feat_2_para%Rc_type(kkk)
               endif
            enddo
         endif
      enddo

      ocut=nn_ff(1)%nn_feat_2_para%Rc_M
      print*,"feature1 & feature2"

   end subroutine f12

   subroutine f34(ff_idx,ocut,m_neigh)
      integer, intent(in) :: ff_idx          ! index of ff to be loaded
      real(8), intent(out) :: ocut           ! cutoff radius
      integer, intent(out) :: m_neigh

      ! reading gen_2bgauss_feature.in part
      read(10,*) nn_ff(ff_idx)%nn_feat_3_para%Rc_M, nn_ff(ff_idx)%ff_max_neigh

      m_neigh = nn_ff(1)%ff_max_neigh

      read(10,*) nn_ff(ff_idx)%ff_num_type
      do i=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) nn_ff(ff_idx)%nn_feat_3_para%iat_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_3_para%Rc_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_3_para%n2b_type(i)
         do j=1,nn_ff(ff_idx)%nn_feat_3_para%n2b_type(i)
            read(10,*) nn_ff(ff_idx)%nn_feat_3_para%grid2(j,i),nn_ff(ff_idx)%nn_feat_3_para%wgauss(j,i)
         enddo

         if(nn_ff(ff_idx)%nn_feat_3_para%Rc_type(i).gt.nn_ff(ff_idx)%nn_feat_3_para%Rc_M) then
            write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in", &
               i,nn_ff(ff_idx)%nn_feat_3_para%Rc_type(i),nn_ff(ff_idx)%nn_feat_3_para%Rc_M
            stop
         endif
      enddo
      read(10,*) nn_ff(ff_idx)%nn_feat_3_para%E_tolerance

      ! reading gen_3bcos_feature.in part
      read(10,*) nn_ff(ff_idx)%nn_feat_4_para%Rc_M, nn_ff(ff_idx)%ff_max_neigh

      m_neigh = nn_ff(1)%ff_max_neigh

      read(10,*) nn_ff(ff_idx)%ff_num_type
      do i=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) nn_ff(ff_idx)%nn_feat_4_para%iat_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_4_para%Rc_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_4_para%n3b_type(i)
         do j=1,nn_ff(ff_idx)%nn_feat_4_para%n3b_type(i)
            read(10,*) nn_ff(ff_idx)%nn_feat_4_para%eta_type(j,i), &
               nn_ff(ff_idx)%nn_feat_4_para%w_type(j,i), &
               nn_ff(ff_idx)%nn_feat_4_para%alamda_type(j,i)
         enddo

         if(nn_ff(ff_idx)%nn_feat_4_para%Rc_type(i).gt.nn_ff(ff_idx)%nn_feat_4_para%Rc_M) then
            write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in", &
               i,nn_ff(ff_idx)%nn_feat_4_para%Rc_type(i),nn_ff(ff_idx)%nn_feat_4_para%Rc_M
            stop
         endif
      enddo
      read(10,*) nn_ff(ff_idx)%nn_feat_4_para%E_tolerance

      ocut=nn_ff(1)%nn_feat_4_para%Rc_M
      print*,"feature3 & feature4"

   end subroutine f34

   subroutine f5(ff_idx,ocut,m_neigh)
      integer, intent(in) :: ff_idx          ! index of ff to be loaded
      real(8), intent(out) :: ocut           ! cutoff radius
      integer, intent(out) :: m_neigh

      ! reading gen_MTP_feature.in part
      read(10,*) nn_ff(ff_idx)%nn_feat_5_para%Rc_M, nn_ff(ff_idx)%ff_max_neigh

      m_neigh = nn_ff(1)%ff_max_neigh

      read(10,*) nn_ff(ff_idx)%ff_num_type
      do itype=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) nn_ff(ff_idx)%nn_feat_5_para%iat_type(itype)
         read(10,*) nn_ff(ff_idx)%nn_feat_5_para%Rc_type(itype),nn_ff(ff_idx)%nn_feat_5_para%Rm_type(itype)
         if (nn_ff(ff_idx)%nn_feat_5_para%Rc_type(itype).gt.nn_ff(ff_idx)%nn_feat_5_para%Rc_M) then
            write(6,*) "Rc_type must be smaller than Rc_M, gen_MTP_feature.in", &
               itype,nn_ff(ff_idx)%nn_feat_5_para%Rc_type(itype),nn_ff(ff_idx)%nn_feat_5_para%Rc_M
            stop
         endif
         read(10,*) nn_ff(ff_idx)%nn_feat_5_para%nMTP_type(itype)

         numCC=0
         do kkk=1,nn_ff(ff_idx)%nn_feat_5_para%nMTP_type(itype)
            read(10,*) nn_ff(ff_idx)%nn_feat_5_para%MTP_num
            backspace(10)
            if(nn_ff(ff_idx)%nn_feat_5_para%MTP_num.gt.4) then
               write(6,*) "we only support contraction up to 4 tensors,stop",nn_ff(ff_idx)%nn_feat_5_para%MTP_num
               stop
            endif
            ! Cannot do double loop with txt
            if(nn_ff(ff_idx)%nn_feat_5_para%MTP_num.eq.1) then
               read(10,*,iostat=ierr) nn_ff(ff_idx)%nn_feat_5_para%MTP_num,&
                  (nn_ff(ff_idx)%nn_feat_5_para%mu(i),i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num),&
                  (nn_ff(ff_idx)%nn_feat_5_para%rank(i),i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num),txt,&
                  (nn_ff(ff_idx)%nn_feat_5_para%ind(j,1),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(1)),txt
            elseif(nn_ff(ff_idx)%nn_feat_5_para%MTP_num.eq.2) then
               read(10,*,iostat=ierr) nn_ff(ff_idx)%nn_feat_5_para%MTP_num,&
                  (nn_ff(ff_idx)%nn_feat_5_para%mu(i),i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num),&
                  (nn_ff(ff_idx)%nn_feat_5_para%rank(i),i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num),txt,&
                  (nn_ff(ff_idx)%nn_feat_5_para%ind(j,1),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(1)),txt,&
                  txt,(nn_ff(ff_idx)%nn_feat_5_para%ind(j,2),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(2)),txt
            elseif(nn_ff(ff_idx)%nn_feat_5_para%MTP_num.eq.3) then
               read(10,*,iostat=ierr) nn_ff(ff_idx)%nn_feat_5_para%MTP_num,&
                  (nn_ff(ff_idx)%nn_feat_5_para%mu(i),i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num),&
                  (nn_ff(ff_idx)%nn_feat_5_para%rank(i),i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num),txt,&
                  (nn_ff(ff_idx)%nn_feat_5_para%ind(j,1),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(1)),txt,&
                  txt,(nn_ff(ff_idx)%nn_feat_5_para%ind(j,2),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(2)),txt,&
                  txt,(nn_ff(ff_idx)%nn_feat_5_para%ind(j,3),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(3)),txt
            elseif(nn_ff(ff_idx)%nn_feat_5_para%MTP_num.eq.4) then
               read(10,*,iostat=ierr) nn_ff(ff_idx)%nn_feat_5_para%MTP_num,&
                  (nn_ff(ff_idx)%nn_feat_5_para%mu(i),i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num),&
                  (nn_ff(ff_idx)%nn_feat_5_para%rank(i),i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num),txt,&
                  (nn_ff(ff_idx)%nn_feat_5_para%ind(j,1),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(1)),txt,&
                  txt,(nn_ff(ff_idx)%nn_feat_5_para%ind(j,2),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(2)),txt,&
                  txt,(nn_ff(ff_idx)%nn_feat_5_para%ind(j,3),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(3)),txt,&
                  txt,(nn_ff(ff_idx)%nn_feat_5_para%ind(j,4),j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(4)),txt
            endif

            if(ierr.ne.0) then
               write(6,*) "the tensor contraction line is not correct",kkk,itype
               stop
            endif

            do i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num
               do j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(i)
                  indi(j,i)=nn_ff(ff_idx)%nn_feat_5_para%ind(j,i)/10
                  indj(j,i)=nn_ff(ff_idx)%nn_feat_5_para%ind(j,i)-indi(j,i)*10
               enddo
            enddo

            iflag_ti=0
            iflag_tj=0
            do i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num
               do j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(i)
                  if(iflag_ti(indj(j,i),indi(j,i)).ne.0) then
                     write(6,*) "contraction error", i,j,indj(j,i),indi(j,i)
                     stop
                  endif
                  iflag_ti(indj(j,i),indi(j,i))=i
                  iflag_tj(indj(j,i),indi(j,i))=j
               enddo
            enddo

            do i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num
               do j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(i)
                  if(iflag_ti(j,i).ne.indi(j,i).or.iflag_tj(j,i).ne.indj(j,i)) then
                     write(6,*) "contraction correspondence confluct",i,j
                     stop
                  endif
               enddo
            enddo

            call get_expand_MT(nn_ff(ff_idx)%ff_num_type,&
               nn_ff(ff_idx)%nn_feat_5_para%MTP_num,&
               indi,indj,nn_ff(ff_idx)%nn_feat_5_para%mu,&
               nn_ff(ff_idx)%nn_feat_5_para%rank,&
               numC,jmu_b,itype_b)
            ! numC is the MTP of this line (after expansion, due mu, and ntypes)

            numc0=1
            do i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num
               numc0=(1+nn_ff(ff_idx)%nn_feat_5_para%mu(i))*numc0*nn_ff(ff_idx)%ff_num_type
            enddo

            write(6,"('num_feat,line(kkk,itype,numc,numc(before reduce)',4(i4,1x))") kkk,itype,numc,numc0
            if(numCC+numc.gt.20000) then
               write(6,*) "too many features",numCC+numC,itype
               stop
            endif

            do kk=1,numC
               kkc=numCC+kk
               nn_numT_all(kkc,itype)=nn_ff(ff_idx)%nn_feat_5_para%MTP_num
               do i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num
                  nn_rank_all(i,kkc,itype)=nn_ff(ff_idx)%nn_feat_5_para%rank(i)
                  nn_mu_all(i,kkc,itype)=nn_ff(ff_idx)%nn_feat_5_para%mu(i)
                  nn_jmu_b_all(i,kkc,itype)=jmu_b(i,kk)  ! the mu of this kk
                  nn_itype_b_all(i,kkc,itype)=itype_b(i,kk)  ! the type of this kk
               enddo
               do i=1,nn_ff(ff_idx)%nn_feat_5_para%MTP_num
                  do j=1,nn_ff(ff_idx)%nn_feat_5_para%rank(i)
                     nn_indi_all(j,i,kkc,itype)=indi(j,i)
                     nn_indj_all(j,i,kkc,itype)=indj(j,i)
                  enddo
               enddo
            enddo
            numCC=numCC+numc
         enddo
         nn_numCC_type(itype)=numCC    ! numCC_type is the total MTP term of this itype
         write(6,*) "numCC_type",itype,nn_numCC_type(itype)
      enddo
      read(10,*) nn_ff(ff_idx)%nn_feat_5_para%E_tolerance

      ocut=nn_ff(1)%nn_feat_5_para%Rc_M
      print*,"feature5"

   end subroutine f5

   subroutine f6(ff_idx,ocut,m_neigh)
      integer, intent(in) :: ff_idx          ! index of ff to be loaded
      real(8), intent(out) :: ocut
      integer, intent(out) :: m_neigh

      ! reading gen_SNAP_feature.in part
      read(10,*) nn_ff(ff_idx)%nn_feat_6_para%Rc_M, nn_ff(ff_idx)%ff_max_neigh

      m_neigh = nn_ff(1)%ff_max_neigh

      read(10,*) nn_ff(ff_idx)%ff_num_type
      do i=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) nn_ff(ff_idx)%nn_feat_6_para%iat_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_6_para%Rc_type(i)
         if (nn_ff(ff_idx)%nn_feat_6_para%Rc_type(i).gt.nn_ff(ff_idx)%nn_feat_6_para%Rc_M) then
            write(6,*) "Rc_type must be smaller than Rc_M, gen_SNAP_feature.in", &
               i,nn_ff(ff_idx)%nn_feat_6_para%Rc_type(i),nn_ff(ff_idx)%nn_feat_6_para%Rc_M
            stop
         endif
         read(10,*) nn_ff(ff_idx)%nn_feat_6_para%snapj_type(i),&
            nn_ff(ff_idx)%nn_feat_6_para%nsnapw_type(i) ! the type of snap within this atom type, each type is indicated by one J
         do j=1,nn_ff(ff_idx)%nn_feat_6_para%nsnapw_type(i)
            read(10,*) (nn_ff(ff_idx)%nn_feat_6_para%wsnap_type(k,j,i),k=1,nn_ff(ff_idx)%ff_num_type)
         enddo
      enddo
      read(10,*) nn_ff(ff_idx)%nn_feat_6_para%E_tolerance

      ocut=nn_ff(1)%nn_feat_6_para%Rc_M
      print*,"feature6"

   end subroutine f6

   subroutine f7(ff_idx,ocut,m_neigh)
      integer, intent(in) :: ff_idx          ! index of ff to be loaded
      real(8), intent(out) :: ocut
      integer, intent(out) :: m_neigh

      ! reading gen_deepMD1_feature.in part
      read(10,*) nn_ff(ff_idx)%nn_feat_7_para%Rc_M, nn_ff(ff_idx)%ff_max_neigh

      m_neigh = nn_ff(1)%ff_max_neigh

      read(10,*) nn_ff(ff_idx)%ff_num_type
      do i=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) nn_ff(ff_idx)%nn_feat_7_para%iat_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_7_para%Rc_type(i), &
            nn_ff(ff_idx)%nn_feat_7_para%Rc2_type(i), &
            nn_ff(ff_idx)%nn_feat_7_para%Rm_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_7_para%M_type(i),nn_ff(ff_idx)%nn_feat_7_para%weight_rterm(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_7_para%M2_type(i),nn_ff(ff_idx)%nn_feat_7_para%w_dummy
         if(nn_ff(ff_idx)%nn_feat_7_para%Rc_type(i).gt.nn_ff(ff_idx)%nn_feat_7_para%Rc_M) then
            write(6,*) "Rc_type must be smaller than Rc_M, gen_deepMD1_feature.in", &
               i,nn_ff(ff_idx)%nn_feat_7_para%Rc_type(i),nn_ff(ff_idx)%nn_feat_7_para%Rc_M
            stop
         endif
      enddo
      read(10,*) nn_ff(ff_idx)%nn_feat_7_para%E_tolerance

      ocut=nn_ff(1)%nn_feat_7_para%Rc_M
      print*,"feature7"

   end subroutine f7

   subroutine f8(ff_idx,ocut,m_neigh)
      integer, intent(in) :: ff_idx          ! index of ff to be loaded
      real(8), intent(out) :: ocut
      integer, intent(out) :: m_neigh

      ! reading gen_deepMD2_feature.in part
      read(10,*) nn_ff(ff_idx)%nn_feat_8_para%Rc_M, nn_ff(ff_idx)%ff_max_neigh

      m_neigh = nn_ff(1)%ff_max_neigh

      read(10,*) nn_ff(ff_idx)%ff_num_type
      do i=1,nn_ff(ff_idx)%ff_num_type
         read(10,*) nn_ff(ff_idx)%nn_feat_8_para%iat_type(i)
         read(10,*) nn_ff(ff_idx)%nn_feat_8_para%Rc_type(i)
         if(nn_ff(ff_idx)%nn_feat_8_para%Rc_type(i).gt.nn_ff(ff_idx)%nn_feat_8_para%Rc_M) then
            write(6,*) "Rc_type must be smaller than Rc_M, gen_deepMD2_feature.in", &
               i,nn_ff(ff_idx)%nn_feat_8_para%Rc_type(i),nn_ff(ff_idx)%nn_feat_8_para%Rc_M
            stop
         endif
         read(10,*) nn_ff(ff_idx)%nn_feat_8_para%n2b_type(i),nn_ff(ff_idx)%nn_feat_8_para%weight_rterm(i)
         do j=1,nn_ff(ff_idx)%nn_feat_8_para%n2b_type(i)
            read(10,*) nn_ff(ff_idx)%nn_feat_8_para%grid2(j,i),nn_ff(ff_idx)%nn_feat_8_para%wgauss(j,i)
         enddo
      enddo
      read(10,*) nn_ff(ff_idx)%nn_feat_8_para%E_tolerance

      ocut=nn_ff(1)%nn_feat_8_para%Rc_M
      print*,"feature8"
   end subroutine f8

   subroutine nn_ff_deallocate(ff_idx) bind(c,name="nn_ff_deallocate")
      ! deallocate the ff pointers
      integer, intent(in) :: ff_idx

      do kk = 1, nn_ff(ff_idx)%ff_nfeat_type
         ! for feature1 & feature2
         if (nn_ff(ff_idx)%ff_ifeat_type(kk).eq.2) then
            deallocate(nn_ff(ff_idx)%nn_feat_1_para%grid2_2)
            deallocate(nn_ff(ff_idx)%nn_feat_2_para%grid31_2)
            deallocate(nn_ff(ff_idx)%nn_feat_2_para%grid32_2)
         endif
      enddo
      deallocate(nn_ff(ff_idx)%ff_itype_atom_sumfe)
      deallocate(nn_ff(ff_idx)%rad_atom)
      deallocate(nn_ff(ff_idx)%E_ave_vdw)
      deallocate(nn_ff(ff_idx)%wp_atom)

   end subroutine nn_ff_deallocate

end module nn_ff_mod
