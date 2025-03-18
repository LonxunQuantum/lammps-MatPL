module calc_deepMD
   !**************************************************
   !  SINGLE thread DP inference routine for LAMMPS
   !       L. Wang 2023.1
   !**************************************************

   use mod_data, only : natoms, nall, ntypes, catype, atype
   use dp_ff_mod, only : ff ! force field data
   use calc_deepmd_f, only: dp_M2,s_neigh,ds_neigh, &
      dxyz_neigh,dxyz_dx_neigh, &
      gen_deepMD_feature

   implicit none

   !********************* module variables *********************

   integer(4) :: m_neigh        !模型所使用的最大近邻数(考虑这个数是否可以不用)

   integer(4) :: num_types
   integer(4) :: dp_atype(100)  !每一种原子的原子属于第几种原子

   integer node_em(20),node_NN(20)
   real*8, allocatable,dimension(:,:,:,:,:) :: Wij_em
   real*8, allocatable,dimension(:,:,:,:) :: B_em
   real*8, allocatable,dimension(:,:,:,:) :: Wij_NN
   real*8, allocatable,dimension(:,:,:) :: B_NN
   real*8, allocatable,dimension(:,:,:) :: W_res_NN
   integer nodeMM_em,nlayer_em,nodeMM_NN,nlayer_NN
   integer iflag_resNN(100)

   integer, allocatable :: iitype_list(:)
   logical, allocatable :: itype_added(:)

   !********************* module variables *********************
contains

   subroutine load_model_deepMD(ff_idx)
      integer, intent(in) :: ff_idx
      integer :: i, itype, ll
      integer :: sliced_itype1, sliced_itype2, itype1, itype2, j1, j2, j
      integer :: iitype_count  ! Counter for matching itypes

      ! ************************************
      !         fetch dp_mod data
      ! ************************************

      ! **************** "read" feat.info ********************

      m_neigh = ff(ff_idx)%dp_ff_max_neigh ! neighbor buffer size

      ! get atom type
      num_types = ff(ff_idx)%dp_ff_num_type
      do i=1,num_types
         dp_atype(i) = ff(ff_idx)%dp_ff_itype_atom(i)
      enddo

      ! ***************** atom_type to itype_list *************
      allocate(iitype_list(num_types))
      allocate(itype_added(num_types))

      iitype_count = 0
      itype_added = .false.
      do i = 1, natoms
        !  write(*,*) 'atype',atype(i)
         do itype = 1, num_types
            if (.not. itype_added(itype) .and. dp_atype(itype)==atype(i)) then
               itype_added(itype) = .true.
               iitype_count = iitype_count + 1
               iitype_list(iitype_count) = itype
            end if
         end do
      enddo
      ! write(*,*) 'iitype_list',iitype_list
      ! ***************** end atom_type to itype_list *********

      ! ****************** read weights********************
      ! ****************** embedding net ******************
      nlayer_em  = ff(ff_idx)%dp_ff_nlayer_em
      node_em(1:nlayer_em+1) = ff(ff_idx)%dp_ff_node_em(1:nlayer_em+1)

      nodeMM_em = ff(ff_idx)%dp_ff_nodeMM_em


      allocate(Wij_em(nodeMM_em,nodeMM_em,nlayer_em,iitype_count,iitype_count))
      allocate(B_em(nodeMM_em,nlayer_em,iitype_count,iitype_count))
      ! Wij_em = ff(ff_idx)%dp_ff_Wij_em
      ! B_em = ff(ff_idx)%dp_ff_B_em

      ! allocate(Wij_em(nodeMM_em,nodeMM_em,nlayer_em,num_types,num_types))
      ! allocate(B_em(nodeMM_em,nlayer_em,num_types,num_types))
      do i=1,iitype_count
         sliced_itype1 = iitype_list(i)
         do j=1,iitype_count
            sliced_itype2 = iitype_list(j)
            ! write(*,*) "sliced_itype1",i,j,sliced_itype1,sliced_itype2
            do ll=1,nlayer_em
               do j1=1,node_em(ll)
                  do j2=1,node_em(ll+1)
                     Wij_em(j1,j2,ll,j,i) = ff(ff_idx)%dp_ff_Wij_em(j1,j2,ll,sliced_itype2,sliced_itype1)
                  enddo
               enddo
               do j2=1,node_em(ll+1)
                  B_em(j2,ll,j,i) = ff(ff_idx)%dp_ff_B_em(j2,ll,sliced_itype2,sliced_itype1)
               enddo
            enddo
         enddo
      enddo
      ! write(*,*) "B_em",B_em
      ! write(*, *) 'Embedding net layers: ', nlayer_em               ! 3
      ! write(*, *) 'Embedding net nodes: ', node_em(1:nlayer_em+1)   ! 1 25 25 25
      ! write(*, *) 'Embedding net nodeMM: ', nodeMM_em               ! 25

      ! ****************** fitting net ******************
      nlayer_NN = ff(ff_idx)%dp_ff_nlayer_NN
      node_nn(1:nlayer_NN+1) = ff(ff_idx)%dp_ff_node_nn(1:nlayer_NN+1)

      nodeMM_nn = ff(ff_idx)%dp_ff_nodeMM_nn

      ! allocate(Wij_nn(nodeMM_nn,nodeMM_nn,nlayer_NN,num_types))
      ! allocate(B_nn(nodeMM_nn,nlayer_NN,num_types))
      allocate(Wij_nn(nodeMM_nn,nodeMM_nn,nlayer_NN,iitype_count))
      allocate(B_nn(nodeMM_nn,nlayer_NN,iitype_count))
      ! Wij_nn = ff(ff_idx)%dp_ff_Wij_nn
      ! B_nn = ff(ff_idx)%dp_ff_B_nn

      do i=1,iitype_count
         sliced_itype1 = iitype_list(i)
         do ll=1,nlayer_NN
            do j1=1,node_nn(ll)
               do j2=1,node_nn(ll+1)
                  Wij_nn(j1,j2,ll,i) = ff(ff_idx)%dp_ff_Wij_nn(j1,j2,ll,sliced_itype1)
               enddo
            enddo
            do j2=1,node_nn(ll+1)
               B_nn(j2,ll,i) = ff(ff_idx)%dp_ff_B_nn(j2,ll,sliced_itype1)
            enddo
         enddo
      enddo
      ! write(*, *) 'Fitting net layers: ', nlayer_NN                 ! 4
      ! write(*, *) 'Fitting net nodes: ', node_nn(1:nlayer_NN+1)     ! 400 50 50 50 1
      ! write(*, *) 'Fitting net nodeMM: ', nodeMM_nn                 ! 400

      ! ****************** reconnect para ******************
      ! allocate(W_res_NN(nodeMM_nn,nlayer_NN+1,num_types))
      allocate(W_res_NN(nodeMM_nn,nlayer_NN+1,iitype_count))

      iflag_resNN(1:nlayer_NN+1) = ff(ff_idx)%dp_ff_iflag_resNN(1:nlayer_NN+1)

      do itype=1,iitype_count
         sliced_itype1 = iitype_list(itype)
         do ll=1,nlayer_NN+1
            if(iflag_resNN(ll).eq.1) then
               W_res_NN(1:node_nn(ll),ll,itype) = &
                  ff(ff_idx)%dp_ff_W_res_NN(1:node_nn(ll),ll,sliced_itype1)
            endif
         enddo
      enddo

   end subroutine load_model_deepMD

   subroutine set_image_info_deepMD(atype_check)
      logical, intent(in) :: atype_check
      integer :: i, iitype, itype

      if (atype_check) then
         do i = 1, natoms
            iitype = 0
            do itype = 1, num_types
               if (dp_atype(itype)==atype(i)) then
                  iitype = itype
               end if
            end do
            if (iitype==0) then
               write (6, *) 'this type not found', atype(i)
            end if
         enddo
      end if

   end subroutine set_image_info_deepMD

   subroutine cal_energy_force_deepMD(num_neigh,list_neigh,dR_neigh,Etot,fatom,virial)

      integer, dimension(ntypes,natoms), intent(in) :: num_neigh
      integer, dimension(m_neigh,ntypes,natoms), intent(in) :: list_neigh
      real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh
      real*8, intent(out) :: Etot
      real*8, dimension(3, nall), intent(out) :: fatom
      real*8, dimension(6), intent(out) :: virial

      integer natom_n_type(50)
      integer,allocatable,dimension(:,:) :: iat_ind
      real(8),allocatable,dimension(:,:,:) :: f_in,f_out,f_back,f_back0
      real(8),allocatable,dimension(:,:,:,:) :: f_d
      real(8),allocatable,dimension(:,:) :: energy_type
      real(8),allocatable,dimension(:,:,:) :: ss
      real(8),allocatable,dimension(:,:,:) :: f_in_NN,f_out_NN,f_d_NN
      real(8),allocatable,dimension(:,:,:,:,:,:) :: d_ss
      real(8),allocatable,dimension(:,:,:,:) :: dE_dx
      real(8),allocatable,dimension(:,:,:,:,:) :: d_ss_fout
      real(8),allocatable,dimension(:,:) :: dE_dfout
      real(8),allocatable,dimension(:,:,:) :: f_back0_em,f_back_em

      integer itype,i,j,ii,jj,kk
      integer iat2
      integer natom_m_type,jjm,itype1,itype2,iat,ll,num,nn1,k,m,m2,k1,k2
      integer j1,j2
      integer iflag_ghost_neigh,neigh_add

      real*8 tsum
      real*8 pi,dE,dFx,dFy,dFz
      real*8 rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd,d,w22_1,w22_2,w22F_1,w22F_2
      real*8 x, y
      real*8 sum1,sum2,sum3
      real*8 d_sum,d_sum2
      real*8 fact1

      integer t

      iflag_ghost_neigh=1   ! 1: use the ghost neigh, 0: not use the ghost neigh

      pi=4*datan(1.d0)

      ! *************** calc DP feature ***************
      call gen_deepMD_feature(num_neigh,dR_neigh)


      ! write(*,*) " --- cal CHECK --- "
      ! do i = 1, natoms
      !   write(*,*) i
      !   do t = 1, ntypes
      !     write(*,*) "  ", t, num_neigh(t,i)
      !     do j = 1, num_neigh(t,i)
      !       write(*,"('     ', A, 2(I5,3X), 3(F12.6, 3X))") "     ", &
      !         j,  list_neigh(j,t,i), dR_neigh(1,j,t,i), dR_neigh(2,j,t,i), dR_neigh(3,j,t,i)
      !     enddo
      !   enddo
      ! enddo

      ! each type atom number
      natom_n_type= 0
      do i = 1, natoms
         itype = catype(i)
         natom_n_type(itype) = natom_n_type(itype) + 1
      enddo

      ! max atom number of one type
      natom_m_type=0
      do itype=1,ntypes
         if(natom_n_type(itype).gt.natom_m_type) natom_m_type=natom_n_type(itype)
      enddo

      allocate(iat_ind(natom_m_type,ntypes))

      ! classify atom with atom type
      natom_n_type= 0
      do i = 1, natoms
         itype = catype(i)
         natom_n_type(itype) = natom_n_type(itype) + 1
         iat_ind(natom_n_type(itype),itype)=i
      enddo
      ! write(*,*) "itype", itype
      ! write(*,*) "natom_n_type", natom_n_type
      ! write(*,*) "iat_ind", iat_ind

      ! the max total neighbor number of one type
      jjm=0
      do itype1=1,ntypes
         do itype2=1,ntypes
            jj=0
            do i=1,natom_n_type(itype1)
               iat=iat_ind(i,itype1)
               neigh_add=0
               if(iflag_ghost_neigh.eq.1.and.num_neigh(itype2,iat).lt.m_neigh) neigh_add=1
               ! the neigh_add is the ghost neighbor
               do j=1,num_neigh(itype2,iat)+neigh_add
                  jj=jj+1
               enddo
            enddo
            if(jj.gt.jjm) jjm=jj
         enddo
      enddo

      nodeMM_em=0
      do ll=1,nlayer_em+1
         if(node_em(ll).gt.nodeMM_em) nodeMM_em=node_em(ll)
      enddo

      nodeMM_NN=0
      do ll=1,nlayer_NN+1
         if(node_NN(ll).gt.nodeMM_NN) nodeMM_NN=node_NN(ll)
         ! write(*,*) "nodeMM_NN", nodeMM_NN     ! 16*25
      enddo

      allocate(f_in(nodeMM_em,jjm,nlayer_em+1))
      allocate(f_out(nodeMM_em,jjm,nlayer_em+1))
      allocate(f_d(nodeMM_em,jjm,nlayer_em+1,ntypes))
      allocate(f_back0_em(nodeMM_em,jjm,nlayer_em+1))
      allocate(f_back_em(nodeMM_em,jjm,nlayer_em+1))
      allocate(ss(4,node_em(nlayer_em+1),natom_m_type))
      allocate(f_in_NN(nodeMM_NN,natom_m_type,nlayer_NN+1))
      allocate(f_out_NN(nodeMM_NN,natom_m_type,nlayer_NN+1))
      allocate(f_back(nodeMM_NN,natom_m_type,nlayer_NN+1))
      allocate(f_back0(nodeMM_NN,natom_m_type,nlayer_NN+1))
      allocate(f_d_NN(nodeMM_NN,natom_m_type,nlayer_NN+1))
      allocate(energy_type(natom_m_type,ntypes))
      allocate(d_ss(4,3,nodeMM_em,m_neigh,ntypes,natom_m_type))
      allocate(dE_dx(3,m_neigh,ntypes,natoms))
      allocate(d_ss_fout(4,nodeMM_em,m_neigh,ntypes,natom_m_type))
      allocate(dE_dfout(nodeMM_em,jjm))

      dE_dx=0.d0

      do itype1=1,ntypes

         f_in_NN=0.d0
         ss=0.d0
         d_ss=0.d0
         d_ss_fout=0.d0

         do itype2=1,ntypes

            jj=0
            do i=1,natom_n_type(itype1)
               iat=iat_ind(i,itype1)
               neigh_add=0
               if(iflag_ghost_neigh.eq.1.and.num_neigh(itype2,iat).lt.m_neigh) neigh_add=1
               do j=1,num_neigh(itype2,iat)+neigh_add
                  jj=jj+1
                  f_in(1,jj,1)=s_neigh(j,itype2,iat)
                  f_out(1,jj,1)=s_neigh(j,itype2,iat)
                  f_d(1,jj,1,itype2)=1.d0
               enddo
            enddo

            num=jj     ! the same (itype2,itype1), all the neigh, and all the atom belong to this CPU

            do ll=1,nlayer_em

               ! take weights
               call dgemm('T', 'N', node_em(ll+1),num,node_em(ll), 1.d0,  &
                  Wij_em(1,1,ll,itype2,itype1),nodeMM_em,f_out(1,1,ll),nodeMM_em,0.d0,f_in(1,1,ll+1),nodeMM_em)
               ! add bias
               do i=1,num
                  do j=1,node_em(ll+1)
                     f_in(j,i,ll+1)=f_in(j,i,ll+1)+B_em(j,ll,itype2,itype1)
                  enddo
               enddo

               ! activation
               do i=1,num
                  do j=1,node_em(ll+1)
                     x=f_in(j,i,ll+1)
                     if(x.gt.20.d0) then
                        y=1.d0
                     elseif(x.gt.-20.d0.and.x.le.20.d0) then
                        y=(exp(x)-exp(-x))/(exp(x)+exp(-x))
                     elseif(x.lt.-20.d0) then
                        y=-1.d0
                     endif
                     ! f_out(j, i, ll) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))  ! tanh ! maybe if softplus, sigmoid, tanh
                     f_out(j, i, ll+1) = y
                     f_d(j, i, ll+1,itype2) = 1.0d0 - f_out(j,i,ll+1)*f_out(j,i,ll+1)
                  enddo
               enddo

               ! ************************************
               !     reconnect w. no weights
               ! ************************************

               if(node_em(ll+1).eq.2*node_em(ll)) then
                  do i=1,num
                     do j=1,node_em(ll)
                        f_out(j,i,ll+1)=f_out(j,i,ll+1)+f_out(j,i,ll)
                        f_out(j+node_em(ll),i,ll+1)=f_out(j+node_em(ll),i,ll+1)+f_out(j,i,ll)
                     enddo
                  enddo
               endif

               !  This is the reconnect
               if(node_em(ll+1).eq.node_em(ll)) then
                  do i=1,num
                     do j=1,node_em(ll)
                        f_out(j,i,ll+1)=f_out(j,i,ll+1)+f_out(j,i,ll)
                        ! write(*,*) 'f_out(j,i,ll+1)',i, j, f_out(j,i,ll+1) ! 10 * 3 * 25 * 16
                     enddo
                  enddo
               endif
            enddo ! ll end

            ! embeding input
            ! write(*,*) 'itype1', itype1
            ! write(*,*) 'itype2', itype2
            ! do i=1,num
            !   write(*,*) 'f_out',i
            !   write(*,*) ,f_out(:,i,1)
            ! enddo



            ! *************************************************
            !   symmetry conserving operations (G*R*R^T*G^T)
            ! *************************************************
            nn1=node_em(nlayer_em+1)
            jj=0

            do i=1,natom_n_type(itype1)
               iat=iat_ind(i,itype1)
               neigh_add=0
               if(iflag_ghost_neigh.eq.1.and.num_neigh(itype2,iat).lt.m_neigh) neigh_add=1
               do j=1,num_neigh(itype2,iat)+neigh_add   ! j is sum over
                  jj=jj+1
                  fact1=1

                  if(neigh_add.eq.1.and.j.eq.num_neigh(itype2,iat)+neigh_add) then  ! the ghost neighbor
                     fact1=m_neigh-num_neigh(itype2,iat)
                  endif

                  do k=1,nn1
                     do m=1,4
                        ! ss(m,k,i)=ss(m,k,i)+dxyz_neigh(m,j,itype2,iat)*f_in(k,jj,nlayer_em+1)
                        ss(m,k,i)=ss(m,k,i)+dxyz_neigh(m,j,itype2,iat)*f_out(k,jj,nlayer_em+1)*fact1

                        d_ss_fout(m,k,j,itype2,i)=d_ss_fout(m,k,j,itype2,i)+dxyz_neigh(m,j,itype2,iat)*fact1

                        !if(j.ne.num_neigh(itype2,iat)+neigh_add) then
                        ! 2023.3 altered
                        if(j.ne.num_neigh(itype2,iat)+1) then
                           do m2=1,3
                              !! It is possible to do this later, to save memory
                              d_ss(m,m2,k,j,itype2,i)=d_ss(m,m2,k,j,itype2,i)+dxyz_dx_neigh(m2,m,j,itype2,iat)*f_out(k,jj,nlayer_em+1)
                              ! d_ss is to assume s_neigh, thus f_out is fixed
                              !!  d_ss(m,m2,k,i)=d_SS(m,k,i)/d_x(m2,k,i)
                              ! d_ss_fout is to take the derivative with respect to f_out (only s_neigh, thus
                              ! f_out is changing.
                           enddo
                        endif
                     enddo
                  enddo
               enddo ! j end
            enddo ! i end
            ! We need to double check, is it first sum in the ss for different itype2,

            ! or sum over ss*ss for different itype2
         enddo ! itype2 end

         ss=ss/(ntypes*m_neigh)
         d_ss=d_ss/(ntypes*m_neigh)
         d_ss_fout=d_ss_fout/(ntypes*m_neigh)

         nn1=node_em(nlayer_em+1)

         do i=1,natom_n_type(itype1)
            do k1=1,nn1
               do k2=1,dp_M2   ! fixed, first index
                  kk=(k1-1)*dp_M2+k2   ! NN feature index
                  tsum=0.d0
                  do m=1,4
                     tsum=tsum+ss(m,k1,i)*ss(m,k2,i)
                  enddo
                  f_in_NN(kk,i,1)=f_in_NN(kk,i,1)+tsum    ! this is sum over itype2
               enddo
            enddo
         enddo

         if(node_NN(1).ne.nn1*dp_M2) then
            write(6,*) "node_NN(1) not equal to nn1*dp_M2,stop",node_NN(1),nn1*dp_M2
            stop
         endif

         num=natom_n_type(itype1)

         ! f_in_NN(nodeMM_NN,natom_m_type,nlayer_NN+1)
         ! dim: (max node dim, max natoms in all types, nlayer + 1)

         ! write(*,*) "printing dbg info: fitting net input"
         ! write(*,*) "type", itype1
         ! write(*,*) "num", num
         ! write(*,*) f_in_NN(1:50,1,1)

         ! *******************************************
         !    propogation through the fitting net
         ! *******************************************
         do ll=1,nlayer_NN
            if(ll.gt.1) then
               do i=1,num
                  do j=1,node_NN(ll)
                     x=f_in_NN(j,i,ll)
                     if(x.gt.20.d0) then
                        y=1.d0
                     elseif(x.gt.-20.d0.and.x.le.20.d0) then
                        y=(exp(x)-exp(-x))/(exp(x)+exp(-x))
                     elseif(x.lt.-20.d0) then
                        y=-1.d0
                     endif
                     ! f_out_NN(j, i, ll) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))
                     ! tanh ! maybe if softplus, sigmoid, tanh
                     f_out_NN(j, i, ll) = y
                     f_d_NN(j, i, ll) = 1.0d0 - f_out_NN(j,i,ll)*f_out_NN(j,i,ll)
                  enddo
               enddo
            else
               do i=1,num
                  do j=1,node_NN(ll)
                     f_out_NN(j,i,ll)=f_in_NN(j,i,ll)
                     f_d_NN(j,i,ll)=1.d0
                  enddo
               enddo
            endif

            !!!!! reconnect
            if(iflag_resNN(ll).eq.1) then
               do i=1,num
                  do j=1,node_NN(ll)
                     f_out_NN(j,i,ll)=f_out_NN(j,i,ll)*W_res_NN(j,ll,itype1)+f_out_NN(j,i,ll-1)
                  enddo
               enddo
            endif
            !endif

            call dgemm('T', 'N', node_NN(ll+1),num,node_NN(ll), 1.d0,  &
               Wij_NN(1,1,ll,itype1),nodeMM_NN,f_out_NN(1,1,ll),nodeMM_NN,0.d0,f_in_NN(1,1,ll+1),nodeMM_NN)

            do i=1,num
               do j=1,node_NN(ll+1)
                  f_in_NN(j,i,ll+1)=f_in_NN(j,i,ll+1)+B_NN(j,ll,itype1)
               enddo
            enddo
         enddo ! ll end

         if(node_NN(nlayer_NN+1).ne.1) then
            write(6,*) "node_NN(nlayer_NN+1).ne.1,stop",node_NN(nlayer_NN+1)
            stop
         endif

         ! copying atomic energy
         do i=1,natom_n_type(itype1)
            energy_type(i,itype1)=f_in_NN(1,i,nlayer_NN+1)
         enddo

         ! ****************************************************************************
         !   Now, we do the back propagation
         !   f_back0(j,i,ll)=dE/d_(f_out(j,i,ll))
         !   f_back(j,i,ll)=dE/d_(f_in(j,i,ll))=f_fac0*df(j,i,ll)*W_res
         !   f_out(j,i,ll)=sigma(f_in(j,i,ll))*W_res+f_out(j,i,ll-1)  ! if there are res
         !   f_out(j,i,ll)=sigma(f_in(j,i,ll))                        ! if no rest
         !   f_in(j,i,ll+1)=W(j2,i,ll)*f_out(j2,i,ll)+B(j,i,ii)
         ! ****************************************************************************

         do i=1,num
            do j=1,node_NN(nlayer_NN)
               f_back0(j,i,nlayer_NN)=Wij_NN(j,1,nlayer_NN,itype1)
            enddo
         enddo

         do ll=nlayer_NN,2,-1
            do i=1,num
               do j=1,node_NN(ll)
                  f_back(j,i,ll)=f_back0(j,i,ll)*f_d_NN(j,i,ll)
                  !  f_back0=dE/d_(f_out(ll))
                  !  f_back=dE/d_(f_in(ll))
               enddo
            enddo

            if(iflag_resNN(ll).eq.1) then
               do i=1,num
                  do j=1,node_NN(ll)
                     f_back(j,i,ll)=f_back(j,i,ll)*W_res_NN(j,ll,itype1)
                  enddo
               enddo
            endif

            call dgemm('N', 'N', node_NN(ll-1),num,node_NN(ll),1.d0,  &
               Wij_NN(1,1,ll-1,itype1),nodeMM_NN,f_back(1,1,ll),nodeMM_NN,0.d0,f_back0(1,1,ll-1),nodeMM_NN)

            if(iflag_resNN(ll).eq.1) then
               do i=1,num
                  do j=1,node_NN(ll-1)
                     f_back0(j,i,ll-1)=f_back0(j,i,ll-1)+f_back0(j,i,ll)
                  enddo
               enddo
            endif
         enddo ! ll end

         !   f_back0(j,i,1)=dE/d_(f_out(j,i,1))=dE/d_(f_in(j,i,1))=dE/df_NN
         !   j is feature index, i, the itype1 atom index
         !   Now, there are two terms for the force:
         !   (dE/df_NN)*(df_NN/d_x)
         !   (dE/df_NN)*(df_NN/d_fem)*(d_fem/d_s)*(d_s/d_x)
         !   let't do the first term
         !
         !   This part is a bit expensive, it will be nice to accelerate it
         nn1=node_em(nlayer_em+1)

         do itype2=1,ntypes
            do i=1,natom_n_type(itype1)
               iat=iat_ind(i,itype1)
               do j=1,num_neigh(itype2,iat)
                  do m2=1,3
                     d_sum2=0.d0
                     do k1=1,nn1
                        do k2=1,dp_M2   ! fixed, first index
                           kk=(k1-1)*dp_M2+k2   ! NN feature index
                           ! d_sum=0.d0
                           ! do m=1,4
                           ! d_sum=d_sum+d_ss(m,m2,k1,j,itype2,i)*ss(m,k2,i)+ss(m,k1,i)*d_ss(m,m2,k2,j,itype2,i)
                           ! enddo
                           d_sum = d_ss(1,m2,k1,j,itype2,i)*ss(1,k2,i)+ss(1,k1,i)*d_ss(1,m2,k2,j,itype2,i)+ &
                              d_ss(2,m2,k1,j,itype2,i)*ss(2,k2,i)+ss(2,k1,i)*d_ss(2,m2,k2,j,itype2,i)+ &
                              d_ss(3,m2,k1,j,itype2,i)*ss(3,k2,i)+ss(3,k1,i)*d_ss(3,m2,k2,j,itype2,i)+ &
                              d_ss(4,m2,k1,j,itype2,i)*ss(4,k2,i)+ss(4,k1,i)*d_ss(4,m2,k2,j,itype2,i)

                           d_sum2=d_sum2+d_sum*f_back0(kk,i,1)
                        enddo
                     enddo
                     ! *** This is for assuming s_neigh, is fixed ***
                     dE_dx(m2,j,itype2,iat)=dE_dx(m2,j,itype2,iat)+d_sum2
                  enddo
               enddo
            enddo
         enddo ! itype2 end

         nn1=node_em(nlayer_em+1)

         do itype2=1,ntypes

            dE_dfout=0.d0
            jj=0

            do i=1,natom_n_type(itype1)
               iat=iat_ind(i,itype1)
               neigh_add=0

               ! altered 2023.3
               if(iflag_ghost_neigh.eq.1.and.num_neigh(itype2,iat).lt.m_neigh) then
                  neigh_add=1
               endif

               do j=1,num_neigh(itype2,iat) + neigh_add
                  !do j=1,num_neigh(itype2,iat)
                  jj=jj+1

                  d_sum2=0.d0
                  do k1=1,nn1
                     do k2=1,dp_M2   ! fixed, first index
                        kk=(k1-1)*dp_M2+k2   ! NN feature index
                        d_sum=0.d0

                        do m=1,4
                           d_sum=d_sum+d_ss_fout(m,k1,j,itype2,i)*ss(m,k2,i)
                        enddo

                        dE_dfout(k1,jj)=dE_dfout(k1,jj)+d_sum*f_back0(kk,i,1)
                        d_sum=0.d0

                        do m=1,4
                           d_sum=d_sum+ss(m,k1,i)*d_ss_fout(m,k2,j,itype2,i)
                        enddo

                        dE_dfout(k2,jj)=dE_dfout(k2,jj)+d_sum*f_back0(kk,i,1)
                     enddo
                  enddo
               enddo
            enddo ! i end

            num=jj

            do i=1,num
               do j=1,node_em(nlayer_em+1)
                  f_back0_em(j,i,nlayer_em+1)=dE_dfout(j,i)
               enddo
            enddo

            do ll=nlayer_em+1,2,-1
               do i=1,num
                  do j=1,node_em(ll)
                     f_back_em(j,i,ll)=f_back0_em(j,i,ll)*f_d(j,i,ll,itype2)
                  enddo
               enddo

               call dgemm('N', 'N', node_em(ll-1),num,node_em(ll),1.d0,  &
                  Wij_em(1,1,ll-1,itype2,itype1),nodeMM_em,f_back_em(1,1,ll),nodeMM_em,0.d0,   &
                  f_back0_em(1,1,ll-1),nodeMM_em)

               !reconnect. Controled by dp_ff data mode
               if(node_em(ll).eq.2*node_em(ll-1)) then
                  do i=1,num
                     do j=1,node_em(ll-1)
                        f_back0_em(j,i,ll-1)=f_back0_em(j,i,ll-1)+f_back0_em(j,i,ll)+ &
                           f_back0_em(j+node_em(ll-1),i,ll)
                     enddo
                  enddo
               endif

               !reconnect
               if(node_em(ll).eq.node_em(ll-1)) then
                  do i=1,num
                     do j=1,node_em(ll-1)
                        f_back0_em(j,i,ll-1)=f_back0_em(j,i,ll-1)+f_back0_em(j,i,ll)
                     enddo
                  enddo
               endif
            enddo ! ll end

            jj=0

            do i=1,natom_n_type(itype1)
               iat=iat_ind(i,itype1)
               ! alter 2023.3
               neigh_add=0
               if(iflag_ghost_neigh.eq.1.and.num_neigh(itype2,iat).lt.m_neigh) neigh_add=1
               !do j=1,num_neigh(itype2,iat)
               do j=1,num_neigh(itype2,iat) + neigh_add
                  jj=jj+1

                  do m2=1,3
                     dE_dx(m2,j,itype2,iat)=dE_dx(m2,j,itype2,iat)+f_back0_em(1,jj,1)*ds_neigh(m2,j,itype2,iat)
                     !1 This is the derivative through s_neigh change, has through the back
                     !propagation of the embedding net
                  enddo
               enddo
            enddo ! i end
         enddo ! itype2 end
         !  back propagation for the embedding net.
      enddo ! itype1 end

      Etot=0.d0

      do itype=1,ntypes
         do i=1,natom_n_type(itype)
            Etot=Etot+energy_type(i,itype)
         enddo
      enddo

      !write(*,*) "Etot=", Etot

      do itype1=1,ntypes
         do i=1,natom_n_type(itype1)
            iat=iat_ind(i,itype1)
            do itype2=1,ntypes
               do j=1,num_neigh(itype2,iat)
                  iat2=list_neigh(j,itype2,iat)
                  fatom(:,iat2)=fatom(:,iat2)+dE_dx(:,j,itype2,iat)
                  fatom(:,iat)=fatom(:,iat)-dE_dx(:,j,itype2,iat)
                  ! the centeratom is a negative derivative
                  virial(1) = virial(1) + dR_neigh(1,j,itype2,iat)*dE_dx(1,j,itype2,iat)
                  virial(2) = virial(2) + dR_neigh(2,j,itype2,iat)*dE_dx(2,j,itype2,iat)
                  virial(3) = virial(3) + dR_neigh(3,j,itype2,iat)*dE_dx(3,j,itype2,iat)
                  virial(4) = virial(4) + dR_neigh(1,j,itype2,iat)*dE_dx(2,j,itype2,iat)
                  virial(5) = virial(5) + dR_neigh(1,j,itype2,iat)*dE_dx(3,j,itype2,iat)
                  virial(6) = virial(6) + dR_neigh(2,j,itype2,iat)*dE_dx(3,j,itype2,iat)
               enddo
            enddo
         enddo
      enddo

      ! write(*,*) "CHECKFORCE"
      ! do i=1,nall
      !   write(6,"(I5,3(1X,F12.6))") i, fatom(:,i)
      ! enddo


      deallocate(iat_ind)

      deallocate(f_in)
      deallocate(f_out)
      deallocate(f_d)
      deallocate(f_back0_em)
      deallocate(f_back_em)
      deallocate(ss)
      deallocate(f_in_NN)
      deallocate(f_out_NN)
      deallocate(f_back)
      deallocate(f_back0)
      deallocate(f_d_NN)
      deallocate(energy_type)
      deallocate(d_ss)
      deallocate(dE_dx)
      deallocate(d_ss_fout)
      deallocate(dE_dfout)

      deallocate(Wij_em)
      deallocate(Wij_NN)
      deallocate(B_em)
      deallocate(B_NN)
      deallocate(W_res_NN)

!$    for gen_deepMD_feature.f90
      deallocate(s_neigh)
      deallocate(ds_neigh)
      deallocate(dxyz_neigh)
      deallocate(dxyz_dx_neigh)

      deallocate(iitype_list)
      deallocate(itype_added)
      return

   end subroutine cal_energy_force_deepMD

end module calc_deepMD
