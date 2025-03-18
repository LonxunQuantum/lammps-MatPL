module calc_NN

   use mod_data, only : natoms, nall, ntypes, catype, atype
   use nn_ff_mod, only : nn_ff ! force field data
   !implicit double precision (a-h, o-z)
   implicit none
   !!!!!!!!!!!!!          以下为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   integer :: m_neigh                                       !模型所使用的最大近邻数(考虑这个数是否可以不用)
   integer :: nfeat1m                                       !不同种原子的原始feature数目中最大者(目前似无意义)
   integer,allocatable,dimension(:) :: nfeat1               !各种原子的原始feature数目

   integer,allocatable,dimension(:) :: num             !属于每种原子的原子个数，但似乎在calc_linear中无用
   integer,allocatable,dimension(:) :: itype_atom      !每一种原子的原子属于第几种原子
!    integer,allocatable,dimension(:) :: iatom_type      !每种原子的种类，即序数在种类列表中的序数

!    real(8),allocatable,dimension(:) :: energy_pred_NN       !每个原子的能量预测值
   real(8),allocatable,dimension(:) :: energy_pred_tmp        !每个原子的能量预测值
!    real(8),allocatable,dimension(:,:) :: force_pred_NN       !每个原子的受力预测值
!    real(8),allocatable,dimension(:,:) :: force_pred_tmp       !每个原子的受力预测值
!    real(8) :: etot_pred_NN
   real(8), allocatable, dimension(:) ::  const_fa,const_fb,const_fc,const_fx,const_fy,const_fz
   integer,allocatable, dimension(:) :: direction,add_force_atom,const_force_atom
   integer :: add_force_num,const_force_num
   real(8) :: alpha, y1, z1
   ! INTEGER*4  access, status
   logical*2 :: alive

   ! real(8),allocatable,dimension(:) :: rad_atom,wp_atom
   real(8),allocatable,dimension(:) :: rad_atom,E_ave_vdw
   real(8),allocatable,dimension(:,:,:) :: wp_atom
   integer :: nfeat_type_n
   integer :: ifeat_type_n(10)
   real(8), allocatable,dimension(:,:) :: a_scaler,b_scaler
   integer, allocatable,dimension(:,:) :: nodeNN
   real(8), allocatable,dimension(:,:,:,:) :: Wij_nn
   real(8), allocatable,dimension(:,:,:) :: B_nn
   integer :: nodeMM,nlayer

   !!!!!!!!!!!!!          以上为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! loop index
   integer :: i,itype,ii

contains

   subroutine load_model_NN(ff_idx)

      integer, intent(in) :: ff_idx

      ! ************************************
      !         fetch ff_mod data
      ! ************************************
      if (allocated(itype_atom)) deallocate(itype_atom)
      if (allocated(nfeat1)) deallocate(nfeat1)
      if (allocated(rad_atom)) deallocate(rad_atom)
      if (allocated(wp_atom)) deallocate(wp_atom)
      if (allocated(E_ave_vdw)) deallocate(E_ave_vdw)
      if (allocated(num)) deallocate(num)                              !image数据,在此处allocate，但在set_image_info中赋值
      if (allocated(add_force_atom)) deallocate(add_force_atom)
      if (allocated(const_fa)) deallocate(const_fa)
      if (allocated(const_fb)) deallocate(const_fb)
      if (allocated(const_fc)) deallocate(const_fc)

      allocate(itype_atom(ntypes))
      allocate(nfeat1(ntypes))
      allocate(num(ntypes))                              !image数据,在此处allocate，但在set_image_info中赋值
      allocate(rad_atom(ntypes))
      allocate(E_ave_vdw(ntypes))
      allocate(wp_atom(ntypes,ntypes,2))

      ! **************** read feat.info ********************
      nfeat_type_n=nn_ff(ff_idx)%ff_nfeat_type
      ifeat_type_n=nn_ff(ff_idx)%ff_ifeat_type
      m_neigh=nn_ff(ff_idx)%ff_max_neigh
      do i=1,ntypes
         itype_atom(i)=nn_ff(ff_idx)%ff_itype_atom_sumfe(i,1)
         nfeat1(i)=nn_ff(ff_idx)%ff_itype_atom_sumfe(i,2)   ! these nfeat1 include all ftype
      enddo
      ! end reading feat.info

      nfeat1m=0   ! the original feature
      ! determine max number of feature. For buffer allocation
      do i=1,ntypes
         if (nfeat1(i).gt.nfeat1m) nfeat1m=nfeat1(i)
      enddo

      ! ****************** read vdw ************************
      ! ************* NOT IN USE NOW **************
      if (1.eq.0) then
         rad_atom=nn_ff(ff_idx)%rad_atom
         E_ave_vdw=nn_ff(ff_idx)%E_ave_vdw
         wp_atom=nn_ff(ff_idx)%wp_atom
      endif
      ! ****************** read vdw end************************

      ! ****************** read Wij starts************************
      nlayer=nn_ff(ff_idx)%ff_nlayer

      if(.not.allocated(nodeNN)) then
         allocate(nodeNN(nlayer+1,ntypes))
      endif
      if(.not.allocated(a_scaler)) then
         allocate(a_scaler(nfeat1m,ntypes))
         allocate(b_scaler(nfeat1m,ntypes))
         allocate(Wij_nn(nodeMM,nodeMM,nlayer,ntypes))
         allocate(B_nn(nodeMM,nlayer,ntypes))
      endif

      ! start load parameters
      do itype=1,ntypes
         do ii=1,nlayer
            nodeNN(ii,itype)=nn_ff(ff_idx)%ff_nodeNN(ii,itype)
            nodeNN(ii+1,itype)=nn_ff(ff_idx)%ff_nodeNN(ii+1,itype)
         enddo
      enddo
      Wij_nn=nn_ff(ff_idx)%ff_Wij_nn
      B_nn=nn_ff(ff_idx)%ff_B_nn
      nodeMM=nn_ff(ff_idx)%ff_nodeMM
      ! ****************** read Wij ends************************

      ! ****************** read scaler starts************************
      a_scaler=nn_ff(ff_idx)%ff_a_scaler
      b_scaler=nn_ff(ff_idx)%ff_b_scaler
      ! ****************** read scaler ends************************

      !********************add_force****************
      inquire(file='add_force',exist=alive)
      if (alive) then
         open(10,file="add_force")
         rewind(10)
         read(10,*) add_force_num, alpha,y1,z1
         allocate(add_force_atom(add_force_num))
         ! allocate(direction(add_force_num))
         allocate(const_fa(add_force_num))
         allocate(const_fb(add_force_num))
         allocate(const_fc(add_force_num))
         do i=1,add_force_num
            read(10,*) add_force_atom(i), const_fa(i), const_fb(i),const_fc(i)
         enddo
         close(10)
      else
         add_force_num=0
      endif

      !******************force_constraint****************
      inquire(file='force_constraint',exist=alive)
      !  tatus = access ("add_force",' ')    ! blank mode
      !  if (status .eq. 0 ) then
      if (alive) then
         open(10,file="force_constraint")
         rewind(10)
         read(10,*) const_force_num
         allocate(const_force_atom(const_force_num))
         ! allocate(direction(add_force_num))
         allocate(const_fx(const_force_num))
         allocate(const_fy(const_force_num))
         allocate(const_fz(const_force_num))
         do i=1,const_force_num
            read(10,*) const_force_atom(i), const_fx(i), const_fy(i), const_fz(i)
         enddo
         close(10)
      else
         const_force_num=0
      endif

   end subroutine load_model_NN

   subroutine set_image_info_NN(atype_check)
      logical, intent(in) :: atype_check
      integer :: iitype

      if (atype_check) then
         do i=1,natoms
            iitype=0
            do itype=1,ntypes
               if(itype_atom(itype)==atype(i)) then
                  iitype=itype
               endif
            enddo
            if(iitype==0) then
               write(6,*) 'this type not found',atype(i)
            endif
         enddo
      endif

   end subroutine set_image_info_NN

   subroutine cal_energy_force_NN(feat,dfeat,num_neigh,list_neigh,dR_neigh,e_atom,Etot,fatom,virial,natom_tmp,nfeat0_tmp,m_neigh_tmp)
      integer :: natom_tmp,nfeat0_tmp,m_neigh_tmp
      real(8),intent(in) :: feat(nfeat0_tmp,natom_tmp)
      real(8), intent(in) :: dfeat(nfeat0_tmp,natom_tmp,m_neigh_tmp,3)
      integer,intent(in) :: num_neigh(natom_tmp)
      integer,intent(in) :: list_neigh(m_neigh_tmp,natom_tmp)
      real(8), dimension(3,m_neigh_tmp,natom_tmp), intent(in) :: dR_neigh
      real(8), dimension(natoms), intent(out) :: e_atom
      real(8), intent(out) :: Etot
      real(8), dimension(3, nall), intent(out) :: fatom
      real(8), dimension(6), intent(out) :: virial

      integer :: j,jj
      real(8) :: direct,mean

      real(8),allocatable,dimension(:,:,:) :: feat_type
      real(8),allocatable,dimension(:,:,:) :: f_in,f_out,f_d,f_back
      real(8),allocatable,dimension(:,:) :: energy_type
      real(8),allocatable,dimension(:,:,:) :: dEdf_type

      real(8) :: pi,dE,dFx,dFy,dFz
      real(8) :: rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd,d,w22_1,w22_2,w22F_1,w22F_2
      integer :: iat1,iat2,ierr
      integer :: ii
      real(8) :: x
      real(8) :: sum1,sum2,sum3


      pi=4*datan(1.d0)

      allocate(feat_type(nfeat1m,natom_tmp,ntypes))
      allocate(f_in(nodeMM,natom_tmp,nlayer+1))
      allocate(f_out(nodeMM,natom_tmp,nlayer+1))
      allocate(f_d(nodeMM,natom_tmp,nlayer+1))
      allocate(f_back(nodeMM,natom_tmp,nlayer+1))
      allocate(energy_type(natom_tmp,ntypes))
      allocate(dEdf_type(nfeat1m,natom_tmp,ntypes))
      !allocate(dfeat_type(nfeat1m,natoms*m_neigh*3,ntypes))


      !   allocate(energy_pred_NN(natoms))
      allocate(energy_pred_tmp(natoms))
      !   allocate(force_pred_NN(3,natoms))
      !   allocate(force_pred_tmp(3,natoms))


      if (nfeat0_tmp/=nfeat1m .or. natom_tmp/=natoms .or. m_neigh_tmp/=m_neigh) then
         write(*,*) "Shape of input arrays don't match the model!"
         write(6,*) nfeat0_tmp,natom_tmp,m_neigh_tmp
         write(6,*) nfeat1m,natoms,m_neigh
         stop
      endif

      num = 0
      iat1=0

      !write(*,*) "***********dbg: feat**************"
      !write(*,*) feat(:,1)

      do i = 1, natoms
         ! input layer initialization.
         ! For SINGLE THREAD ONLY
         iat1=iat1+1
         itype = catype(i) ! center atom type:1,2,3...ntypes: classify atom with atom type
         num(itype) = num(itype) + 1
         do j=1,nfeat1(itype)
            feat_type(j,num(itype),itype) = feat(j, iat1)*a_scaler(j,itype)+b_scaler(j,itype)
         enddo
      enddo

      ! **************************************************
      !             forward and backward
      ! **************************************************
      do itype=1,ntypes
         ! for each type of atom, perform forward propagation
         do i=1,num(itype)
            do j=1,nodeNN(1,itype)
               f_in(j,i,1)=feat_type(j,i,itype)
            enddo
         enddo

         do ii=1,nlayer
            if(ii.ne.1) then
               do i=1,num(itype)
                  do j=1,nodeNN(ii,itype)
                     x=f_in(j,i,ii)

                     if(x.gt.-150.d0.and.x.lt.150.d0) then

                        f_out(j, i, ii) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))  ! tanh
                        f_d(j, i, ii) = 1.0d0 - f_out(j,i,ii)*f_out(j,i,ii)

                     elseif(x.le.-150.d0) then
                        f_out(j,i,ii)=0.d0
                        f_d(j,i,ii)=0.d0
                     elseif(x.ge.150.d0) then
                        f_out(j,i,ii)=x
                        f_d(j,i,ii)=1.d0
                     endif
                  enddo
               enddo

            elseif(ii.eq.1) then
               do i=1,num(itype)
                  do j=1,nodeNN(ii,itype)
                     f_out(j,i,ii)=f_in(j,i,ii)
                     f_d(j,i,ii)=1.d0
                  enddo
               enddo
            endif

            !W * x
            !write(*,*) "nodeNN(ii+1,itype),num(itype),nodeNN(ii,itype)", nodeNN(ii+1,itype),num(itype),nodeNN(ii,itype)
            call dgemm('T','N',nodeNN(ii+1,itype),num(itype),nodeNN(ii,itype),1.d0,&
               Wij_nn(1,1,ii,itype),nodeMM,&
               f_out(1,1,ii),nodeMM,0.d0,&
               f_in(1,1,ii+1),nodeMM)
            !   ** # row of C *    *# col of C*  *#row of A*

            do i=1,num(itype)
               do j=1,nodeNN(ii+1,itype)
                  f_in(j,i,ii+1)=f_in(j,i,ii+1)+B_nn(j,ii,itype)
               enddo
            enddo

         enddo
         do i=1,num(itype)
            energy_type(i,itype)=f_in(1,i,nlayer+1)
         enddo

         !  Now, back propagation for the derivative for energy, in respect to the f_in(j,i,1)
         do i=1,num(itype)
            do j=1,nodeNN(nlayer,itype)
               f_back(j,i,nlayer)=Wij_nn(j,1,nlayer,itype)*f_d(j,i,nlayer)
            enddo
         enddo

         do ii=nlayer,2,-1

            call dgemm('N', 'N', nodeNN(ii-1,itype),num(itype),nodeNN(ii,itype), 1.d0, &
               Wij_nn(1,1,ii-1,itype),nodeMM, &
               f_back(1,1,ii),nodeMM,0.d0, &
               f_back(1,1,ii-1),nodeMM)

            if(ii-1.ne.1) then
               do i=1,num(itype)
                  do j=1,nodeNN(ii-1,itype)
                     f_back(j,i,ii-1)=f_back(j,i,ii-1)*f_d(j,i,ii-1)
                  enddo
               enddo
            endif
         enddo

         do i=1,num(itype)
            do j=1,nfeat1(itype)
               dEdf_type(j,i,itype)=f_back(j,i,1)
            enddo
         enddo
      enddo

      energy_pred_tmp = 0.d0
      num = 0
      iat1 = 0

      ! **************************************************
      !           inference : collect energy
      ! **************************************************

      do i = 1, natoms
         iat1=iat1+1
         itype = catype(i)
         num(itype) = num(itype) + 1
         energy_pred_tmp(i)=energy_type(num(itype),itype)
      enddo

      Etot=0.d0
      e_atom=0.d0

      do i = 1, natoms
         e_atom(i) = energy_pred_tmp(i)
         Etot = Etot + energy_pred_tmp(i)
      enddo

      ! **************************************************
      !           inference : force
      ! **************************************************
      !   force_pred_tmp=0.d0
      fatom=0.d0
      virial=0.d0
      num = 0
      iat1=0

      do i = 1, natoms
         iat1=iat1+1
         itype = catype(i)
         num(itype) = num(itype) + 1
         do jj = 1, num_neigh(i)
            iat2=list_neigh(jj,i)
            sum1=0.d0
            sum2=0.d0
            sum3=0.d0
            do j=1,nfeat1(itype)
               sum1=sum1+dEdf_type(j,num(itype),itype)*dfeat(j,iat1,jj,1)*a_scaler(j,itype)
               sum2=sum2+dEdf_type(j,num(itype),itype)*dfeat(j,iat1,jj,2)*a_scaler(j,itype)
               sum3=sum3+dEdf_type(j,num(itype),itype)*dfeat(j,iat1,jj,3)*a_scaler(j,itype)

            enddo
            !P=P+sum1*(R(1,iat2)-R(1,i))  (dR(1,j))
            fatom(1,iat2)=fatom(1,iat2)+sum1
            fatom(2,iat2)=fatom(2,iat2)+sum2
            fatom(3,iat2)=fatom(3,iat2)+sum3
            
            virial(1) = virial(1) + dR_neigh(1,jj,iat1)*sum1
            virial(2) = virial(2) + dR_neigh(2,jj,iat1)*sum2
            virial(3) = virial(3) + dR_neigh(3,jj,iat1)*sum3
            virial(4) = virial(4) + dR_neigh(1,jj,iat1)*sum2
            virial(5) = virial(5) + dR_neigh(1,jj,iat1)*sum3
            virial(6) = virial(6) + dR_neigh(2,jj,iat1)*sum3
         enddo
      enddo
      ! **************************************************
      !                finalize output
      ! **************************************************

      iat1=0
      ! **************************************************
      !              vdw: not in use yet
      ! **************************************************

      !   if (1.eq.0) then
      !      do i=1,natoms
      !            iat1=iat1+1
      !            rad1=rad_atom(catype(i))

      !            dE=0.d0
      !            dFx=0.d0
      !            dFy=0.d0
      !            dFz=0.d0

      !            do jj=1,num_neigh(i)
      !               j=list_neigh(jj,i)

      !               if(i.ne.j) then
      !                  rad2=rad_atom(catype(j))
      !                  rad=rad1+rad2

      !                  dx1=mod(xatom(1,j)-xatom(1,i)+100.d0,1.d0)
      !                  if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
      !                  dx2=mod(xatom(2,j)-xatom(2,i)+100.d0,1.d0)
      !                  if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
      !                  dx3=mod(xatom(3,j)-xatom(3,i)+100.d0,1.d0)
      !                  if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1

      !                  dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
      !                  dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
      !                  dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
      !                  dd=dsqrt(dx**2+dy**2+dz**2)

      !                  if(dd.lt.2*rad) then

      !                     w22_1=wp_atom(catype(j),catype(i),1)
      !                     w22_2=wp_atom(catype(j),catype(i),2)
      !                     w22F_1=(wp_atom(catype(j),catype(i),1)+wp_atom(catype(i),catype(j),1))/2     ! take the average for force calc.
      !                     w22F_2=(wp_atom(catype(j),catype(i),2)+wp_atom(catype(i),catype(j),2))/2     ! take the average for force calc.

      !                     yy=pi*dd/(4*rad)

      !                     dE=dE+0.5*4*(w22_1*(rad/dd)**12*cos(yy)**2+w22_2*(rad/dd)**6*cos(yy)**2)
      !                     dEdd=4*(w22F_1*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)   &
      !                        +W22F_2*(-6*(rad/dd)**6/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**6))

      !                     dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
      !                     dFy=dFy-dEdd*dy/dd
      !                     dFz=dFz-dEdd*dz/dd
      !                  endif
      !               endif
      !            enddo

      !            energy_pred_tmp(i)=energy_pred_tmp(i)+dE
      !            force_pred_tmp(1,i)=force_pred_tmp(1,i)+dFx   ! Note, assume force=dE/dx, no minus sign
      !            force_pred_tmp(2,i)=force_pred_tmp(2,i)+dFy
      !            force_pred_tmp(3,i)=force_pred_tmp(3,i)+dFz

      !      enddo
      !   endif

      !   ! **************************************************
      !   !    add & const force from input: not in use yet
      !   ! **************************************************

      !   if (1.eq.0) then
      !      mean=0.0
      !      do j=1,add_force_num

      !         if ((xatom(3,add_force_atom(j))-0.5).gt.0.0) direct = 1.0
      !         if ((xatom(3,add_force_atom(j))-0.5).lt.0.0) direct = - 1.0
      !         if (abs(xatom(3,add_force_atom(j))-0.5).lt.1.0E-5) direct=0.0

      !         const_fa(j)=0.0
      !         const_fb(j)= - alpha*direct*(xatom(3,add_force_atom(j))-z1)*AL(3,3)
      !         const_fc(j)=   alpha*direct*(xatom(2,add_force_atom(j))-y1)*AL(2,2)
      !         mean=mean+ const_fb(j)
      !      enddo

      !      do j=1,add_force_num

      !         force_pred_NN(1,add_force_atom(j))= force_pred_NN(1,add_force_atom(j))+const_fa(j)   !give a force on x axis
      !         force_pred_NN(2,add_force_atom(j))= force_pred_NN(2,add_force_atom(j))+const_fb(j)- mean/add_force_num ! wtf?
      !         force_pred_NN(3,add_force_atom(j))= force_pred_NN(3,add_force_atom(j))+const_fc(j)

      !      enddo

      !      do j=1,const_force_num
      !         force_pred_NN(1,const_force_atom(j))= const_fx(j)   !give a force on x axis
      !         force_pred_NN(2,const_force_atom(j))= const_fy(j)
      !         force_pred_NN(3,const_force_atom(j))= const_fz(j)

      !      enddo
      !   endif

      ! **************************************************
      !                finalize output
      ! **************************************************

      !   etot_pred_NN = 0.d0
      !   do i = 1, natoms
      !      !write(*,*) energy_pred_NN(i)
      !      etot_pred_NN = etot_pred_NN + energy_pred_tmp(i)
      !   enddo
      deallocate(feat_type)
      deallocate(energy_type)
      deallocate(dEdf_type)
      deallocate(f_in)
      deallocate(f_out)
      deallocate(f_d)
      deallocate(f_back)
      deallocate(energy_pred_tmp)

   end subroutine cal_energy_force_NN


end module calc_NN


