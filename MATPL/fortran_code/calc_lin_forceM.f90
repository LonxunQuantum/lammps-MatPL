module calc_lin

   use mod_data, only : natoms, nall, ntypes, catype, atype
   use li_ff_mod, only : li_ff ! force field data
   !implicit double precision (a-h, o-z)
   implicit none

   !!!!!!!!!!!!!          以下为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   integer :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
   integer :: nfeat1m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
   integer :: nfeat2m                                  !不同种原子的PCA之后feature数目中最大者
   integer :: nfeat2tot                                !PCA之后各种原子的feature数目之和
   integer,allocatable,dimension(:) :: nfeat1          !各种原子的原始feature数目
   integer,allocatable,dimension(:) :: nfeat2          !各种原子PCA之后的feature数目
   integer,allocatable,dimension(:) :: nfeat2i         !用来区分计算时各段各属于哪种原子的分段端点序号
   integer :: iatom_tmp(100)                             !每一种原子的原子属于第几种原子

   real(8),allocatable,dimension(:) :: bb                 !计算erergy和force时与new feature相乘的系数向量w
   real(8),allocatable,dimension(:,:) :: bb_type0         !将bb分别归类到不同种类的原子中，第二维才是代表原子种类

   real(8),allocatable,dimension(:,:,:) :: pv             !PCA所用的转换矩阵
   real(8),allocatable,dimension(:,:) :: feat2_shift     !PCA之后用于标准化feat2的平移矩阵
   real(8),allocatable,dimension(:,:) :: feat2_scale     !PCA之后用于标准化feat2的伸缩系数矩阵

   integer,allocatable,dimension(:) :: num             !属于每种原子的原子个数，但似乎在calc_linear中无用

   ! real(8),allocatable,dimension(:) :: energy_pred_lin       !每个原子的能量预测值
   real(8),allocatable,dimension(:) :: energy_pred_tmp        !每个原子的能量预测值
   ! real(8),allocatable,dimension(:,:) :: force_pred_lin       !每个原子的受力预测值
   ! real(8),allocatable,dimension(:,:) :: force_pred_tmp       !每个原子的受力预测值
   ! real(8) :: etot_pred_lin

   real(8),allocatable,dimension(:) :: rad_atom,E_ave_vdw
   real(8),allocatable,dimension(:,:,:) :: wp_atom
   integer :: nfeat_type_l
   integer :: ifeat_type_l(10)

   !!!!!!!!!!!!!          以上为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! loop index
   integer :: i,itype

contains

   subroutine load_model_lin(ff_idx)

      integer, intent(in) :: ff_idx
      integer :: k,j,kkk
      integer :: ntmp,itmp,nterm

      ! ************************************
      !         fetch ff_mod data
      ! ************************************
      if (allocated(nfeat1)) deallocate(nfeat1)
      if (allocated(nfeat2)) deallocate(nfeat2)
      if (allocated(nfeat2i)) deallocate(nfeat2i)
      if (allocated(rad_atom)) deallocate(rad_atom)
      if (allocated(E_ave_vdw)) deallocate(E_ave_vdw)
      if (allocated(wp_atom)) deallocate(wp_atom)
      if (allocated(bb)) deallocate(bb)
      if (allocated(bb_type0)) deallocate(bb_type0)
      if (allocated(pv)) deallocate(pv)
      if (allocated(feat2_shift)) deallocate(feat2_shift)
      if (allocated(feat2_scale)) deallocate(feat2_scale)
      if (allocated(num)) deallocate(num)

      allocate(num(ntypes))
      allocate(nfeat1(ntypes))
      allocate(nfeat2(ntypes))
      allocate(nfeat2i(ntypes))
      allocate(rad_atom(ntypes))
      allocate(E_ave_vdw(ntypes))
      allocate(wp_atom(ntypes,ntypes,2))
      ! **************** read vdw_fitB.ntype *************
      nterm=li_ff(ff_idx)%nterm
      do itype=1,ntypes
         rad_atom(itype)=li_ff(ff_idx)%rad_atom(itype)
         E_ave_vdw(itype)=li_ff(ff_idx)%E_ave_vdw(itype)
         do j=1,nterm
            do i=1,ntypes
               wp_atom(i,itype,j)=li_ff(ff_idx)%wp_atom(i,itype,j)
            enddo
         enddo
      enddo
      ! **************** read feat.info ********************
      m_neigh=li_ff(ff_idx)%ff_max_neigh
      nfeat_type_l=li_ff(ff_idx)%ff_nfeat_type
      do kkk=1,nfeat_type_l
         ifeat_type_l(kkk)=li_ff(ff_idx)%ff_ifeat_type(kkk)   ! the index (1,2,3) of the feature type
      enddo
      ! ntype=li_ff(ff_idx)%ff_num_type
      do i=1,ntypes
         iatom_tmp(i)=li_ff(ff_idx)%ff_itype_atom_sumfe(i,1)
         nfeat1(i)=li_ff(ff_idx)%ff_itype_atom_sumfe(i,2) ! these nfeat1,nfeat2 include all ftype
         nfeat2(i)=li_ff(ff_idx)%ff_itype_atom_sumfe(i,3)
      enddo

      ! cccccccc Right now, nfeat1,nfeat2,for different types
      ! cccccccc must be the same. We will change that later, allow them
      ! cccccccc to be different
      nfeat1m=0   ! the original feature
      nfeat2m=0   ! the new PCA, PV feature
      nfeat2tot=0 ! tht total feature of diff atom type
      nfeat2i=0   ! the starting point
      nfeat2i(1)=0
      do i=1,ntypes
         if(nfeat1(i).gt.nfeat1m) nfeat1m=nfeat1(i)
         if(nfeat2(i).gt.nfeat2m) nfeat2m=nfeat2(i)
         nfeat2tot=nfeat2tot+nfeat2(i)
         if(i.gt.1) then
            nfeat2i(i)=nfeat2i(i-1)+nfeat2(i-1)
         endif

      enddo

      allocate(bb(nfeat2tot))
      allocate(bb_type0(nfeat2m,ntypes))

      ntmp=li_ff(ff_idx)%nfeat2tot
      if (ntmp/=nfeat2tot) then
         write (6, *) 'ntmp.not.right,linear_fitb.ntypes', ntmp, nfeat2tot
         stop
      endif
      do i = 1, nfeat2tot
         itmp=li_ff(ff_idx)%ifeat2tot
         bb(i) = li_ff(ff_idx)%BB(i)
      enddo
      do itype = 1, ntypes
         do k = 1, nfeat2(itype)
            bb_type0(k, itype) = bb(k+nfeat2i(itype))
         enddo
      enddo

      allocate(pv(nfeat1m,nfeat2m,ntypes))
      allocate(feat2_shift(nfeat2m,ntypes))
      allocate(feat2_scale(nfeat2m,ntypes))

      !TODO: without PCA : need to check, just test
      do itype=1,ntypes
         do k=1,nfeat2m
            do j=1, nfeat1m
               if(j.eq.k) then
                  pv(j,k,itype) = 1.d0
               else
                  pv(j,k,itype) = 0.d0
               endif
            enddo
         enddo
      enddo

      feat2_shift=li_ff(ff_idx)%ff_feat2_shift
      feat2_scale=li_ff(ff_idx)%ff_feat2_scale

   end subroutine load_model_lin

   subroutine set_image_info_lin(atype_check)
      logical, intent(in) :: atype_check
      integer :: iitype

      if (atype_check) then
         do i=1,natoms
            iitype=0
            do itype=1,ntypes
               if(iatom_tmp(itype)==atype(i)) then
                  iitype=itype
               endif
            enddo
            if(iitype==0) then
               write(6,*) 'this type not found',atype(i)
            endif
         enddo
      endif

   end subroutine set_image_info_lin

   subroutine cal_energy_force_lin(feat,dfeat,num_neigh,list_neigh,dR_neigh,e_atom,Etot,fatom,virial,natom_tmp,nfeat0_tmp,m_neigh_tmp)

      integer :: natom_tmp,nfeat0_tmp,m_neigh_tmp

      real(8),intent(in) :: feat(nfeat0_tmp,natoms)
      real(8), intent(in) :: dfeat(nfeat0_tmp,natoms,m_neigh_tmp,3)
      integer, dimension(natom_tmp), intent(in) :: num_neigh
      ! integer, dimension(ntypes,natoms), intent(in) :: num_neigh
      ! integer, dimension(m_neigh,ntypes,natoms), intent(in) :: list_neigh
      integer, dimension(m_neigh_tmp,natom_tmp), intent(in) :: list_neigh
      real(8), dimension(3,m_neigh_tmp,natom_tmp), intent(in) :: dR_neigh
      real(8), dimension(natoms), intent(out) :: e_atom
      real(8), intent(out) :: Etot
      real(8), dimension(3, nall), intent(out) :: fatom
      real(8), dimension(6), intent(out) :: virial

      integer :: j,jj
      real(8) :: sum
      ! real(8), intent(in) :: AL(3,3)
      ! real(8),dimension(:,:),intent(in) :: xatom

      real(8),allocatable,dimension(:,:) :: feat2
      real(8),allocatable,dimension(:,:,:) :: feat_type
      real(8),allocatable,dimension(:,:,:) :: feat2_type
      integer,allocatable,dimension(:,:) :: ind_type
      real(8),allocatable,dimension(:,:,:) :: dfeat_type
      real(8),allocatable,dimension(:,:,:) :: dfeat2_type
      real(8),allocatable,dimension(:,:,:,:) :: dfeat2
      real(8),allocatable,dimension(:,:,:) :: dE_dx

      real(8) :: pi
      ! real(8) dE,dFx,dFy,dFz
      ! real(8) rad1,rad2,rad,dx1,dx2,dx3,dx,dy,dz,dd,yy,w22,dEdd,d,w22_1,w22_2,w22F_1,w22F_2
      integer :: iat,iat1,iat2,itype1,itype2

      pi=4*datan(1.d0)

      allocate(feat2(nfeat2m,natoms))
      allocate(feat_type(nfeat1m,natoms,ntypes))
      allocate(feat2_type(nfeat2m,natoms,ntypes))
      allocate(ind_type(natoms,ntypes))
      allocate(dfeat_type(nfeat1m,natoms*m_neigh*3,ntypes))
      allocate(dfeat2_type(nfeat2m,natoms*m_neigh*3,ntypes))
      allocate(dfeat2(nfeat2m,natoms,m_neigh,3))
      allocate(dE_dx(3,m_neigh,natoms))

      ! allocate(energy_pred_lin(natoms))
      allocate(energy_pred_tmp(natoms))
      ! allocate(force_pred_lin(3,natoms))
      ! allocate(force_pred_tmp(3,natoms))

      if (nfeat0_tmp/=nfeat1m .or. natom_tmp/=natoms .or. m_neigh_tmp/=m_neigh) then
         write(*,*) "Shape of input arrays don't match the model!"
         stop
      endif

      !cccccccccccccccccccccccccccccccccccccccccccccccc
      ! max atom number of one type
      ! each type atom number
      num=0
      iat1=0
      do i = 1,natoms
         iat1=iat1+1
         itype=catype(i)    ! center atom type:1,2,3...ntypes: classify atom with atom type
         ! itype=iatom_type(i)  ! center atom type:1,2,3...ntypes
         num(itype)=num(itype)+1 ! num(itype) is the number of each atom type itype
         ind_type(num(itype),itype)=iat1
         feat_type(:, num(itype), itype)=feat(:, iat1)
      enddo
      ! write(*,*) "catype",catype
      ! write(*,*) "num",num
      ! write(*,*) "ind_type",ind_type
      ! write(*,*) "feat_type",feat_type

      do itype = 1, ntypes
         call dgemm('T', 'N', nfeat2(itype), num(itype), nfeat1(itype), 1.d0, &
            pv(1,1,itype), nfeat1m, &
            feat_type(1,1,itype), nfeat1m, 0.d0, &
            feat2_type(1,1,itype), nfeat2m)
      enddo

      do itype = 1, ntypes
         do i = 1, num(itype)
            do j = 1, nfeat2(itype) - 1
               ! write(*,*) "feat2_shift(j,itype)",feat2_shift(j,itype)
               ! write(*,*) "feat2_scale(j, itype)",feat2_scale(j, itype)
               feat2_type(j, i, itype) = (feat2_type(j,i,itype)-feat2_shift(j,itype))*feat2_scale(j, itype)
            enddo
            feat2_type(nfeat2(itype), i, itype) = 1.d0
         enddo
      enddo

      ! write(*,*) "feat2_type222222222",feat2_type

      num = 0
      iat1=0
      do i = 1, natoms
         iat1=iat1+1
         itype=catype(i)
         num(itype) = num(itype) + 1
         feat2(:, iat1) = feat2_type(:, num(itype), itype)
      enddo

      ! write(*,*) "feat200000000000",feat2

      energy_pred_tmp=0.d0

      iat1=0
      do i = 1, natoms
         iat1=iat1+1
         itype = catype(i)
         sum = 0.d0
         do j = 1, nfeat2(itype)
            sum = sum + feat2(j, iat1)*bb_type0(j, itype)
         enddo
         energy_pred_tmp(i) = sum
      enddo

      Etot=0.d0
      e_atom=0.d0

      do i = 1, natoms
         e_atom(i) = energy_pred_tmp(i)
         Etot = Etot + energy_pred_tmp(i)
      enddo

      num = 0
      iat1=0
      do i = 1, natoms
         iat1=iat1+1
         itype = catype(i)
         do jj = 1, num_neigh(i)
            ! do jj = 1, num_neigh(itype,i)
            num(itype) = num(itype) + 1
            dfeat_type(:, num(itype), itype) = dfeat(:, iat1, jj, 1)
            num(itype) = num(itype) + 1
            dfeat_type(:, num(itype), itype) = dfeat(:, iat1, jj, 2)
            num(itype) = num(itype) + 1
            dfeat_type(:, num(itype), itype) = dfeat(:, iat1, jj, 3)
         enddo
      enddo
      !cccccccc note: num(itype) is rather large, in the scane of natoms*num_neigh

      do itype = 1, ntypes
         call dgemm('T', 'N', nfeat2(itype), num(itype), nfeat1(itype), 1.d0, &
            pv(1,1,itype), nfeat1m, &
            dfeat_type(1,1,itype), nfeat1m, 0.d0, &
            dfeat2_type(1,1,itype), nfeat2m)
      enddo

      num = 0
      iat1=0
      do i = 1, natoms
         iat1=iat1+1
         itype = catype(i)
         do jj = 1, num_neigh(i)
            ! do jj = 1, num_neigh(itype,i)
            ! itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
            num(itype) = num(itype) + 1
            do j = 1, nfeat2(itype) - 1
               dfeat2(j, iat1, jj, 1) = dfeat2_type(j, num(itype), itype)*feat2_scale(j, itype)
            enddo
            dfeat2(nfeat2(itype), iat1, jj, 1) = 0.d0
            num(itype) = num(itype) + 1
            do j = 1, nfeat2(itype) - 1
               dfeat2(j, iat1, jj, 2) = dfeat2_type(j, num(itype), itype)*feat2_scale(j, itype)
            enddo
            dfeat2(nfeat2(itype), iat1, jj, 2) = 0.d0
            num(itype) = num(itype) + 1
            do j = 1, nfeat2(itype) - 1
               dfeat2(j, iat1, jj, 3) = dfeat2_type(j, num(itype), itype)*feat2_scale(j, itype)
            enddo
            dfeat2(nfeat2(itype), iat1, jj, 3) = 0.d0
         enddo
      enddo

      !cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      !cc  the new dfeat2 is:
      !cc dfeat2(nfeat2,natoms,j_neigh,3): dfeat2(j,i,jj,3)= d/dr(jj_neigh)(feat2(j,i))
      !cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      !cccc now, we have the new features, we need to calculate the distance to reference state

      fatom=0.d0
      ! force_pred_tmp=0.d0
      virial=0.d0

      iat1=0
      do i = 1, natoms
         iat1=iat1+1
         itype = catype(i)
         do jj = 1, num_neigh(i)
            ! do jj = 1, num_neigh(itype,i)
            ! iat2=list_neigh(jj,itype,i)
            iat2=list_neigh(jj,i)
            dE_dx=0.d0
            do j = 1, nfeat2(itype)
               fatom(1,iat2)=fatom(1,iat2)+dfeat2(j,iat1,jj,1)*bb_type0(j,itype)
               fatom(2,iat2)=fatom(2,iat2)+dfeat2(j,iat1,jj,2)*bb_type0(j,itype)
               fatom(3,iat2)=fatom(3,iat2)+dfeat2(j,iat1,jj,3)*bb_type0(j,itype)

               dE_dx(1,jj,iat1)=dE_dx(1,jj,iat1)+dfeat2(j,iat1,jj,1)*bb_type0(j,itype)
               dE_dx(2,jj,iat1)=dE_dx(2,jj,iat1)+dfeat2(j,iat1,jj,2)*bb_type0(j,itype)
               dE_dx(3,jj,iat1)=dE_dx(3,jj,iat1)+dfeat2(j,iat1,jj,3)*bb_type0(j,itype)
            enddo

            virial(1) = virial(1) + dR_neigh(1,jj,iat1)*dE_dx(1,jj,iat1)
            virial(2) = virial(2) + dR_neigh(2,jj,iat1)*dE_dx(2,jj,iat1)
            virial(3) = virial(3) + dR_neigh(3,jj,iat1)*dE_dx(3,jj,iat1)
            virial(4) = virial(4) + dR_neigh(1,jj,iat1)*dE_dx(2,jj,iat1)
            virial(5) = virial(5) + dR_neigh(1,jj,iat1)*dE_dx(3,jj,iat1)
            virial(6) = virial(6) + dR_neigh(2,jj,iat1)*dE_dx(3,jj,iat1)
         enddo
      enddo

      !ccccccccccccccccccccccccccccccccccccccccccc
      !       iat1=0
      !       do i=1,natoms
      !          iat1=iat1+1
      !          rad1=rad_atom(catype(i))   ! rad_atom = 0
      !          dE=0.d0
      !          dFx=0.d0
      !          dFy=0.d0
      !          dFz=0.d0
      !          do jj=1,num_neigh(i)
      !             j=list_neigh(jj,i)
      !             if(i.ne.j) then
      !                rad2=rad_atom(catype(j))
      !                rad=rad1+rad2
      !                dx1=mod(xatom(1,j)-xatom(1,i)+100.d0,1.d0)
      !                if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
      !                dx2=mod(xatom(2,j)-xatom(2,i)+100.d0,1.d0)
      !                if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
      !                dx3=mod(xatom(3,j)-xatom(3,i)+100.d0,1.d0)
      !                if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1
      !                dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
      !                dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
      !                dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
      !                dd=dsqrt(dx**2+dy**2+dz**2)
      !                if(dd.lt.2*rad) then
      ! !        w22=dsqrt(wp_atom(catype(i))*wp_atom(catype(j)))
      ! !        yy=pi*dd/(4*rad)
      ! ! !       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
      ! ! !       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
      ! ! !     &   -(pi/(2*rad))*cos(yy)*sin(yy))
      ! !        dE=dE+0.5*4*w22*(rad/dd)**12*cos(yy)**2
      ! !        dEdd=4*w22*(-12*(rad/dd)**12/dd*cos(yy)**2  &
      ! !         -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)
      !                   w22_1=wp_atom(catype(j),catype(i),1)
      !                   w22_2=wp_atom(catype(j),catype(i),2)
      !                   w22F_1=(wp_atom(catype(j),catype(i),1)+wp_atom(catype(i),catype(j),1))/2     ! take the average for force calc.
      !                   w22F_2=(wp_atom(catype(j),catype(i),2)+wp_atom(catype(i),catype(j),2))/2     ! take the average for force calc.

      !                   yy=pi*dd/(4*rad)
      ! ! c       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
      ! ! c       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
      ! ! c     &   -(pi/(2*rad))*cos(yy)*sin(yy))
      !                   dE=dE+0.5*4*(w22_1*(rad/dd)**12*cos(yy)**2+w22_2*(rad/dd)**6*cos(yy)**2)
      !                   dEdd=4*(w22F_1*(-12*(rad/dd)**12/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)   &
      !                      +W22F_2*(-6*(rad/dd)**6/dd*cos(yy)**2-(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**6))


      !                   dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
      !                   dFy=dFy-dEdd*dy/dd
      !                   dFz=dFz-dEdd*dz/dd

      !                endif
      !             endif
      !          enddo

      !          energy_pred_tmp(i)=energy_pred_tmp(i)+dE
      !          force_pred_tmp(1,i)=force_pred_tmp(1,i)+dFx   ! Note, assume force=dE/dx, no minus sign
      !          force_pred_tmp(2,i)=force_pred_tmp(2,i)+dFy
      !          force_pred_tmp(3,i)=force_pred_tmp(3,i)+dFz
      !       enddo
!ccccccccccccccccccccccccccccccccccccccccccc

      ! etot_pred_lin = 0.d0
      ! do i = 1, natoms
      !    !etot = etot + energy(i)
      !    etot_pred_lin = etot_pred_lin + energy_pred_lin(i)
      ! enddo
      deallocate(feat2)
      deallocate(feat_type)
      deallocate(feat2_type)
      deallocate(ind_type)
      deallocate(dfeat_type)
      deallocate(dfeat2_type)
      deallocate(dfeat2)
      deallocate(dE_dx)
      ! deallocate(nfeat1)
      ! deallocate(nfeat2)
      ! deallocate(nfeat2i)
      ! deallocate(rad_atom)
      ! deallocate(E_ave_vdw)
      ! deallocate(wp_atom)
      ! deallocate(bb)
      ! deallocate(bb_type0)
      ! deallocate(pv)
      ! deallocate(feat2_shift)
      ! deallocate(feat2_scale)
      ! deallocate(num)
      ! deallocate(energy_pred_lin)
      deallocate(energy_pred_tmp)
      ! deallocate(force_pred_lin)
      ! deallocate(force_pred_tmp)
   end subroutine cal_energy_force_lin

end module calc_lin


