module calc_deepMD_f
    ! This is a strange version, with the ghost neighbore atoms, this is due to 
    ! original DP bug, and we keep the bug here. 
    use dp_ff_mod, only: ff 

    use mod_data, only : natoms, ntypes, catype, atype 

    IMPLICIT NONE

    !ccccccccccccccccccccc  The variables to be used in global feature type
    integer :: m_neigh,m_neigh_t5
    
    integer :: dp_M2
    real*8 Rc_type(100), R_cs(100)
    real*8 ave_shift(4,100),ave_norm(4,100)
    real*8,allocatable,dimension (:,:,:) :: s_neigh
    real*8,allocatable,dimension (:,:,:,:) :: ds_neigh
    real*8,allocatable,dimension (:,:,:,:) :: dxyz_neigh
    real*8,allocatable,dimension (:,:,:,:,:) :: dxyz_dx_neigh
    
    contains

    subroutine load_model_deepMD_f(ff_idx)
        integer, intent(in) :: ff_idx        
        integer :: i,itype
        integer :: dp_atype(100)  !每一种原子的原子属于第几种原子
        integer :: iitype_count  ! Counter for matching itypes
        integer :: sliced_itype1
        integer, allocatable :: iitype_list(:)
        logical, allocatable :: itype_added(:)

        dp_M2 = ff(ff_idx)%dp_ff_M2

        ! do i=1,ff(ff_idx)%dp_ff_num_type
        !   Rc_type(i) = ff(ff_idx)%dp_ff_Rc_type(i)
        !   R_cs(i) = ff(ff_idx)%dp_ff_R_cs(i)
        !   ave_shift(:,i) = ff(ff_idx)%dp_ff_ave_shift(:,i) 
        !   ave_norm(:,i) = ff(ff_idx)%dp_ff_ave_norm(:,i) 
        ! enddo
        ! get atom type
        do i=1,ff(ff_idx)%dp_ff_num_type
          dp_atype(i) = ff(ff_idx)%dp_ff_itype_atom(i)
        enddo
        ! ***************** atom_type to itype_list *************
      allocate(iitype_list(ff(ff_idx)%dp_ff_num_type))
      allocate(itype_added(ff(ff_idx)%dp_ff_num_type))

      iitype_count = 0
      itype_added = .false.
      do i = 1, natoms
         do itype = 1, ff(ff_idx)%dp_ff_num_type
            if (.not. itype_added(itype) .and. dp_atype(itype)==atype(i)) then
               itype_added(itype) = .true.
               iitype_count = iitype_count + 1
               iitype_list(iitype_count) = itype
            end if
         end do
      enddo

      ! ***************** end atom_type to itype_list *********

        do i=1,iitype_count
          sliced_itype1 = iitype_list(i)
          Rc_type(i) = ff(ff_idx)%dp_ff_Rc_type(sliced_itype1)
          R_cs(i) = ff(ff_idx)%dp_ff_R_cs(sliced_itype1)
          ave_shift(:,i) = ff(ff_idx)%dp_ff_ave_shift(:,sliced_itype1) 
          ave_norm(:,i) = ff(ff_idx)%dp_ff_ave_norm(:,sliced_itype1) 
        enddo

        ! deallocate(iitype_list)
        ! deallocate(itype_added)
    end subroutine load_model_deepMD_f    

    subroutine set_image_info_deepMD_f(ff_idx)
        integer, intent(in) :: ff_idx        
        
        m_neigh = ff(ff_idx)%dp_ff_max_neigh
        m_neigh_t5 = m_neigh
        allocate(s_neigh(m_neigh,ntypes,natoms))
        allocate(ds_neigh(3,m_neigh,ntypes,natoms))
        allocate(dxyz_neigh(4,m_neigh,ntypes,natoms))
        allocate(dxyz_dx_neigh(3,4,m_neigh,ntypes,natoms))
        ! These arrays should be deallocated at the end! 
 
    end subroutine set_image_info_deepMD_f

    subroutine gen_deepMD_feature(num_neigh,dR_neigh)

        integer, dimension(ntypes,natoms), intent(in) :: num_neigh
        real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh

        integer(4)  :: itype1,itype,i,j
        integer iat
        real*8 s,rr,r,pi,x,yy,dyy
        real*8 dr(3),ds(3)
        integer m,m2

        ! SINGLE THREAD find_neighbor 
        ! this step won't be necessary for lammps.  
        !call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
        !    dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
        !    num_neigh_M,iat_neigh_M,inode,nnodes)


        pi=4*datan(1.d0)
        dxyz_dx_neigh=0.d0

        ! *************************************************** 
        !           meaning of variables 
        !       dR_neigh: R_ij , distance between neighboring atoms 
        ! ***************************************************
        do iat=1,natoms
          itype1=catype(iat)  ! center atom type
          do itype=1,ntypes
            do j=1,num_neigh(itype,iat)
              rr=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
              r=dsqrt(rr)
              ! don't forgot, there is also a derivative with respective to the center i atom
              dr(1)=dR_neigh(1,j,itype,iat)/r
              dr(2)=dR_neigh(2,j,itype,iat)/r
              dr(3)=dR_neigh(3,j,itype,iat)/r

              if(r.lt.R_cs(itype1)) then
                s=1/r
                ds(:)=-dr(:)/r**2
              elseif(r.ge.R_cs(itype1).and.r.le.Rc_type(itype1)) then
                !cs=1/r*(cos(pi*(r-R_cs(itype))/(Rc_type(itype)-R_cs(itype)))+1)*0.5
                x=(r-R_cs(itype))/(Rc_type(itype1)-R_cs(itype1))
                yy=x**3*(-6*x**2+15*x-10)+1
                dyy=3*x**2*(-6*x**2+15*x-10)+x**3*(-12*x+15)
                s=1/r*yy
                ds(:)=-dr(:)/r**2*yy+1/r*dyy*dr(:)/(Rc_type(itype1)-R_cs(itype1))
              elseif(r.gt.Rc_type(itype1)) then
                s=0.d0
                ds=0.d0
              endif
              dxyz_neigh(1,j,itype,iat)=(s-ave_shift(1,itype1))/ave_norm(1,itype1)
              dxyz_neigh(2,j,itype,iat)=(dR_neigh(1,j,itype,iat)*s/r-ave_shift(2,itype1))/ave_norm(2,itype1)
              dxyz_neigh(3,j,itype,iat)=(dR_neigh(2,j,itype,iat)*s/r-ave_shift(3,itype1))/ave_norm(3,itype1)
              dxyz_neigh(4,j,itype,iat)=(dR_neigh(3,j,itype,iat)*s/r-ave_shift(4,itype1))/ave_norm(4,itype1)         

              s_neigh(j,itype,iat)=dxyz_neigh(1,j,itype,iat)

              if(j.eq.num_neigh(itype,iat).and.j.lt.m_neigh) then  ! this is to keep with the DP bug ?
                dxyz_neigh(1,j+1,itype,iat)=(0.d0-ave_shift(1,itype1))/ave_norm(1,itype1)                     
                dxyz_neigh(2,j+1,itype,iat)=(0.d0-ave_shift(2,itype1))/ave_norm(2,itype1)
                dxyz_neigh(3,j+1,itype,iat)=(0.d0-ave_shift(3,itype1))/ave_norm(3,itype1)
                dxyz_neigh(4,j+1,itype,iat)=(0.d0-ave_shift(4,itype1))/ave_norm(4,itype1)

                ! wlj altered ?
                s_neigh(j+1,itype,iat)=dxyz_neigh(1,j+1,itype,iat)
                !s_neigh(j+1:m_neigh,itype,iat)=dxyz_neigh(1,j+1,itype,iat)
              endif

              do m2=2,4
                dxyz_dx_neigh(m2-1,m2,j,itype,iat)=s/r/ave_norm(m2,itype1)
              enddo
                    
              do m=1,3
                dxyz_dx_neigh(m,1,j,itype,iat)=dxyz_dx_neigh(m,1,j,itype,iat)+ &
                                               ds(m)/ave_norm(1,itype1)
                ds_neigh(m,j,itype,iat)=dxyz_dx_neigh(m,1,j,itype,iat)
                do m2=2,4
                  dxyz_dx_neigh(m,m2,j,itype,iat)=dxyz_dx_neigh(m,m2,j,itype,iat)+ &
                    dR_neigh(m2-1,j,itype,iat)*(ds(m)/r-s/r**2*dr(m))/ave_norm(m2,itype1)
                enddo
              enddo
            enddo ! j end

              do j=num_neigh(itype,iat),m_neigh-1
                dxyz_neigh(1,j+1,itype,iat)=(0.d0-ave_shift(1,itype1))/ave_norm(1,itype1)                     
                dxyz_neigh(2,j+1,itype,iat)=(0.d0-ave_shift(2,itype1))/ave_norm(2,itype1)
                dxyz_neigh(3,j+1,itype,iat)=(0.d0-ave_shift(3,itype1))/ave_norm(3,itype1)
                dxyz_neigh(4,j+1,itype,iat)=(0.d0-ave_shift(4,itype1))/ave_norm(4,itype1)

                s_neigh(j+1,itype,iat)=dxyz_neigh(1,j+1,itype,iat)
              enddo ! j end
            
          enddo ! itype end
        enddo ! iat end

        ! interface to python Ri
        ! do iat=1,natoms
        !   itype1=catype(iat)  ! center atom type
        !   write(*,"(A, 1(I3, 1X))") "natom", iat
        !   do itype=1,ntypes
        !     do j=1,m_neigh
        !       write(*,"(4(F12.6, 2X))") dxyz_neigh(1,j,itype,iat), &
        !         dxyz_neigh(2,j,itype,iat), &
        !         dxyz_neigh(3,j,itype,iat), &
        !         dxyz_neigh(4,j,itype,iat)
        !     enddo
        !   enddo
        ! enddo

    end subroutine gen_deepMD_feature

end module calc_deepMD_f


