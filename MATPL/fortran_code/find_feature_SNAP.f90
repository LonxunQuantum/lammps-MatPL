subroutine find_feature_snap(Rc,nsnapw_type,snapj_type,wsnap_type,&
   num_neigh,list_neigh,dR_neigh,m_neigh,&
   nBB,nBBm,jjj123,CC_func,Clebsch_Gordan,jmm,&
   nfeat0m,feat_all,dfeat_all)

   use mod_data, only : natoms, ntypes, catype
   implicit none
   real(8), intent(in) :: Rc(ntypes)
   integer, intent(in) :: nsnapw_type(50)
   real(8), intent(in) :: snapj_type(50)
   real(8), intent(in) :: wsnap_type(50,10,50)
   integer, dimension(ntypes,natoms), intent(in) :: num_neigh
   real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh
   integer, intent(in) :: m_neigh
   integer, intent(in) :: list_neigh(m_neigh,ntypes,natoms)
   integer, intent(in) :: nBB(50)
   integer, intent(in) :: nBBm
   integer, intent(in) :: jjj123(3,nBBm,ntypes)
   integer, intent(in) :: jmm
   real(8), intent(in) :: CC_func(0:jmm,-jmm:jmm,-jmm:jmm,0:jmm)    ! jmm is the double index
   real(8), intent(in) :: Clebsch_Gordan(-jmm:jmm,-jmm:jmm,-jmm:jmm,0:jmm,0:jmm,0:jmm)
   integer, intent(in) :: nfeat0m
   real(8), intent(out) :: feat_all(nfeat0m,natoms)
   real(8), intent(out) :: dfeat_all(nfeat0m,natoms,m_neigh,3)

   real(8) :: dR_neigh_alltype(3,m_neigh,natoms)
   integer :: num_neigh_alltype(natoms)
   integer :: jm,itype0,j1,j2,jj,jj3,jj4

   integer :: i,j,k,kk,kkk,num,iat,iat1,itype
   real(8) :: d,dd,y
   real(8) :: dx(3)
   real(8) :: pi,pi2
   real(8) :: f2
   real(8) :: sum
   real(8) :: ww(10),dww_dx(10,3)
   integer :: m11,m12,m21,m22,m1,m2
   integer :: mm_neigh

   integer :: ind_all_neigh(m_neigh,ntypes,natoms),list_neigh_alltype(m_neigh,natoms)

   complex(8),allocatable,dimension (:,:,:,:) :: UJ
   complex(8),allocatable,dimension (:,:,:,:,:) :: dUJ
   real(8),allocatable,dimension (:) :: dsum


   num_neigh_alltype=0
   do iat=1,natoms
      num=1    ! including the original
      list_neigh_alltype(1,iat)=iat   ! the first neighbore is itself
      dR_neigh_alltype(:,1,iat)=0.d0
      do itype=1,ntypes
         do j=1,num_neigh(itype,iat)
            num=num+1
            if(num.gt.m_neigh) then
               write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
               stop
            endif
            ind_all_neigh(j,itype,iat)=num
            list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
            dR_neigh_alltype(:,num,iat)=dR_neigh(:,j,itype,iat)
         enddo
      enddo
      num_neigh_alltype(iat)=num
   enddo

   !ccccccccccccccccccccccccccccccccccccccccc
   feat_all=0.d0
   dfeat_all=0.d0

   pi=4*datan(1.d0)
   pi2=2*pi

   iat1=0
   do 3000 iat=1,natoms
      iat1=iat1+1
      itype0=catype(iat)
      jm=snapj_type(itype0)*2*1.001
      mm_neigh=num_neigh_alltype(iat)   ! include the origin at 1
      allocate(UJ(-jm:jm,-jm:jm,0:jm,nsnapw_type(itype0)))
      allocate(dUJ(3*mm_neigh,-jm:jm,-jm:jm,0:jm,nsnapw_type(itype0)))

      UJ=cmplx(0.d0,0.d0)
      dUJ=cmplx(0.d0,0.d0)

      do kk=1,nsnapw_type(itype0)
         ww(kk)=wsnap_type(itype0,kk,itype0)
         dww_dx(kk,:)=0.d0
      enddo

!cccccccccc Not really so sure whether this is necessary.
! In any case, in their SNAP paper, they have this, so I will add it here
! Is shows it is the spherical expansion for the whole rho, including
! the origin point. It might only affect j=0 point
      dx=0.d0   ! origin
      d=0.d0

      call calc_U_JM1M2(UJ,dUJ,mm_neigh,dx,d,Rc(itype0),1,nsnapw_type(itype0), &
         ww,dww_dx,jm,CC_func,jmm)
      dUJ=cmplx(0.d0,0.d0)
!cccccccccccccccccccccccccccccccccc

      do 1000 itype=1,ntypes
         do 1000 k=1,num_neigh(itype,iat)   ! this does not include the original point
            jj=ind_all_neigh(k,itype,iat)
            dx(1)=dR_neigh(1,k,itype,iat)
            dx(2)=dR_neigh(2,k,itype,iat)
            dx(3)=dR_neigh(3,k,itype,iat)
            dd=dx(1)**2+dx(2)**2+dx(3)**2
            d=dsqrt(dd)
            if(d.gt.Rc(itype0)) goto 1001

            y=pi*d/Rc(itype0)
            f2=0.5*(cos(y)+1)
            do kk=1,nsnapw_type(itype0)
               ww(kk)=f2*wsnap_type(itype,kk,itype0)
               dww_dx(kk,:)=-0.5*sin(y)*pi/Rc(itype0)*dx(:)/d*wsnap_type(itype,kk,itype0)
            enddo

            call calc_U_JM1M2(UJ,dUJ,mm_neigh,dx,d,Rc(itype0),jj,nsnapw_type(itype0), &
               ww,dww_dx,jm,CC_func,jmm)

1001        continue
1000  continue

      allocate(dsum(mm_neigh*3))
      do 1500 kk=1,nsnapw_type(itype0)
         do 1500 kkk=1,nBB(itype0)
            j1=jjj123(1,kkk,itype0)    ! stored the double index
            j2=jjj123(2,kkk,itype0)
            j=jjj123(3,kkk,itype0)
            !  Note: |j1-j2|.le.j, j.le.j1+j2, and mod(j1+j2-j,2).eq.0 (i.e, the original j1+j2-j is an integer

            sum=0.d0
            dsum=0.d0
            do m11=-j1,j1,2
               do m12=-j1,j1,2
                  do m21=-j2,j2,2
                     do m22=-j2,j2,2
                        ! do m1=-j,j,2
                        ! do m2=-j,j,2
                        ! if(m11+m21.eq.m1.and.m12+m22.eq.m2) then
                        m1=m11+m21
                        m2=m12+m22
                        if(abs(m1).le.j.and.abs(m2).le.j) then
                           y=Clebsch_Gordan(m1,m11,m21,j,j1,j2)*Clebsch_Gordan(m2,m12,m22,j,j1,j2)
                           sum=sum+y*conjg(UJ(m1,m2,j,kk))*UJ(m11,m12,j1,kk)*UJ(m21,m22,j2,kk)
                           !    write(*,*) "@@@@UJ(m1,m2,j,kk)",UJ(m1,m2,j,kk)
                           !    write(*,*) "@@@@UJ(m11,m12,j1,kk)",UJ(m11,m12,j1,kk)
                           !    write(*,*) "@@@@UJ(m21,m22,j2,kk)",UJ(m21,m22,j2,kk)
                           !    stop
                           dsum(:)=dsum(:)+y*  &
                              (conjg(DUJ(:,m1,m2,j,kk))*UJ(m11,m12,j1,kk)*UJ(m21,m22,j2,kk)+  &
                              conjg(UJ(m1,m2,j,kk))*DUJ(:,m11,m12,j1,kk)*UJ(m21,m22,j2,kk)+  &
                              conjg(UJ(m1,m2,j,kk))*UJ(m11,m12,j1,kk)*DUJ(:,m21,m22,j2,kk))
                           ! checked, indeed, after all the sum, only real part remain
                        endif
                        ! enddo
                        ! enddo
                     enddo
                  enddo
               enddo
            enddo
            feat_all(kkk+(kk-1)*nBB(itype0),iat1)=sum

            do jj3=1,3
               do jj=1,mm_neigh
                  jj4=jj+(jj3-1)*mm_neigh
                  dfeat_all(kkk+(kk-1)*nBB(itype0),iat1,jj,jj3)=dsum(jj4)
               enddo
            enddo
1500  continue
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      deallocate(UJ)
      deallocate(dUJ)
      deallocate(dsum)
3000 continue

   return
end subroutine find_feature_snap



