subroutine find_feature_2bgauss(num_neigh,dR_neigh,m_neigh,list_neigh,&
   n2b,n2bm,grid2,wgauss,Rc,&
   feat_all,dfeat_all,nfeat0m)

   use mod_data, only : natoms, ntypes, catype
   implicit none
   integer, dimension(ntypes,natoms), intent(in) :: num_neigh
   real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh
   integer, intent(in) :: m_neigh
   integer, intent(in) :: list_neigh(m_neigh,ntypes,natoms)
   integer, intent(in) :: n2b(ntypes), n2bm
   real(8), intent(in) :: grid2(200,50),wgauss(200,50)
   real(8), intent(in) :: Rc(ntypes)
   integer, intent(in) :: nfeat0m
   real(8), intent(out) :: feat_all(nfeat0m,natoms)
   real(8), intent(out) :: dfeat_all(nfeat0m,natoms,m_neigh,3)

   integer :: num_neigh_alltype(natoms),nneigh
   real(8) :: dR_neigh_alltype(3,m_neigh,natoms)
   integer :: ind_all_neigh(m_neigh,ntypes,natoms),list_neigh_alltype(m_neigh,natoms)

   integer :: i,j,k,jj,num,itype,iat,iat1
   real(8) :: d,dd
   real(8) :: pi,pi2,x,f1
   real(8) :: Rt,f2,ff,dt
   integer :: itype0

   real(8) :: feat2(n2bm,ntypes,natoms)
   real(8) :: dfeat2(n2bm,ntypes,natoms,m_neigh,3)

   integer :: nfeat_atom_tmp(natoms)


   num_neigh_alltype=0
   do iat=1,natoms
      num=1
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
   pi=4*datan(1.d0)
   pi2=2*pi

   feat2=0.d0
   dfeat2=0.d0
   iat1=0
   do 3000 iat=1,natoms
      iat1=iat1+1
      itype0=catype(iat)
      do 1000 itype=1,ntypes
         do 1000 j=1,num_neigh(itype,iat)
            jj=ind_all_neigh(j,itype,iat)
            dd=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
            d=dsqrt(dd) ! Rij
            if(d.gt.Rc(itype0)) goto 1001
            do k=1,n2b(itype0)
               Rt=wgauss(k,itype0)
               dt=grid2(k,itype0)

               f1=exp(-((d-dt)/Rt)**2)
               x=pi*d/Rc(itype0)
               f2=0.5*(cos(x)+1)
               ff=-2*f1*f2/Rt**2*(d-dt)/d  ! derivative on f1
               ff=ff-0.5*sin(x)*pi/Rc(itype0)*f1/d   ! derivative on f2

               feat2(k,itype,iat1)=feat2(k,itype,iat1)+f1*f2

               dfeat2(k,itype,iat1,jj,:)=dfeat2(k,itype,iat1,jj,:)+ff*dR_neigh(:,j,itype,iat)
               dfeat2(k,itype,iat1,1,:)=dfeat2(k,itype,iat1,1,:)-ff*dR_neigh(:,j,itype,iat)
               ! Note, (k+1,itype) is the feature inde
            enddo
            !cccccccccccc So, one Rij will always have two features k, k+1  (1,2)
1001        continue
1000  continue
3000 continue

   !   Now, the three body feature
   !ccccccccccccccccccccccccccccccccccccc


   !cccccccccccccccccccccccccccccccccccccccccccccccc
   !cccccccccccccccccccccccccccccccccccccccccccccccc
   !   Now, we collect every types together, collapse the index (k,itype)
   !   feat2, into a single feature.
   nfeat_atom_tmp=0
   iat1=0
   do 5000 iat=1,natoms
      iat1=iat1+1
      itype0=catype(iat)
      nneigh=num_neigh_alltype(iat)
      num=0
      do itype=1,ntypes
         do k=1,n2b(itype0)
            num=num+1
            feat_all(num,iat1)=feat2(k,itype,iat1)
            dfeat_all(num,iat1,1:nneigh,:)=dfeat2(k,itype,iat1,1:nneigh,:)
         enddo
      enddo
      nfeat_atom_tmp(iat)=num
      if(num.gt.nfeat0m) then
         write(6,*) "num.gt.nfeat0m,stop",num,nfeat0m
         stop
      endif
5000 continue

!ccccccccccccccccccccccccccccccccccc
!  Now, we have to redefine the dfeat_all in another way.
!  dfeat_all(i,iat,jneigh,3) means:
!  d_ith_feat_of_iat/d_R(jth_neigh_of_iat)
!  dfeat_allR(i,iat,jneigh,3) means:
!  d_ith_feat_of_jth_neigh/d_R(iat)
!cccccccccccccccccccccccccccccccccccccc

!ccccccccccccccccccccccccccccccccccccc

   return
end subroutine find_feature_2bgauss



