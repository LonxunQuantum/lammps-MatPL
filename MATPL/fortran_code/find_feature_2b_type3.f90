subroutine find_feature_2b_type3(num_neigh,dR_neigh,m_neigh,list_neigh,&
   n2b,n2bm,grid2_2,&
   feat_all,dfeat_all,nfeat0m)

   ! ******************************************
   !      feature calc on a SINGLE core
   ! ******************************************
   use mod_data, only : natoms, ntypes, catype

   implicit none
   integer, dimension(ntypes,natoms), intent(in) :: num_neigh
   real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh
   integer, intent(in) :: m_neigh
   integer, intent(in) :: list_neigh(m_neigh,ntypes,natoms)
   integer, intent(in) :: n2b(ntypes), n2bm
   real(8), dimension(2,n2bm+1,ntypes), intent(in) :: grid2_2
   integer, intent(in) :: nfeat0m
   real(8), intent(out) :: feat_all(nfeat0m,natoms)
   real(8), intent(out) :: dfeat_all(nfeat0m,natoms,m_neigh,3)

   integer :: num_neigh_alltype(natoms),nneigh
   real(8) :: dR_neigh_alltype(3,m_neigh,natoms)
   integer :: ind_all_neigh(m_neigh,ntypes,natoms),list_neigh_alltype(m_neigh,natoms)

   integer :: j,jj,num,itype,k
   integer :: iat,iat1
   real(8) :: d,dd
   real(8) :: pi,pi2,x,f1
   integer :: itype0
   real(8) :: y,y2

   real(8) :: feat2(n2bm,ntypes,natoms)
   real(8) :: dfeat2(n2bm,ntypes,natoms,m_neigh,3)

   integer :: nfeat_atom_tmp(natoms)


   jj = 0
   !  We need to clean us this later, everyone should only jave natom_n
   num_neigh_alltype=0
   do iat=1,natoms
      num=1
      list_neigh_alltype(1,iat)=iat   ! the first neighbore is itself
      dR_neigh_alltype(:,1,iat)=0.d0
      do itype=1,ntypes
         do j=1,num_neigh(itype,iat)
            num=num+1
            if(num.gt.m_neigh) then
               write(6,*) "total num_neigh > m_neigh, stop",m_neigh
               stop
            endif
            ind_all_neigh(j,itype,iat)=num
            list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
            dR_neigh_alltype(:,num,iat)=dR_neigh(:,j,itype,iat)
         enddo
      enddo
      num_neigh_alltype(iat)=num
   enddo

   pi=4*datan(1.d0)
   pi2=2*pi

   feat2=0.d0
   dfeat2=0.d0
   iat1=0

   ! write(*,*) "dbg info"
   ! write(*,*) "natoms:", natoms
   do iat=1,natoms
      iat1=iat1+1
      itype0=catype(iat)
      do itype=1,ntypes
         do j=1,num_neigh(itype,iat)
            jj=ind_all_neigh(j,itype,iat)
            dd=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
            d=dsqrt(dd)
            do k=1,n2b(itype0)
               if(d.ge.grid2_2(1,k,itype0).and.d.lt.grid2_2(2,k,itype0)) then
                  x=(d-grid2_2(1,k,itype0))/(grid2_2(2,k,itype0)-grid2_2(1,k,itype0))
                  y=(x-0.5d0)*pi2
                  f1=0.5d0*(cos(y)+1)
                  ! write(*,*) "dbg info",f1
                  feat2(k,itype,iat1)=feat2(k,itype,iat1) + f1
                  y2=-pi*sin(y)/(d*(grid2_2(2,k,itype0)-grid2_2(1,k,itype0)))

                  dfeat2(k,itype,iat1,jj,:) = dfeat2(k,itype,iat1,jj,:) + y2*dR_neigh(:,j,itype,iat)
                  dfeat2(k,itype,iat1, 1,:) = dfeat2(k,itype,iat1,1 ,:) - y2*dR_neigh(:,j,itype,iat)
               endif
            enddo
            !*********** So, one Rij will always have two features k, k+1  (1,2)****************
         enddo
      enddo
   enddo

   !   Now, we collect everything together, collapse the index (k,itype)
   !   feat2, into a single feature.
   !       feat_alltmp=0.d0
   !       dfeat_alltmp=0.d0

   nfeat_atom_tmp=0
   iat1=0

   do iat=1,natoms
      iat1=iat1+1
      itype0=catype(iat)
      nneigh=num_neigh_alltype(iat)
      num=0
      do itype=1,ntypes
         do k=1,n2b(itype0)
            num=num+1
            feat_all(num,iat1) = feat2(k,itype,iat1)
            dfeat_all(num,iat1,1:nneigh,:) = dfeat2(k,itype,iat1,1:nneigh,:)
         enddo
      enddo
      nfeat_atom_tmp(iat)=num
      if(num.gt.nfeat0m) then
         write(6,*) "num > nfeat0m,stop",num,nfeat0m
         stop
      endif
   enddo

   return
end subroutine find_feature_2b_type3