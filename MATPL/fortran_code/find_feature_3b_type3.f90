subroutine find_feature_3b_type3(num_neigh,dR_neigh,m_neigh,list_neigh,&
   n3b1,n3b2,n3b1m,n3b2m,&
   Rc2_type,grid31_2,grid32_2,&
   feat_all,dfeat_all,nfeat0m)

   use mod_data, only : natoms, ntypes, catype

   implicit none
   integer, dimension(ntypes,natoms), intent(in) :: num_neigh
   real(8), dimension(3,m_neigh,ntypes,natoms), intent(in) :: dR_neigh
   integer, intent(in) :: m_neigh
   integer, intent(in) :: list_neigh(m_neigh,ntypes,natoms)
   integer, intent(in) :: n3b1(ntypes),n3b2(ntypes),n3b1m,n3b2m
   real(8), intent(in) :: Rc2_type(ntypes),grid31_2(2,n3b1m,ntypes),grid32_2(2,n3b2m,ntypes)
   integer, intent(in) :: nfeat0m
   real(8), intent(out) :: feat_all(nfeat0m,natoms)
   real(8), intent(out) :: dfeat_all(nfeat0m,natoms,m_neigh,3)

   integer :: num_neigh_alltype(natoms),nneigh
   real(8) :: dR_neigh_alltype(3,m_neigh,natoms)
   integer :: nfeat_atom_tmp(natoms)
   real(8) :: feat3_tmp(n3b1m,m_neigh,ntypes)
   real(8) :: dfeat3_tmp(n3b1m,m_neigh,ntypes,3)
   integer :: ind_f(n3b1m,m_neigh,ntypes,natoms)
   integer :: num_k(m_neigh,ntypes),num12
   real(8) :: f32(n3b2m),df32(n3b2m,2,3)
   integer :: itype12,ind_f32(n3b2m)
   integer :: ind_all_neigh(m_neigh,ntypes,natoms),list_neigh_alltype(m_neigh,natoms)
   real(8) :: feat3(n3b1m*n3b1m*n3b2m,ntypes*(ntypes+1)/2,natoms)
   real(8) :: dfeat3(n3b1m*n3b1m*n3b2m,ntypes*(ntypes+1)/2,natoms,m_neigh,3)

   integer :: num,itype,itype0,itype1,itype2
   integer :: i1,i2,j,j1,j2,iat,iat1
   integer :: k,k1,k2,k12,j12,ii_f,jj,jj1,jj2
   real(8) :: d,dd
   real(8) :: pi,pi2,x,f1
   real(8) :: y,y2


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
   feat3=0.d0
   dfeat3=0.d0

   iat1=0
   do iat=1,natoms
      iat1=iat1+1
      itype0=catype(iat)
      do 1000 itype=1,ntypes
         do 1000 j=1,num_neigh(itype,iat) ! for itype, the self is not in the neighbore list
            jj=ind_all_neigh(j,itype,iat)   ! but itself is in jj=1
            dd=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
            d=dsqrt(dd)
            !ccccccccccccccccccccccccccccccccccccccccccccccccccccc
            num=0
            do k=1,n3b1(itype0)
               if(d.ge.grid31_2(1,k,itype0).and.d.lt.grid31_2(2,k,itype0)) then
                  num=num+1
                  x=(d-grid31_2(1,k,itype0))/(grid31_2(2,k,itype0)-grid31_2(1,k,itype0))
                  y=(x-0.5d0)*pi2
                  f1=0.5d0*(cos(y)+1)
                  feat3_tmp(num,j,itype)=f1
                  ind_f(num,j,itype,iat)=k
                  y2=-pi*sin(y)/(d*(grid31_2(2,k,itype0)-grid31_2(1,k,itype0)))
                  dfeat3_tmp(num,j,itype,:)=y2*dR_neigh(:,j,itype,iat)
               endif
            enddo
            num_k(j,itype)=num

            !cccccccccccc So, one Rij will always have two features k, k+1  (1,2)
            !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
1000  continue
      !   Now, the three body feature
      !ccccccccccccccccccccccccccccccccccccc
      do 2000 itype2=1,ntypes
         do 2000 itype1=1,itype2
            itype12=itype1+((itype2-1)*itype2)/2
            do 2000 j2=1,num_neigh(itype2,iat)
               do 2000 j1=1,num_neigh(itype1,iat)
                  !  if(itype1.eq.itype2.and.j1.ge.j2) goto 2000
                  if(itype1.eq.itype2.and.j1.eq.j2) goto 2000
                  jj1=ind_all_neigh(j1,itype1,iat)
                  jj2=ind_all_neigh(j2,itype2,iat)
                  dd=(dR_neigh(1,j1,itype1,iat)-dR_neigh(1,j2,itype2,iat))**2+ &
                     (dR_neigh(2,j1,itype1,iat)-dR_neigh(2,j2,itype2,iat))**2+ &
                     (dR_neigh(3,j1,itype1,iat)-dR_neigh(3,j2,itype2,iat))**2
                  d=dsqrt(dd)
                  if(d.gt.Rc2_type(itype0).or.d.lt.1.D-4) goto 2000
                  num=0
                  do k=1,n3b2(itype0)
                     if(d.ge.grid32_2(1,k,itype0).and.d.lt.grid32_2(2,k,itype0)) then
                        num=num+1
                        x=(d-grid32_2(1,k,itype0))/(grid32_2(2,k,itype0)-grid32_2(1,k,itype0))
                        y=(x-0.5d0)*pi2
                        f1=0.5d0*(cos(y)+1)
                        f32(num)=f1
                        ind_f32(num)=k
                        y2=-pi*sin(y)/(d*(grid32_2(2,k,itype0)-grid32_2(1,k,itype0)))
                        df32(num,1,:)=y2*(dR_neigh(:,j1,itype1,iat)-dR_neigh(:,j2,itype2,iat))
                        df32(num,2,:)=-df32(num,1,:)
                     endif
                  enddo
                  num12=num

                  !cccccccccccccccccccccccc
                  !   Each R has two k features, so for the three R, we have the following
                  do i1=1,num_k(j1,itype1)
                     do i2=1,num_k(j2,itype2)
                        do j12=1,num12
                           k1=ind_f(i1,j1,itype1,iat)
                           k2=ind_f(i2,j2,itype2,iat)
                           k12=ind_f32(j12)

                           ii_f=0
                           if(itype1.ne.itype2) then
                              ii_f=k1+(k2-1)*n3b1(itype0)+(k12-1)*n3b1(itype0)**2
                           endif
                           if(itype1.eq.itype2.and.k1.le.k2) then
                              ii_f=k1+((k2-1)*k2)/2+(k12-1)*(n3b1(itype0)*(n3b1(itype0)+1))/2
                           endif

                           if(ii_f.ne.0) then
                              feat3(ii_f,itype12,iat1)=feat3(ii_f,itype12,iat1)+ &
                                 feat3_tmp(i1,j1,itype1)*feat3_tmp(i2,j2,itype2)*f32(j12)

                              dfeat3(ii_f,itype12,iat1,jj1,:)=dfeat3(ii_f,itype12,iat1,jj1,:)+ &
                                 dfeat3_tmp(i1,j1,itype1,:)*feat3_tmp(i2,j2,itype2)*f32(j12)+ &
                                 feat3_tmp(i1,j1,itype1)*feat3_tmp(i2,j2,itype2)*df32(j12,1,:)

                              dfeat3(ii_f,itype12,iat1,jj2,:)=dfeat3(ii_f,itype12,iat1,jj2,:)+ &
                                 feat3_tmp(i1,j1,itype1)*dfeat3_tmp(i2,j2,itype2,:)*f32(j12)+ &
                                 feat3_tmp(i1,j1,itype1)*feat3_tmp(i2,j2,itype2)*df32(j12,2,:)

                              dfeat3(ii_f,itype12,iat1,1,:)=dfeat3(ii_f,itype12,iat1,1,:)- &
                                 dfeat3_tmp(i1,j1,itype1,:)*feat3_tmp(i2,j2,itype2)*f32(j12)- &
                                 feat3_tmp(i1,j1,itype1)*dfeat3_tmp(i2,j2,itype2,:)*f32(j12)

                              !cccc (ii_f,itype12) is the feature index
                           endif
                        enddo
                     enddo
                  enddo
2000  continue

   enddo
   !cccccccccccccccccccccccccccccccccccccccccccccccc
   !cccccccccccccccccccccccccccccccccccccccccccccccc
   !   Now, we collect everything together, collapse the index (k,itype)
   !   and feat2,feat3, into a single feature.

   nfeat_atom_tmp=0
   iat1=0
   do iat=1,natoms
      !if(mod(iat-1,nnodes).eq.inode-1) then
      iat1=iat1+1
      itype0=catype(iat)
      nneigh=num_neigh_alltype(iat)
      num=0
      do itype2=1,ntypes
         do itype1=1,itype2
            itype12=itype1+((itype2-1)*itype2)/2
            do k1=1,n3b1(itype0)
               do k2=1,n3b1(itype0)
                  do k12=1,n3b2(itype0)
                     ii_f=0
                     if(itype1.ne.itype2) then
                        ii_f=k1+(k2-1)*n3b1(itype0)+(k12-1)*n3b1(itype0)**2
                     endif
                     if(itype1.eq.itype2.and.k1.le.k2) then
                        ii_f=k1+((k2-1)*k2)/2+(k12-1)*(n3b1(itype0)*(n3b1(itype0)+1))/2
                     endif
                     if(ii_f.gt.0) then
                        num=num+1
                        feat_all(num,iat1)=feat3(ii_f,itype12,iat1)
                        dfeat_all(num,iat1,1:nneigh,:)=dfeat3(ii_f,itype12,iat1,1:nneigh,:)
                     endif
                  enddo
               enddo
            enddo
         enddo
      enddo
      nfeat_atom_tmp(iat)=num
      if(num.gt.nfeat0m) then
         write(6,*) "num.gt.nfeat0m,stop",num,nfeat0m
         stop
      endif
   enddo
   !ccccccccccccccccccccccccccccccccccc
   !  Now, we have to redefine the dfeat_all in another way.
   !  dfeat_all(nfeat,iat,jneigh,3) means:
   !  d_jth_feat_of_iat/d_R(jth_neigh_of_iat)
   !  dfeat_allR(nfeat,iat,jneigh,3) means:
   !  d_jth_feat_of_jth_neigh/d_R(iat)
   !  Now, just output dfeat_all
   !cccccccccccccccccccccccccccccccccccccc

   return
end subroutine find_feature_3b_type3
