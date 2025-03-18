module dp_ff_mod
    
    ! **************************************************************
    !           Overarching data module for forece field  
    !   Will be called from LAMMPS pair constructor and destructor 
    !               L. Wang 2023.2 
    ! **************************************************************
    use iso_c_binding
    implicit none

    ! wrap up a ff with an struct
    type single_ff
      ! local temporary variables
      integer(4) dp_ff_is_fn_recon               ! marker for fitting net reconnected 
      
      !character(200) ff_name

      ! force field paras
      integer(4) dp_ff_model_idx
      integer(4) dp_ff_num_type
      integer(4) dp_ff_max_neigh         ! max nieghbor num 
      integer(4) dp_ff_itype_atom(10)        ! atom type list 

      ! embedding net part 
      integer(4) dp_ff_ntype_pair 
      integer(4) dp_ff_nlayer_em
      integer(4) dp_ff_node_em(20)
      integer(4) dp_ff_nodeMM_em          ! max node dim embedding net
      real(8), allocatable,dimension(:,:,:,:,:) :: dp_ff_Wij_em
      real(8), allocatable,dimension(:,:,:,:) :: dp_ff_B_em

      ! fitting net part 
      integer(4) dp_ff_nlayer_nn
      integer(4) dp_ff_node_nn(20) 
      integer(4) dp_ff_nodeMM_nn
      real(8), allocatable,dimension(:,:,:,:) :: dp_ff_Wij_NN
      real(8), allocatable,dimension(:,:,:) :: dp_ff_B_NN

      ! recconnect paras 
      integer dp_ff_iflag_resNN(100) 
      real(8), allocatable, dimension(:,:,:) :: dp_ff_W_res_NN

      ! gen_dp file 
      integer(4) dp_ff_iat_type(100)
      
      integer(4) dp_ff_max
      real(8) dp_ff_Rc_M 
      integer(4) dp_ff_M2               ! M2 in DP net

      real(8) dp_ff_Rc_type(100), dp_ff_R_cs(100)
      real(8) dp_ff_ave_shift(4,100),dp_ff_ave_norm(4,100)
    end type single_ff

    ! structs for ff data 
    type(single_ff) ff(4) 
    
    ! loop index
    integer(4) i, j1, j2, ii
    integer(4) ll 
    integer(4) itype, itype1, itype2

    ! others tmps
    integer(4) tmp
    integer(4) ntype_tmp, nlayer_nn_tmp
    integer(4) node_nn_tmp(100)   
    integer(4) node_tmp
    logical(2) alive_dp 

    contains 
        
    subroutine dp_ff_load(name_ptr, ff_idx, slen, ocut, m_neigh) bind(c,name="dp_ff_load") 
      !is_fn_recon = .true. 

      character, dimension(300), intent(in) :: name_ptr    ! name string pointer
      integer, intent(in) :: slen            ! length of name string 
      integer, intent(in) :: ff_idx          ! index of ff to be loaded 
      real(8), intent(out) :: ocut
      integer, intent(out) :: m_neigh
      character(300) temp
      
      !write(*,*) "name_ptr(1) ", name_ptr(1) 

      temp = trim(name_ptr(1))
      !write(*,*) "start", temp
      do i=2,slen
        temp = trim(temp)//trim(name_ptr(i))
      enddo 

      inquire(file=temp,exist=alive_dp)
      if (alive_dp.ne..true.) then 
        write(*,*) "force field named ", temp, " not found. Terminate."
        stop
      endif 
      
      ! *********************************************
      !      Allocate the ff pointers and load
      ! *********************************************
      
      ! reading feat.info part
      open(10,file=temp)
      read(10,*) ff(ff_idx)%dp_ff_model_idx
      if (ff(ff_idx)%dp_ff_model_idx.ne.5) then
        write(*,*) "Model type error. Should be 5"
        stop
      endif 
      
      read(10,*) ff(ff_idx)%dp_ff_is_fn_recon 

      read(10,*) ff(ff_idx)%dp_ff_num_type, ff(ff_idx)%dp_ff_max_neigh

      m_neigh = ff(1)%dp_ff_max_neigh
      
      do i=1,ff(ff_idx)%dp_ff_num_type
        read(10,*) ff(ff_idx)%dp_ff_itype_atom(i)
      enddo
      
      ! reading embedding net part
      read(10,*) ff(ff_idx)%dp_ff_ntype_pair
      read(10,*) ff(ff_idx)%dp_ff_nlayer_em   ! _em: embedding  3
      read(10,*) (ff(ff_idx)%dp_ff_node_em(i), &
                   i=1, ff(ff_idx)%dp_ff_nlayer_em+1)   ! 1 25 25 25

      !enddo 

      if(ff(ff_idx)%dp_ff_ntype_pair.ne.ff(ff_idx)%dp_ff_num_type**2) then
        write(6,*) "ntype_pair not equal to ntype**2, terminate. ", &
          ff(ff_idx)%dp_ff_ntype_pair, ff(ff_idx)%dp_ff_num_type
        stop
      endif
      
      ff(ff_idx)%dp_ff_nodeMM_em=0 
      
      do itype=1,ff(ff_idx)%dp_ff_ntype_pair
        do ii=1,ff(ff_idx)%dp_ff_nlayer_em+1
          if(ff(ff_idx)%dp_ff_node_em(ii).gt.ff(ff_idx)%dp_ff_nodeMM_em) then
            ff(ff_idx)%dp_ff_nodeMM_em = ff(ff_idx)%dp_ff_node_em(ii)
          endif
        enddo
      enddo

      allocate(ff(ff_idx)%dp_ff_Wij_em( ff(ff_idx)%dp_ff_nodeMM_em, & 
                                        ff(ff_idx)%dp_ff_nodeMM_em, & 
                                        ff(ff_idx)%dp_ff_nlayer_em, &
                                        ff(ff_idx)%dp_ff_num_type, & 
                                        ff(ff_idx)%dp_ff_num_type))
      
      allocate(ff(ff_idx)%dp_ff_B_em( ff(ff_idx)%dp_ff_nodeMM_em, & 
                                      ff(ff_idx)%dp_ff_nlayer_em, & 
                                      ff(ff_idx)%dp_ff_num_type, & 
                                      ff(ff_idx)%dp_ff_num_type))

      do itype1=1,ff(ff_idx)%dp_ff_num_type
        do itype2=1,ff(ff_idx)%dp_ff_num_type
          do ll=1,ff(ff_idx)%dp_ff_nlayer_em
            do j1=1,ff(ff_idx)%dp_ff_node_em(ll)
              read(10,*) (ff(ff_idx)%dp_ff_Wij_em(j1,j2,ll,itype2,itype1), &
                           j2=1,ff(ff_idx)%dp_ff_node_em(ll+1))
                          !  write(6,*) (ff(ff_idx)%dp_ff_Wij_em(j1,1,3,2,2), &
                          !  j2=1,ff(ff_idx)%dp_ff_node_em(ll+1))
            enddo
            read(10,*) (ff(ff_idx)%dp_ff_B_em(j2,ll,itype2,itype1), &
                         j2=1,ff(ff_idx)%dp_ff_node_em(ll+1))
                        !  write(6,*) (ff(ff_idx)%dp_ff_B_em(j2,1,1,1), &
                        !  j2=1,ff(ff_idx)%dp_ff_node_em(ll+1))
          enddo
        enddo
      enddo

      ! reading fitting net part
      read(10,*) tmp
      read(10,*) ff(ff_idx)%dp_ff_nlayer_nn                ! layer 
      read(10,*) (ff(ff_idx)%dp_ff_node_nn(ll),ll=1,ff(ff_idx)%dp_ff_nlayer_nn+1)

      ff(ff_idx)%dp_ff_nodeMM_nn=0

      do itype=1,ff(ff_idx)%dp_ff_num_type
        do ii=1,ff(ff_idx)%dp_ff_nlayer_nn+1
          if(ff(ff_idx)%dp_ff_node_nn(ii).gt.ff(ff_idx)%dp_ff_nodeMM_nn) then
            ff(ff_idx)%dp_ff_nodeMM_nn = ff(ff_idx)%dp_ff_node_nn(ii)
          endif
        enddo
      enddo
      
      allocate(ff(ff_idx)%dp_ff_Wij_nn( ff(ff_idx)%dp_ff_nodeMM_nn, & 
                                        ff(ff_idx)%dp_ff_nodeMM_nn, & 
                                        ff(ff_idx)%dp_ff_nlayer_nn, &
                                        ff(ff_idx)%dp_ff_num_type))
      
      allocate(ff(ff_idx)%dp_ff_B_nn( ff(ff_idx)%dp_ff_nodeMM_nn, & 
                                      ff(ff_idx)%dp_ff_nlayer_nn, & 
                                      ff(ff_idx)%dp_ff_num_type))
      
      do itype=1,ff(ff_idx)%dp_ff_num_type
        do ll=1,ff(ff_idx)%dp_ff_nlayer_nn
          do j1=1,ff(ff_idx)%dp_ff_node_nn(ll)
            read(10,*) (ff(ff_idx)%dp_ff_Wij_nn(j1,j2,ll,itype), &
                         j2=1,ff(ff_idx)%dp_ff_node_nn(ll+1))
          enddo
          read(10,*) (ff(ff_idx)%dp_ff_B_nn(j2,ll,itype), &
                       j2=1,ff(ff_idx)%dp_ff_node_nn(ll+1))
        enddo
      enddo
      
      ! paras for reconnection 
      ! DEFAULT: only fitting net 
      if (ff(ff_idx)%dp_ff_is_fn_recon .eq. 1) then
        read(10,*) ntype_tmp
        read(10,*) nlayer_nn_tmp
        read(10,*) (node_nn_tmp(ll),ll=1,nlayer_nn_tmp+1)

        if(ntype_tmp.ne.ff(ff_idx)%dp_ff_num_type) then
          write(*,*) "number of atom type mismatch, terminate", &
            ntype_tmp, ff(ff_idx)%dp_ff_num_type
          stop
        endif

        if(nlayer_nn_tmp.ne.ff(ff_idx)%dp_ff_nlayer_nn) then
          write(*,*) "number of layer mismatch, terminate"
          stop
        endif

        allocate(ff(ff_idx)%dp_ff_W_res_NN( ff(ff_idx)%dp_ff_nodeMM_nn, &
                                            ff(ff_idx)%dp_ff_nlayer_nn+1, & 
                                            ff(ff_idx)%dp_ff_num_type))

        read(10,*) (ff(ff_idx)%dp_ff_iflag_resNN(ll), &
                     ll=1, ff(ff_idx)%dp_ff_nlayer_nn+1)
        
        do itype=1,ff(ff_idx)%dp_ff_num_type
          do ll=1,ff(ff_idx)%dp_ff_nlayer_nn+1
            if(ff(ff_idx)%dp_ff_iflag_resNN(ll).eq.1) then
              read(10,*) node_tmp
              read(10,*) (ff(ff_idx)%dp_ff_W_res_NN(j1,ll,itype),j1=1,node_tmp)
            endif
          enddo
        enddo

      endif 

      !reading gen_dp.in part 
      read(10,*) ff(ff_idx)%dp_ff_Rc_M
      read(10,*) ff(ff_idx)%dp_ff_M2 
      read(10,*) tmp 
              
      do i=1,ff(ff_idx)%dp_ff_num_type
        read(10,*) ff(ff_idx)%dp_ff_iat_type(i)
        read(10,*) ff(ff_idx)%dp_ff_Rc_type(i),ff(ff_idx)%dp_ff_R_cs(i)
        
        if(ff(ff_idx)%dp_ff_Rc_type(i).gt.ff(ff_idx)%dp_ff_Rc_M) then
          write(6,*) "Rc_type must be smaller than Rc_M", &
            i, ff(ff_idx)%dp_ff_Rc_type(i), ff(ff_idx)%dp_ff_Rc_M
          stop
        endif

        read(10,*) ff(ff_idx)%dp_ff_ave_shift(1,i), &
                   ff(ff_idx)%dp_ff_ave_shift(2,i), &
                   ff(ff_idx)%dp_ff_ave_shift(3,i), &
                   ff(ff_idx)%dp_ff_ave_shift(4,i)
        read(10,*) ff(ff_idx)%dp_ff_ave_norm(1,i), &
                   ff(ff_idx)%dp_ff_ave_norm(2,i), &
                   ff(ff_idx)%dp_ff_ave_norm(3,i), &
                   ff(ff_idx)%dp_ff_ave_norm(4,i)
      enddo

      ocut = ff(1)%dp_ff_Rc_M

      close(10)
        
    end subroutine

    subroutine dp_ff_deallocate(ff_idx) bind(c,name="dp_ff_deallocate") 
      ! deallocate the ff pointers
      integer, intent(in) :: ff_idx
      ! embedding net 

      !if (allocated(ff(ff_idx)%dp_ff_Wij_em)) then
        deallocate(ff(ff_idx)%dp_ff_Wij_em)
      !endif

      !if (allocated(ff(ff_idx)%dp_ff_B_em)) then
        deallocate(ff(ff_idx)%dp_ff_B_em)
      !endif

      ! fitting net 
      !if (allocated(ff(ff_idx)%dp_ff_Wij_NN)) then
        deallocate(ff(ff_idx)%dp_ff_Wij_NN)
      !endif

      !if (allocated(ff(ff_idx)%dp_ff_B_nn)) then
        deallocate(ff(ff_idx)%dp_ff_B_nn)
      !endif
      
      ! recon net 
      !if (allocated(ff(ff_idx)%dp_ff_W_res_NN)) then
        deallocate(ff(ff_idx)%dp_ff_W_res_NN)
      !endif

    end subroutine
    
end module dp_ff_mod
