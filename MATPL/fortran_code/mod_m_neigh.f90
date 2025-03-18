module mod_m_neigh
   use mod_data, only : iflag_model
   use li_ff_mod, only : li_ff
   use nn_ff_mod, only : nn_ff
   use dp_ff_mod, only : ff

   implicit none
   integer :: m_neigh   ! model max num of neighbors

contains

   subroutine load_m_neigh()
      if (iflag_model == 1) then
         m_neigh = li_ff(1)%ff_max_neigh
      elseif (iflag_model == 3) then
         m_neigh = nn_ff(1)%ff_max_neigh
      elseif (iflag_model == 5) then
         m_neigh = ff(1)%dp_ff_max_neigh
      else
         write(*,*) 'error: iflag_model is wrong'
         stop
      endif
   end subroutine load_m_neigh

end module mod_m_neigh
