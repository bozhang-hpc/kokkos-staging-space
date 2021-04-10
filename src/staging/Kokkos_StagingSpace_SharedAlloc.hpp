#ifndef KOKKOS_STAGINGSPACE_SHARED_ALLOC_HPP
#define KOKKOS_STAGINGSPACE_SHARED_ALLOC_HPP

#include <cstdint>
#include <string>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace Kokkos {
namespace Impl {
/*
 *  Memory space specialization of SharedAllocationRecord< Kokkos::StagingSpace , void >
 * requires :
 *
 *  SharedAllocationRecord< Kokkos::StagingSpace , void > : public SharedAllocationRecord< void
 * , void >
 *  {
 *    // delete allocated user memory via static_cast to this type.
 *    static void deallocate( const SharedAllocationRecord<void,void> * );
 *    Space m_space ;
 *  }
 */

template <class DestroyFunctor>
class StagingSharedAllocationRecord
    : public SharedAllocationRecord<Kokkos::StagingSpace, void> {

private:
  StagingSharedAllocationRecord(const Kokkos::StagingSpace& arg_space,
                         const std::string& arg_label,
                         const size_t arg_alloc,
                         const size_t rank,
                         const enum Kokkos::StagingSpace::data_layout layout,
                         const size_t elem_size,
                         const size_t ub_N0, const size_t ub_N1,
                         const size_t ub_N2, const size_t ub_N3,
                         const size_t ub_N4, const size_t ub_N5,
                         const size_t ub_N6, const size_t ub_N7)
      //  Allocate user memory as [ SharedAllocationHeader , user_memory ] 
      : SharedAllocationRecord<Kokkos::StagingSpace, void>(
            arg_space, arg_label, arg_alloc, rank, layout, elem_size,
            ub_N0, ub_N1, ub_N2, ub_N3, ub_N4, ub_N5, ub_N6, ub_N7,
            &Kokkos::Impl::deallocate<Kokkos::StagingSpace, DestroyFunctor>),
        m_destroy() {}

public:
  DestroyFunctor m_destroy;

  // Allocate with a zero use count.  Incrementing the use count from zero to
  // one inserts the record into the tracking list.  Decrementing the count from
  // one to zero removes from the trakcing list and deallocates.
  KOKKOS_INLINE_FUNCTION static StagingSharedAllocationRecord* allocate(
      const Kokkos::StagingSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc, const size_t rank,
      const enum Kokkos::StagingSpace::data_layout layout, const size_t elem_size,
      const size_t ub_N0, const size_t ub_N1,
      const size_t ub_N2, const size_t ub_N3,
      const size_t ub_N4, const size_t ub_N5,
      const size_t ub_N6, const size_t ub_N7) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new StagingSharedAllocationRecord(arg_space, arg_label, 
                                      arg_alloc, rank, layout, elem_size,
                                      ub_N0, ub_N1,
                                      ub_N2, ub_N3,
                                      ub_N4, ub_N5,
                                      ub_N6, ub_N7);
#else
    (void)arg_space;
    (void)arg_label;
    (void)arg_alloc;
    return (StagingSharedAllocationRecord*)0;
#endif
  }

};


} //Impl
} //Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_STAGINGSPACE_SHARED_ALLOC_HPP */