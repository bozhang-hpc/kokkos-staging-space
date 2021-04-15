#ifndef KOKKOS_STAGINGSPACE_SHARED_ALLOC_HPP
#define KOKKOS_STAGINGSPACE_SHARED_ALLOC_HPP

#include <cstdint>
#include <string>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace Kokkos {
namespace Impl {

template <class DestroyFunctor>
class StagingSharedAllocationRecord
    : public SharedAllocationRecord<Kokkos::StagingSpace, void> {

private:
  StagingSharedAllocationRecord(const Kokkos::StagingSpace& arg_space,
                         const std::string& arg_label,
                         const size_t arg_alloc,
                         const size_t rank,
                         const enum Kokkos::StagingSpace::data_layout layout,
                         const size_t elem_size, const size_t* ub)
      //  Allocate user memory as [ SharedAllocationHeader , user_memory ] 
      : SharedAllocationRecord<Kokkos::StagingSpace, void>(
            arg_space, arg_label, arg_alloc, rank, layout, elem_size, ub,
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
      const enum Kokkos::StagingSpace::data_layout layout,
      const size_t elem_size, const size_t* ub) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new StagingSharedAllocationRecord(arg_space, arg_label, 
                                      arg_alloc, rank, layout, elem_size, ub);
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