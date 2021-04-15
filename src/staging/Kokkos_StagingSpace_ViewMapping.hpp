#ifndef KOKKOS_STAGINGSPACE_VIEW_MAPPING_HPP
#define KOKKOS_STAGINGSPACE_VIEW_MAPPING_HPP

#include <type_traits>
#include <iostream>
#include <string>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_Extents.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_ViewTracker.hpp>
#include <impl/Kokkos_ViewCtor.hpp>
#include <impl/Kokkos_Tools.hpp>
#include <Kokkos_StagingSpace.hpp>

namespace Kokkos {

template <class... Prop>
struct ViewTraits<void, Kokkos::StagingSpace, Prop...> {
  // Specify Space, memory traits should be the only subsequent argument.

  static_assert(
    std::is_same<typename ViewTraits<void, Prop...>::execution_space,
                  void>::value &&
    std::is_same<typename ViewTraits<void, Prop...>::memory_space,
                  void>::value &&
    std::is_same<typename ViewTraits<void, Prop...>::HostMirrorSpace,
                  void>::value &&
    std::is_same<typename ViewTraits<void, Prop...>::array_layout,
                  void>::value,
    "Only one View Execution or Memory Space template argument");

  using execution_space  = typename Kokkos::StagingSpace::execution_space;
  using memory_space    = typename Kokkos::StagingSpace::memory_space;
  using HostMirrorSpace = typename Kokkos::Impl::HostMirror<Kokkos::StagingSpace>::Space;
  using array_layout    = typename execution_space::array_layout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = Kokkos::StagingSpaceSpecializeTag;
};

namespace Impl {

template <class ViewTraits>
struct StagingDataElement {
  using value_type           = typename ViewTraits::value_type;
  using const_value_type     = typename ViewTraits::const_value_type;
  using non_const_value_type = typename ViewTraits::non_const_value_type;
  volatile value_type* const ptr;

  KOKKOS_INLINE_FUNCTION
  StagingDataElement(value_type* ptr_) : ptr(nullptr) {}


};

template <class ViewTraits>
struct StagingViewDataHandle {
  typename ViewTraits::value_type* ptr;

  KOKKOS_INLINE_FUNCTION
  StagingViewDataHandle() : ptr(nullptr) {}
  KOKKOS_INLINE_FUNCTION
  StagingViewDataHandle(typename ViewTraits::value_type* ptr_) : ptr(nullptr) {} 

};

template <class Traits>
struct ViewDataHandle<
    Traits, typename std::enable_if<
                  std::is_same<typename Traits::specialize,
                              Kokkos::StagingSpaceSpecializeTag>::value>::type> {

  using value_type  = typename Traits::value_type;
  using handle_type = typename Kokkos::Impl::StagingViewDataHandle<Traits>;
  using return_type = typename Kokkos::Impl::StagingDataElement<Traits>;
  using track_type  = Kokkos::Impl::SharedAllocationTracker;

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type *arg_data_ptr,
                            track_type const &arg_tracker) {
    return handle_type(arg_data_ptr);
  }

};

template <class Traits> 
class ViewMapping<Traits, Kokkos::StagingSpaceSpecializeTag> {

public:
  using offset_type = ViewOffset<typename Traits::dimension,
                                 typename Traits::array_layout, void>;
  
  using handle_type = typename ViewDataHandle<Traits>::handle_type;

  handle_type m_impl_handle;
  offset_type m_impl_offset;

private:
  template <class, class...> friend class ViewMapping;

  KOKKOS_INLINE_FUNCTION
  ViewMapping(const handle_type& arg_handle, const offset_type& arg_offset)
      : m_impl_handle(arg_handle), m_impl_offset(arg_offset) {}

public:
  using printable_label_typedef = void;
  enum { is_managed = Traits::is_managed };

  //----------------------------------------
  // Domain dimensions

  enum { Rank = Traits::dimension::rank };

  template <typename iType>
  KOKKOS_INLINE_FUNCTION constexpr size_t extent(const iType& r) const {
    return m_impl_offset.m_dim.extent(r);
  }

  static KOKKOS_INLINE_FUNCTION constexpr size_t static_extent(
      const unsigned r) noexcept {
    using dim_type = typename offset_type::dimension_type;
    return dim_type::static_extent(r);
  }

  KOKKOS_INLINE_FUNCTION constexpr typename Traits::array_layout layout()
      const {
    return m_impl_offset.layout();
  }

  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0() const {
    return m_impl_offset.dimension_0();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_1() const {
    return m_impl_offset.dimension_1();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_2() const {
    return m_impl_offset.dimension_2();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_3() const {
    return m_impl_offset.dimension_3();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_4() const {
    return m_impl_offset.dimension_4();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_5() const {
    return m_impl_offset.dimension_5();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_6() const {
    return m_impl_offset.dimension_6();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_7() const {
    return m_impl_offset.dimension_7();
  }

  // Is a regular layout with uniform striding for each index.
  using is_regular = typename offset_type::is_regular;

  KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() const {
    return m_impl_offset.stride_0();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() const {
    return m_impl_offset.stride_1();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() const {
    return m_impl_offset.stride_2();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() const {
    return m_impl_offset.stride_3();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() const {
    return m_impl_offset.stride_4();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() const {
    return m_impl_offset.stride_5();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() const {
    return m_impl_offset.stride_6();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() const {
    return m_impl_offset.stride_7();
  }

  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType* const s) const {
    m_impl_offset.stride(s);
  }

  //----------------------------------------
  // Range span

  /** \brief  Span of the mapped range */
  KOKKOS_INLINE_FUNCTION constexpr size_t span() const {
    return m_impl_offset.span();
  }

  /** \brief  Is the mapped range span contiguous */
  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return m_impl_offset.span_is_contiguous();
  }

  using reference_type = typename ViewDataHandle<Traits>::return_type;
  using pointer_type   = typename Traits::value_type*;

  //----------------------------------------

private:
  enum { MemorySpanMask = 8 - 1 /* Force alignment on 8 byte boundary */ };
  enum { MemorySpanSize = sizeof(typename Traits::value_type) };

public:
  /** \brief  Span, in bytes, of the referenced memory */
  KOKKOS_INLINE_FUNCTION constexpr size_t memory_span() const {
    return (m_impl_offset.span() * sizeof(typename Traits::value_type) +
            MemorySpanMask) &
           ~size_t(MemorySpanMask);
  }

  //----------------------------------------

  KOKKOS_DEFAULTED_FUNCTION ~ViewMapping() = default;
  KOKKOS_INLINE_FUNCTION ViewMapping() : m_impl_handle(), m_impl_offset() {}
  KOKKOS_DEFAULTED_FUNCTION ViewMapping(const ViewMapping&) = default;
  KOKKOS_DEFAULTED_FUNCTION ViewMapping& operator=(const ViewMapping&) =
      default;

  KOKKOS_DEFAULTED_FUNCTION ViewMapping(ViewMapping&&) = default;
  KOKKOS_DEFAULTED_FUNCTION ViewMapping& operator=(ViewMapping&&) = default;

  //----------------------------------------

  /**\brief  Span, in bytes, of the required memory */
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t memory_span(
      typename Traits::array_layout const& arg_layout) {
    using padding = std::integral_constant<unsigned int, 0>;
    return (offset_type(padding(), arg_layout).span() * MemorySpanSize +
            MemorySpanMask) &
           ~size_t(MemorySpanMask);
  }

  /**\brief  Wrap a span of memory */
  template <class... P>
  KOKKOS_INLINE_FUNCTION ViewMapping(
      Kokkos::Impl::ViewCtorProp<P...> const& arg_prop,
      typename Traits::array_layout const& arg_layout)
      : m_impl_handle(
            ((Kokkos::Impl::ViewCtorProp<void, pointer_type> const&)arg_prop)
                .value),
        m_impl_offset(std::integral_constant<unsigned, 0>(), arg_layout) {}

  //----------------------------------------
  /*  Allocate and construct mapped array.
   *  Allocate via shared allocation record and
   *  return that record for allocation tracking.
   */

  template <class... P>
  Kokkos::Impl::SharedAllocationRecord<>* allocate_shared(
      Kokkos::Impl::ViewCtorProp<P...> const& arg_prop,
      typename Traits::array_layout const& arg_layout) {

    using alloc_prop        = Kokkos::Impl::ViewCtorProp<P...>;
    using execution_space   = typename alloc_prop::execution_space;
    using memory_space      = typename Traits::memory_space;
    using value_type        = typename Traits::value_type;
    using functor_type      = ViewValueFunctor<execution_space, value_type>;
    // using functor_type    = ViewValueFunctor<execution_space, value_type>;
    using record_type       = Kokkos::Impl::StagingSharedAllocationRecord<functor_type>;

    // Query the mapping for byte-size of allocation.
    // If padding is allowed then pass in sizeof value type
    // for padding computation.
    using padding         = std::integral_constant<
                              unsigned int, alloc_prop::allow_padding ? sizeof(value_type) : 0>;

    m_impl_offset = offset_type(padding(), arg_layout);

    const size_t alloc_size =
        (m_impl_offset.span() * MemorySpanSize + MemorySpanMask) &
        ~size_t(MemorySpanMask);

    const std::string& alloc_name =
        static_cast<Kokkos::Impl::ViewCtorProp<void, std::string> const&>(
            arg_prop)
            .value;

    enum Kokkos::StagingSpace::data_layout data_layout;
    if(std::is_same<typename Traits::array_layout, Kokkos::LayoutLeft>::value) {
      data_layout = Kokkos::StagingSpace::data_layout::LAYOUT_LEFT;
    } else {
      data_layout = Kokkos::StagingSpace::data_layout::LAYOUT_RIGHT;
    }
    size_t ub[8];
    ub[0] = dimension_0();
    ub[1] = dimension_1();
    ub[2] = dimension_2();
    ub[3] = dimension_3();
    ub[4] = dimension_4();
    ub[5] = dimension_5();
    ub[6] = dimension_6();
    ub[7] = dimension_7();

    // Create shared memory tracking record with allocate memory from the memory
    // space
    record_type* const record = record_type::allocate(
        static_cast<Kokkos::Impl::ViewCtorProp<void, memory_space> const&>(
            arg_prop)
            .value,
        alloc_name, alloc_size, 
        Rank, data_layout, sizeof(value_type), ub);

    m_impl_handle = handle_type(reinterpret_cast<pointer_type>(record->data()));

    return record;
  
  }
  



};
} // Impl
} // Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_STAGINGSPACE_VIEW_MAPPING_HPP */