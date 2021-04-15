#include <Kokkos_Macros.hpp>
#include <Kokkos_StagingSpace.hpp>

namespace Kokkos {
namespace Staging {

inline void initialize() {
    Kokkos::StagingSpace::initialize();
}

inline void finalize() {
    Kokkos::StagingSpace::finalize();
}

template <class DT, class... DP>
inline void set_lower_bound(
    const View<DT, DP...>& dst, 
    const size_t lb_N0,
    const size_t lb_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t lb_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t lb_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t lb_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t lb_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t lb_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t lb_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    typename std::enable_if<(
        std::is_same<typename ViewTraits<DT, DP...>::specialize, 
        Kokkos::StagingSpaceSpecializeTag>::value &&
        unsigned(ViewTraits<DT, DP...>::rank) != 0)>::type* = nullptr) {

    using dst_type          = View<DT, DP...>;
    using dst_memory_space  = typename dst_type::memory_space;

    unsigned int rank_check = (lb_N1 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 1 : 
                     (lb_N2 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 2 :
                     (lb_N3 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 3 :
                     (lb_N4 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 4 :
                     (lb_N5 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 5 :
                     (lb_N6 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 6 :
                     (lb_N7 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 7 : 8

    static_assert((unsigned(dst_type::rank) == rank_check),
                    "set_lower_bound requires same rank as View has");


    Kokkos::Impl::SharedAllocationRecord<dst_memory_space, void>* 
                                  dst_record = dst.impl_track().template get_record<dst_memory_space>();

    const_cast<dst_memory_space&> (src_record->m_space).set_lb(lb_N0, lb_N1, lb_N2, lb_N3,
                                                                    lb_N4, lb_N5, lb_N6, lb_N7);

}

template <class DT, class... DP>
inline void set_upper_bound(
    const View<DT, DP...>& dst, 
    const size_t ub_N0,
    const size_t ub_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t ub_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t ub_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t ub_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t ub_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t ub_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    const size_t ub_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
    typename std::enable_if<(
        std::is_same<typename ViewTraits<DT, DP...>::specialize, 
        Kokkos::StagingSpaceSpecializeTag>::value &&
        unsigned(ViewTraits<DT, DP...>::rank) != 0)>::type* = nullptr) {

    using dst_type          = View<DT, DP...>;
    using dst_memory_space  = typename dst_type::memory_space;

    unsigned int rank_check = (lb_N1 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 1 : 
                     (lb_N2 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 2 :
                     (lb_N3 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 3 :
                     (lb_N4 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 4 :
                     (lb_N5 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 5 :
                     (lb_N6 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 6 :
                     (lb_N7 == KOKKOS_IMPL_CTOR_DEFAULT_ARG) ? 7 : 8

    static_assert((unsigned(dst_type::rank) == rank_check),
                    "set_upper_bound requires same rank as View has");

    Kokkos::Impl::SharedAllocationRecord<dst_memory_space, void>* 
                                  dst_record = dst.impl_track().template get_record<dst_memory_space>();

    const_cast<dst_memory_space&> (src_record->m_space).set_ub(ub_N0, ub_N1, ub_N2, ub_N3,
                                                                    ub_N4, ub_N5, ub_N6, ub_N7);

}

template <class DT, class... DP>
inline void set_version(const View<DT, DP...>& dst, size_t version,
                        typename std::enable_if<
                        std::is_same<typename ViewTraits<DT, DP...>::specialize, 
                        Kokkos::StagingSpaceSpecializeTag>::value>::type* = nullptr) {
    using dst_type          = View<DT, DP...>;
    using dst_memory_space  = typename dst_type::memory_space;

    Kokkos::Impl::SharedAllocationRecord<dst_memory_space, void>* 
                                  dst_record = dst.impl_track().template get_record<dst_memory_space>();

    const_cast<dst_memory_space&> (src_record->m_space).set_version(version);

}



inline void transform_layout()

} //namespace Staging
} //namespace Kokkos