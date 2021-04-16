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
inline void set_version(const View<DT, DP...>& dst, size_t version,
                        typename std::enable_if<
                        std::is_same<typename ViewTraits<DT, DP...>::specialize, 
                        Kokkos::StagingSpaceSpecializeTag>::value>::type* = nullptr) {
    using dst_type          = View<DT, DP...>;
    using dst_memory_space  = typename dst_type::memory_space;

    Kokkos::Impl::SharedAllocationRecord<dst_memory_space, void>* 
                                  dst_record = dst.impl_track().template get_record<dst_memory_space>();

    const_cast<dst_memory_space&> (dst_record->m_space).set_version(version);

}

template <class DT, class... DP, class ST, class... SP>
inline void view_bind_layout(const View<DT, DP...>& dst, const View<ST, SP...>& src,
                        typename std::enable_if<(
                        std::is_same<typename ViewTraits<DT, DP...>::specialize, 
                        Kokkos::StagingSpaceSpecializeTag>::value &&
                        std::is_same<typename ViewTraits<ST, SP...>::specialize,
                        Kokkos::StagingSpaceSpecializeTag>::value)>::type* = nullptr) {
    using dst_type            = View<DT, DP...>;
    using src_type            = View<ST, SP...>;
    using dst_memory_space    = typename dst_type::memory_space;
    using src_memory_space    = typename src_type::memory_space;
    using dst_value_type      = typename dst_type::value_type;
    using src_value_type      = typename src_type::value_type;

    static_assert((unsigned(dst_type::rank) == unsigned(src_type::rank)),
                "view_bind_layout requires Views of equal rank");

    static_assert((std::is_same<typename dst_type::value_type, typename src_type::value_type>::value),
                "view_bind_layout requires Views of same value_type");

    static_assert((!std::is_same<typename dst_type::array_layout, typename src_type::array_layout>::value),
                "view_bind_layout requires Views of different array_layout");

    // throw if dimension mismatch
    if ((src.extent(0) != dst.extent(0)) || (src.extent(1) != dst.extent(1)) ||
        (src.extent(2) != dst.extent(2)) || (src.extent(3) != dst.extent(3)) ||
        (src.extent(4) != dst.extent(4)) || (src.extent(5) != dst.extent(5)) ||
        (src.extent(6) != dst.extent(6)) || (src.extent(7) != dst.extent(7))) {
      std::string message (
          "Deprecation Error: Kokkos::view_bind_layout extents of views don't "
          "match: ");
      message += dst.label();
      message += "(";
      for (int r = 0; r < dst_type::Rank - 1; r++) {
        message += std::to_string(dst.extent(r));
        message += ",";
      }
      message += std::to_string(dst.extent(dst_type::Rank - 1));
      message += ") ";
      message += src.label();
      message += "(";
      for (int r = 0; r < src_type::Rank - 1; r++) {
        message += std::to_string(src.extent(r));
        message += ",";
      }
      message += std::to_string(src.extent(src_type::Rank - 1));
      message += ") ";

      Kokkos::Impl::throw_runtime_exception(message);
    }

    Kokkos::Impl::SharedAllocationRecord<dst_memory_space, void>* 
                                  dst_record = dst.impl_track().template get_record<dst_memory_space>();

    Kokkos::Impl::SharedAllocationRecord<src_memory_space, void>* 
                                  src_record = src.impl_track().template get_record<dst_memory_space>();

    const_cast<Kokkos::StagingSpace&> (dst_record->m_space).set_var_name(
        const_cast<Kokkos::StagingSpace&> (dst_record->m_space).get_var_name());

}

} //namespace Staging
} //namespace Kokkos 