#ifndef KOKKOS_STAGINGSPACE_HPP
#define KOKKOS_STAGINGSPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

#include <dspaces.h>
#include <mpi.h>


/*--------------------------------------------------------------------------*/

namespace Kokkos {

struct StagingSpaceSpecializeTag {};
/** \brief  DataSpaces Staging memory management */
class StagingSpace {
  public:
  //! Tag this class as a kokkos memory space
  using memory_space    = StagingSpace;
  using size_type       = size_t;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP)
  using execution_space = Kokkos::OpenMP;
#elif defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS)
  using execution_space = Kokkos::Threads;
//#elif defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined(KOKKOS_ENABLE_OPENMP)
  using execution_space = Kokkos::OpenMP;
#elif defined(KOKKOS_ENABLE_THREADS)
  using execution_space = Kokkos::Threads;
//#elif defined( KOKKOS_ENABLE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined(KOKKOS_ENABLE_SERIAL)
  using execution_space = Kokkos::Serial;
#else
#  error "At least one of the following host execution spaces must be defined: Kokkos::OpenMP, Kokkos::Threads, Kokkos::Qthreads, or Kokkos::Serial.  You might be seeing this message if you disabled the Kokkos::Serial device explicitly using the Kokkos_ENABLE_Serial:BOOL=OFF CMake option, but did not enable any of the other host execution space devices."
#endif

  //! This memory space preferred device_type
  using device_type   = Kokkos::Device<execution_space, memory_space>;

  /*--------------------------------*/

  enum data_layout {LAYOUT_LEFT = 0,
                    LAYOUT_RIGHT = 1 };

  /**\brief  Default memory space instance */
  StagingSpace();
  StagingSpace(StagingSpace&& rhs); // need to copy std::vector in class 
  StagingSpace(const StagingSpace& rhs); // need to copy std::vector in class 
  StagingSpace& operator=(StagingSpace&& rhs); // need to copy std::vector in class
  StagingSpace& operator=(const StagingSpace &rhs); // need to copy std::vector in class
  ~StagingSpace() = default;

  /**\brief  Allocate untracked memory in the space */
  void* allocate(const size_t arg_alloc_size, const std::string& path_,
                  const size_t rank_,
                  const enum data_layout layout,
                  const size_t elem_size_,
                  const size_t* ub);

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void * const arg_alloc_ptr, const size_t arg_alloc_size) const;

  static void initialize();
  static void finalize();

  size_t write_data(const void * src, const size_t src_size);

  size_t read_data(void * dst, const size_t dst_size);

  static void initialize();
  static void finalize();

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  static void restore_all_views();
  static void restore_view(const std::string name);
  static void checkpoint_views();
  static void checkpoint_create_view_targets();

  static void set_default_path( const std::string path );

  void set_lb(const size_t lb_N0, const size_t lb_N1,
              const size_t lb_N2, const size_t lb_N3,
              const size_t lb_N4, const size_t lb_N5,
              const size_t lb_N6, const size_t lb_N7);

  void set_ub(const size_t ub_N0, const size_t ub_N1,
              const size_t ub_N2, const size_t ub_N3,
              const size_t ub_N4, const size_t ub_N5,
              const size_t ub_N6, const size_t ub_N7);

  void set_version(const size_t ver);

  static std::string s_default_path;

  //static dspaces_client_t ndcl;

  //static std::map<const std::string, KokkosDataspacesAccessor> m_accessor_map;

private:
  
  std::string get_timestep(std::string path, size_t &ts);
  void index_reverse();

  size_t rank;            // rank of the dataset (number of dimensions)
  size_t version;         // version of the dataset
  int appid;              // dataspaces client handle
  size_t elem_size;       // size of single element size, e.g. sizeof(double)
  uint64_t* lb;         // coordinates for the lower corner of the local bounding box.
  uint64_t* ub;         // coordinates for the upper corner of the local bounding box.
  //int mpi_size;
  //int mpi_rank;
  MPI_Comm gcomm;

  size_t data_size;
  std::string var_name;
  bool is_contiguous;

  enum ds_layout_type m_layout;
  int m_timeout;

  static dspaces_client_t ndcl;
  static constexpr const char* m_name = "Staging";
  bool m_is_initialized;
  friend class Kokkos::Impl::SharedAllocationRecord< Kokkos::StagingSpace, void>;

};
} // namespace Kokkos


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <>
class SharedAllocationRecord<Kokkos::StagingSpace, void>
    : public SharedAllocationRecord<void, void>
{
private:
  friend Kokkos::StagingSpace;

  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase*);

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this HDF5Space instance */
  static RecordBase s_root_record;
#endif

  
  

protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;
  SharedAllocationRecord(const Kokkos::StagingSpace& arg_space,
                          const std::string& arg_label, 
                          const size_t arg_alloc_size, 
                          const size_t rank,
                          const enum Kokkos::StagingSpace::data_layout layout,
                          const size_t elem_size, const size_t* ub,
                          const RecordBase::function_type arg_dealloc = &deallocate
                          );

  

public:

  const Kokkos::StagingSpace m_space;

  inline std::string get_label() const {
    return std::string(RecordBase::head()->m_label);
  }

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* 
  allocate(const Kokkos::StagingSpace& arg_space, 
          const std::string& arg_label, 
          const size_t arg_alloc_size,
          const size_t rank,
          const enum Kokkos::StagingSpace::data_layout layout,
          const size_t elem_size, const size_t* ub) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size,
                                      rank, layout, elem_size, ub);
#else
    return (SharedAllocationRecord *)0;
#endif
  }

  /**\brief  Allocate tracked memory in the space */
  static void* allocate_tracked(const Kokkos::StagingSpace& arg_space, 
                                const std::string & arg_label, 
                                const size_t arg_alloc_size);

  /**\brief  Reallocate tracked memory in the space */
  static void* reallocate_tracked(void* const arg_alloc_ptr, 
                                  const size_t arg_alloc_size);

  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void* const arg_alloc_ptr);

  static SharedAllocationRecord* get_record(void* arg_alloc_ptr);

  static void print_records(std::ostream &, const Kokkos::StagingSpace& , bool detail = false);

};
} // Impl
} // Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class ExecutionSpace>
struct DeepCopy<Kokkos::StagingSpace, Kokkos::HostSpace, ExecutionSpace> {
  inline DeepCopy(SharedAllocationRecord<Kokkos::StagingSpace,
                           void>* dst_record, const void* src, size_t n) {
    //SharedAllocationRecord<Kokkos::StagingSpace,
    //                       void>* const Record = SharedAllocationRecord<Kokkos::StagingSpace,
    //                                                               void>::get_record(dst);
    const_cast<Kokkos::StagingSpace&> (dst_record->m_space).write_data(src, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void * src, size_t n) {
    exec.fence();
    SharedAllocationRecord<Kokkos::StagingSpace,
                           void>* const Record = SharedAllocationRecord<Kokkos::StagingSpace,
                                                                   void>::get_record(dst);
    const_cast<Kokkos::StagingSpace&> (Record->m_space).write_data(src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::StagingSpace, ExecutionSpace> {
  inline DeepCopy(void* dst, SharedAllocationRecord<Kokkos::StagingSpace,
                           void>* src_record, size_t n) {
    //SharedAllocationRecord<Kokkos::StagingSpace,
    //                       void>* const Record = SharedAllocationRecord<Kokkos::StagingSpace,
    //                                                               void>::get_record(const_cast<void*> (src));
    const_cast<Kokkos::StagingSpace&> (src_record->m_space).read_data(dst, n);
  }

  inline DeepCopy(const ExecutionSpace& exec, void* dst, const void * src, size_t n) {
    exec.fence();
    SharedAllocationRecord<Kokkos::StagingSpace,
                           void>* const Record = SharedAllocationRecord<Kokkos::StagingSpace,
                                                                   void>::get_record(const_cast<void*> (src));
    const_cast<Kokkos::StagingSpace&> (Record->m_space).read_data(dst, n);
  }
};

} // Impl
} // Kokkos

#include <Kokkos_StagingSpace_SharedAlloc.hpp>
#include <Kokkos_StagingSpace_ViewMapping.hpp>
#include <Kokkos_StagingSpace_CopyViews.hpp>

#endif //KOKKOS_STAGINGSPACE_HPP