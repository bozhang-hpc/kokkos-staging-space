#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_StagingSpace.hpp>

#include <string>
#include <regex>
#include <iostream>
#include <sstream>
#include <limits>
#include <unistd.h>


namespace Kokkos {

dspaces_client_t StagingSpace::ndcl = dspaces_CLIENT_NULL;

std::string StagingSpace::get_timestep(std::string path, size_t& ts) {
  std::smatch result;
  std::regex pattern("/[Tt](\\d+)/?");
  if(std::regex_search(path, result, pattern)) {
    ts = std::strtoull(result[1].str().c_str(), NULL, 0);
      return result.prefix().str();
  }
  return "";
}

void StagingSpace::index_reverse() {
  for(int i=0; i<rank/2; i++) {
    uint64_t temp_lb = lb[i];
    lb[i] = lb[rank-1-i];
    lb[rank-1-i] = temp_lb;
    uint64_t temp_ub = ub[i];
    ub[i] = ub[rank-1-i];
    ub[rank-1-i] = temp_ub;
  }
}

StagingSpace::StagingSpace(): rank(1),
                              version(0),
                              elem_size(1),
                              //lb{0,0,0,0,0,0,0,0},
                              //ub{0,0,0,0,0,0,0,0},
                              //mpi_size(1),
                              //mpi_rank(0),
                              gcomm(MPI_COMM_WORLD),
                              m_timeout(-1),
                              m_is_initialized(false) { }

StagingSpace::StagingSpace(StagingSpace&& rhs) {
  rank = rhs.rank;
  version = rhs.version;
  //appid = rhs.appid;
  elem_size = rhs.elem_size;
  // DeepCopy
  lb = (uint64_t*) malloc(rank*sizeof(uint64_t));
  ub = (uint64_t*) malloc(rank*sizeof(uint64_t));
  memcpy(lb, rhs.lb, rank*sizeof(uint64_t));
  memcpy(ub, rhs.ub, rank*sizeof(uint64_t));
  gcomm = rhs.gcomm;
  data_size = rhs.data_size;
  var_name = rhs.var_name;
  is_contiguous = rhs.is_contiguous;
  m_layout = rhs.m_layout;
  m_timeout = rhs.m_timeout;
  m_is_initialized = rhs.m_is_initialized;
  //m_layout_space_map = rhs.m_layout_space_map;
}

StagingSpace::StagingSpace(const StagingSpace& rhs) {
  rank = rhs.rank;
  version = rhs.version;
  //appid = rhs.appid;
  elem_size = rhs.elem_size;
  // DeepCopy
  lb = (uint64_t*) malloc(rank*sizeof(uint64_t));
  ub = (uint64_t*) malloc(rank*sizeof(uint64_t));
  memcpy(lb, rhs.lb, rank*sizeof(uint64_t));
  memcpy(ub, rhs.ub, rank*sizeof(uint64_t));
  gcomm = rhs.gcomm;
  data_size = rhs.data_size;
  var_name = rhs.var_name;
  is_contiguous = rhs.is_contiguous;
  m_layout = rhs.m_layout;
  m_timeout = rhs.m_timeout;
  m_is_initialized = rhs.m_is_initialized;
  //m_layout_space_map = rhs.m_layout_space_map;
}

StagingSpace& StagingSpace::operator=(StagingSpace&& rhs) {
  rank = rhs.rank;
  version = rhs.version;
  //appid = rhs.appid;
  elem_size = rhs.elem_size;
  // DeepCopy
  lb = (uint64_t*) malloc(rank*sizeof(uint64_t));
  ub = (uint64_t*) malloc(rank*sizeof(uint64_t));
  memcpy(lb, rhs.lb, rank*sizeof(uint64_t));
  memcpy(ub, rhs.ub, rank*sizeof(uint64_t));
  gcomm = rhs.gcomm;
  data_size = rhs.data_size;
  var_name = rhs.var_name;
  is_contiguous = rhs.is_contiguous;
  m_layout = rhs.m_layout;
  m_timeout = rhs.m_timeout;
  m_is_initialized = rhs.m_is_initialized;
  //m_layout_space_map = rhs.m_layout_space_map;
  return *this;
}

StagingSpace& StagingSpace::operator=(const StagingSpace &rhs) {
  rank = rhs.rank;
  version = rhs.version;
  //appid = rhs.appid;
  elem_size = rhs.elem_size;
  // DeepCopy
  lb = (uint64_t*) malloc(rank*sizeof(uint64_t));
  ub = (uint64_t*) malloc(rank*sizeof(uint64_t));
  memcpy(lb, rhs.lb, rank*sizeof(uint64_t));
  memcpy(ub, rhs.ub, rank*sizeof(uint64_t));
  gcomm = rhs.gcomm;
  data_size = rhs.data_size;
  var_name = rhs.var_name;
  is_contiguous = rhs.is_contiguous;
  m_layout = rhs.m_layout;
  m_timeout = rhs.m_timeout;
  m_is_initialized = rhs.m_is_initialized;
  //m_layout_space_map = rhs.m_layout_space_map;
  return *this;
}

void StagingSpace::initialize() {
  int mpi_rank, mpi_size;
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
  MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank);
  ndcl = dspaces_CLIENT_NULL;
  dspaces_init(mpi_rank, &ndcl);
}

void StagingSpace::finalize() {
  dspaces_fini(ndcl);
}

void* StagingSpace::allocate(const size_t arg_alloc_size, const std::string& path_,
                              const size_t rank_, 
                              const enum data_layout layout,
                              const size_t elem_size_,
                              const size_t* ub_) {
    if(!m_is_initialized) {
      std::string path_prefix = get_timestep(path_, version);
      if (path_prefix.compare("") == 0) 
        var_name = path_;
      else
        var_name = path_prefix;

      data_size = arg_alloc_size;
      rank = rank_;
      elem_size = elem_size_;

      lb = (uint64_t*) malloc(rank_*sizeof(uint64_t));
      ub = (uint64_t*) malloc(rank_*sizeof(uint64_t));

      //lb.resize(rank_);
      //ub.resize(rank_);

      for(int i=0; i<rank; i++) {
        lb[i] = 0;
        ub[i] = ub_[i]-1;
      }


      switch (layout)
      {
      case LAYOUT_LEFT:
        m_layout = dspaces_LAYOUT_LEFT;
        break;

      case LAYOUT_RIGHT:
        m_layout = dspaces_LAYOUT_RIGHT;
        index_reverse();
        break;

      default:
        m_layout = dspaces_LAYOUT_RIGHT;
        index_reverse();
        break;
      }

      //m_layout_space_map[m_layout] = *this;

      /*
      std::string pid_str = std::to_string((int)getpid());
      std::hash<std::string> str_hash;
      appid = str_hash(pid_str) % std::numeric_limits<int>::max();
      MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
      MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank);
      MPI_Bcast(&appid, 1, MPI_INT, 0, gcomm);
      MPI_Barrier(gcomm);
      dspaces_init(mpi_size, appid, &gcomm, NULL); 
      */
      m_is_initialized = true;
    }

    return reinterpret_cast<void*>(this);
}

void StagingSpace::deallocate(void * const arg_alloc_ptr
                                    , const size_t arg_alloc_size ) const {
  free(lb);
  free(ub);
}



size_t StagingSpace::write_data(const void* src, const size_t src_size){
  size_t m_written = 0;
  //dspaces_lock_on_write(var_name.c_str(), &gcomm);
  int err = dspaces_put_layout(ndcl, var_name.c_str(), version, elem_size, rank, lb, ub, m_layout, src);
  //dspaces_unlock_on_write(var_name.c_str(), &gcomm);
  if(err == 0) {
    m_written = src_size;
  } else {
    printf("Dataspaces: write failed \n");
  }
  return m_written;
}

size_t StagingSpace::read_data(void * dst, const size_t dst_size) {
  size_t dataRead = 0;
  //dspaces_lock_on_read(var_name.c_str(), &gcomm);
  int err = dspaces_get_layout(ndcl, var_name.c_str(), version, elem_size, rank, lb, ub, m_layout, dst, m_timeout);
  //dspaces_unlock_on_read(var_name.c_str(), &gcomm);
  if(err == 0) {
    dataRead = dst_size; 
  } else {
    printf("Error with read: %d \n", err);
  }
  return dataRead;
}

void StagingSpace::set_lb(const size_t lb_N0, const size_t lb_N1,
                          const size_t lb_N2, const size_t lb_N3,
                          const size_t lb_N4, const size_t lb_N5,
                          const size_t lb_N6, const size_t lb_N7) {
  lb[0] = lb_N0;
  lb[1] = lb_N1;
  lb[2] = lb_N2;
  lb[3] = lb_N3;
  lb[4] = lb_N4;
  lb[5] = lb_N5;
  lb[6] = lb_N6;
  lb[7] = lb_N7;

}

void StagingSpace::set_ub(const size_t ub_N0, const size_t ub_N1,
                          const size_t ub_N2, const size_t ub_N3,
                          const size_t ub_N4, const size_t ub_N5,
                          const size_t ub_N6, const size_t ub_N7) {
  ub[0] = ub_N0;
  ub[1] = ub_N1;
  ub[2] = ub_N2;
  ub[3] = ub_N3;
  ub[4] = ub_N4;
  ub[5] = ub_N5;
  ub[6] = ub_N6;
  ub[7] = ub_N7;
  
}

void StagingSpace::set_version(const size_t ver) {
  version = ver;
}

void StagingSpace::set_var_name(const std::string var_name_) {
  var_name = var_name_;
}

} // Kokkos

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord< void , void >
  SharedAllocationRecord< Kokkos::StagingSpace , void >::s_root_record ;
#endif

void
SharedAllocationRecord< Kokkos::StagingSpace , void >::
deallocate(SharedAllocationRecord< void , void > * arg_rec)
{
    delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::StagingSpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
    if(Kokkos::Profiling::profileLibraryLoaded()) {
        Kokkos::Profiling::deallocateData(
            Kokkos::Profiling::SpaceHandle(Kokkos::StagingSpace::name()),RecordBase::m_alloc_ptr->m_label,
            data(),size());
    }
    #endif

  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                      , SharedAllocationRecord< void , void >::m_alloc_size
                      );
}

SharedAllocationRecord< Kokkos::StagingSpace , void >::
SharedAllocationRecord( const Kokkos::StagingSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t arg_alloc_size
                      , const size_t rank
                      , const enum Kokkos::StagingSpace::data_layout layout
                      , const size_t elem_size
                      , const size_t* ub
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                       )
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord< void , void >
        (
#ifdef KOKKOS_DEBUG
        & SharedAllocationRecord< Kokkos::StagingSpace , void >::s_root_record,
#endif
          reinterpret_cast<SharedAllocationHeader*>( const_cast<Kokkos::StagingSpace&> 
                                                      (arg_space).allocate(arg_alloc_size, arg_label,
                                                                            rank, layout, elem_size,
                                                                            ub ) )
        , arg_alloc_size
        , arg_dealloc
        )
    , m_space( arg_space )
{
#if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
   }
#endif
    // Fill in the Header information
    RecordBase::m_alloc_ptr->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

    strncpy( RecordBase::m_alloc_ptr->m_label
            , arg_label.c_str()
            , SharedAllocationHeader::maximum_label_length
            );
    // Set last element zero, in case c_str is too long
    RecordBase::m_alloc_ptr->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char) 0;

}

//----------------------------------------------------------------------------

SharedAllocationRecord< Kokkos::StagingSpace , void > *
SharedAllocationRecord< Kokkos::StagingSpace , void >::get_record( void * alloc_ptr )
{
  typedef SharedAllocationHeader  Header ;
  typedef SharedAllocationRecord< Kokkos::StagingSpace , void >  RecordHost ;

  SharedAllocationHeader const * const head   = alloc_ptr ? Header::get_header( alloc_ptr ) : (SharedAllocationHeader *)0 ;
  RecordHost                   * const record = head ? static_cast< RecordHost * >( head->m_record ) : (RecordHost *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::StagingSpace , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord< Kokkos::StagingSpace , void >::
print_records( std::ostream & s , const Kokkos::StagingSpace & , bool detail )
{
#ifdef KOKKOS_DEBUG
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "StagingSpace" , & s_root_record , detail );
#else
  throw_runtime_exception("SharedAllocationRecord<Kokkos::StagingSpace>::print_records only works with KOKKOS_DEBUG enabled");
#endif
}

} // Impl
} // Kokkos