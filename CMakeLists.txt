cmake_minimum_required(VERSION 3.14)
project(KokkosStaging 
        LANGUAGES CXX 
        VERSION 0.1.0)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

find_package(Kokkos REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(MPI REQUIRED)
find_package(DataSpaces REQUIRED)

option(Kokkos_ENABLE_TESTS   "Whether to enable tests" OFF)

set(KOKKOS_STAGING_SRCS)
file(GLOB SRCS src/staging/*.cpp)
list(APPEND KOKKOS_STAGING_SRCS ${SRCS})
set(KOKKOS_STAGING_HEADERS)
file(GLOB HDRS src/staging/*.hpp)
list(APPEND KOKKOS_STAGING_HEADERS ${HDRS} )

add_library(staging ${KOKKOS_STAGING_SRCS} ${KOKKOS_STAGING_HEADERS})
add_library(Kokkos::staging ALIAS staging)

# Require C++14
target_compile_features(staging PUBLIC cxx_std_14)

target_include_directories(staging PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/staging>)
target_include_directories(staging PUBLIC $<INSTALL_INTERFACE:include>)

target_link_libraries(staging PUBLIC Kokkos::kokkos)
target_link_libraries(staging PUBLIC MPI::MPI_CXX)
target_link_libraries(staging PRIVATE DataSpaces::DataSpaces)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/KokkosStagingConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/KokkosStagingConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake
)
write_basic_package_version_file( 
  "${CMAKE_CURRENT_BINARY_DIR}/KokkosStagingConfigVersion.cmake"
  VERSION "${KokkosStaging_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/KokkosStagingConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/KokkosStagingConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosStaging)

install(FILES
  ${KOKKOS_STAGING_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  TARGETS staging
  EXPORT KokkosStagingTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT 
  KokkosStagingTargets
  NAMESPACE Kokkos:: 
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosStaging
)

IF (Kokkos_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
ENDIF()