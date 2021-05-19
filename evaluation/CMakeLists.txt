add_executable(kokkos_test_writer test_writer.cpp)
target_link_libraries(kokkos_test_writer PUBLIC Kokkos::staging)

add_executable(kokkos_test_reader test_reader.cpp)
target_link_libraries(kokkos_test_reader PUBLIC Kokkos::staging)

add_executable(kokkos_test_subset_reader test_subset_reader.cpp)
target_link_libraries(kokkos_test_subset_reader PUBLIC Kokkos::staging)