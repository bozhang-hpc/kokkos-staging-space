#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_StagingSpace.hpp>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    Kokkos::initialize(argc, argv);

    Kokkos::StagingSpace::initialize();

    ::testing::InitGoogleTest(&argc, argv);

    int ret = RUN_ALL_TESTS();

    Kokkos::StagingSpace::finalize();

    Kokkos::finalize();

    MPI_Finalize();
    return ret;
}