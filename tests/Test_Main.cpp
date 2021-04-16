#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_StagingSpace.hpp>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    Kokkos::initialize(argc, argv);

    Kokkos::Staging::initialize();

    ::testing::InitGoogleTest(&argc, argv);

    int ret = RUN_ALL_TESTS();

    Kokkos::Staging::finalize();

    Kokkos::finalize();

    MPI_Finalize();
    return ret;
}