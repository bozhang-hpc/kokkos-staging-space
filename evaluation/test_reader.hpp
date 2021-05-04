#ifndef TEST_READER_HPP
#define TEST_READER_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_StagingSpace.hpp>
#include <iostream>
#include <string>
#include "timer.hpp"
#include "mpi.h"
// only support 1 var_num now.
template <class Data_t, unsigned int Dims, class Layout>
struct kokkos_run {
    static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                        std::string log_name, bool terminate);
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 1, Layout> {
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    std::string log_name, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    Kokkos::Staging::initialize();

    using ViewHost_t    = Kokkos::View<Data_t*, Layout, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t*, Layout, Kokkos::StagingSpace>;

    uint64_t* off = (uint64_t*) malloc(1*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(1*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(1*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<1; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_G("GetView", sp[0]);
    ViewStaging_t v_S("StagingView_1D", sp[0]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0]);

    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_read;

        timer_read.start();

        Kokkos::deep_copy(v_G, v_S);

        double time_read = timer_read.stop();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_read);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 2, Layout> {
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    std::string log_name, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    Kokkos::Staging::initialize();

    using ViewHost_t    = Kokkos::View<Data_t**, Layout, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t**, Layout, Kokkos::StagingSpace>;

    uint64_t* off = (uint64_t*) malloc(2*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(2*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(2*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<2; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_G("GetView", sp[0], sp[1]);
    ViewStaging_t v_S("StagingView_2D", sp[0], sp[1]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1]);

    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_read;

        timer_read.start();

        Kokkos::deep_copy(v_G, v_S);

        double time_read = timer_read.stop();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_read);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 3, Layout> {
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    std::string log_name, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    Kokkos::Staging::initialize();

    using ViewHost_t    = Kokkos::View<Data_t***, Layout, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t***, Layout, Kokkos::StagingSpace>;

    uint64_t* off = (uint64_t*) malloc(3*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(3*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(3*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<3; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_G("GetView", sp[0], sp[1], sp[2]);
    ViewStaging_t v_S("StagingView_3D", sp[0], sp[1], sp[2]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2]);

    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_read;

        timer_read.start();

        Kokkos::deep_copy(v_G, v_S);

        double time_read = timer_read.stop();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                for(int i2=0; i2<sp[2]; i2++) {
                    std::cout<<v_G(i0,i1,i2)<<"\t";
                }
                std::cout<<std::endl;
            }
            std::cout<<"******************"<<std::endl;
        });
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_read);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 4, Layout> {
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    std::string log_name, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    Kokkos::Staging::initialize();

    using ViewHost_t    = Kokkos::View<Data_t****, Layout, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t****, Layout, Kokkos::StagingSpace>;

    uint64_t* off = (uint64_t*) malloc(4*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(4*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(4*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<4; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_G("GetView", sp[0], sp[1], sp[2], sp[3]);
    ViewStaging_t v_S("StagingView_4D", sp[0], sp[1], sp[2], sp[3]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], ub[3]);

    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_read;

        timer_read.start();

        Kokkos::deep_copy(v_G, v_S);

        double time_read = timer_read.stop();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_read);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 5, Layout> {
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    std::string log_name, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    Kokkos::Staging::initialize();

    using ViewHost_t    = Kokkos::View<Data_t*****, Layout, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t*****, Layout, Kokkos::StagingSpace>;

    uint64_t* off = (uint64_t*) malloc(5*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(5*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(5*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<5; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_G("GetView", sp[0], sp[1], sp[2], sp[3], sp[4]);
    ViewStaging_t v_S("StagingView_5D", sp[0], sp[1], sp[2], sp[3], sp[4]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3], lb[4]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], ub[3], ub[4]);

    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_read;

        timer_read.start();

        Kokkos::deep_copy(v_G, v_S);

        double time_read = timer_read.stop();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_read);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 6, Layout> {
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    std::string log_name, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    Kokkos::Staging::initialize();

    using ViewHost_t    = Kokkos::View<Data_t******, Layout, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t******, Layout, Kokkos::StagingSpace>;

    uint64_t* off = (uint64_t*) malloc(6*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(6*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(6*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<6; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_G("GetView", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5]);
    ViewStaging_t v_S("StagingView_6D", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3], lb[4], lb[5]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], ub[3], ub[4], ub[5]);

    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_read;

        timer_read.start();

        Kokkos::deep_copy(v_G, v_S);

        double time_read = timer_read.stop();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_read);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 7, Layout> {
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    std::string log_name, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    Kokkos::Staging::initialize();

    using ViewHost_t    = Kokkos::View<Data_t*******, Layout, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t*******, Layout, Kokkos::StagingSpace>;

    uint64_t* off = (uint64_t*) malloc(7*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(7*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(7*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<7; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_G("GetView", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5], sp[6]);
    ViewStaging_t v_S("StagingView_7D", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5], sp[6]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3], lb[4], lb[5], lb[6]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], ub[3], ub[4], ub[5], ub[6]);

    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_read;

        timer_read.start();

        Kokkos::deep_copy(v_G, v_S);

        double time_read = timer_read.stop();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_read);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 8, Layout> {
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    std::string log_name, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);

    Kokkos::Staging::initialize();

    using ViewHost_t    = Kokkos::View<Data_t********, Layout, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t********, Layout, Kokkos::StagingSpace>;

    uint64_t* off = (uint64_t*) malloc(8*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(8*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(8*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<8; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_G("GetView", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5], sp[6], sp[7]);
    ViewStaging_t v_S("StagingView_8D", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5], sp[6], sp[7]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3], lb[4], lb[5], lb[6], lb[7]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], ub[3], ub[4], ub[5], ub[6], ub[7]);

    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_read;

        timer_read.start();

        Kokkos::deep_copy(v_G, v_S);

        double time_read = timer_read.stop();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_read);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Reader sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

#endif // TEST_READER_HPP