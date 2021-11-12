#ifndef TEST_WRITER_HPP
#define TEST_WRITER_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_StagingSpace.hpp>
#include <iostream>
#include "timer.hpp"
#include "unistd.h"
#include "mpi.h"
// only support 1 var_num now.
template <class Data_t, unsigned int Dims, class Layout>
struct kokkos_run {
    static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                        int delay, std::string log_name, bool terminate);
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 1, Layout> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    int delay, std::string log_name, bool terminate)
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

    ViewHost_t v_P("PutView", sp[0]);
    ViewStaging_t v_S("StagingView_1D", sp[0]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            v_P(i0) = i0 + 0.01*ts;         
        });

        // 1s sleep as the lightest computation overhead
        sleep(1);

        sleep(delay);

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 2, Layout> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    int delay, std::string log_name, bool terminate)
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

    ViewHost_t v_P("PutView", sp[0], sp[1]);
    ViewStaging_t v_S("StagingView_2D", sp[0], sp[1]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                v_P(i0,i1) = i0*sp[1]+i1 + 0.01*ts;
            }
        });

        sleep(delay);

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 3, Layout> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    int delay, std::string log_name, bool terminate)
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

    ViewHost_t v_P("PutView", sp[0], sp[1], sp[2]);
    ViewStaging_t v_S("StagingView_3D", sp[0], sp[1], sp[2]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                for(int i2=0; i2<sp[2]; i2++) {
                    v_P(i0,i1,i2) = (i0*sp[1]+i1)*sp[2]+i2 + 0.01*ts;
                    //std::cout<<v_P(i0,i1,i2)<<"\t";
                }
                //std::cout<<std::endl;
            }
            //std::cout<<"******************"<<std::endl;
        });

        sleep(delay);

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 4, Layout> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    int delay, std::string log_name, bool terminate)
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

    ViewHost_t v_P("PutView", sp[0], sp[1], sp[2], sp[3]);
    ViewStaging_t v_S("StagingView_4D", sp[0], sp[1], sp[2], sp[3]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], lb[3]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                for(int i2=0; i2<sp[2]; i2++) {
                    for(int i3=0; i3<sp[3]; i3++) {
                        v_P(i0,i1,i2,i3) = ((i0*sp[1]+i1)*sp[2]+i2)*sp[3]+i3 + 0.01*ts;
                    }
                }
            }
        });

        sleep(delay);

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 5, Layout> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    int delay, std::string log_name, bool terminate)
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

    ViewHost_t v_P("PutView", sp[0], sp[1], sp[2], sp[3], sp[4]);
    ViewStaging_t v_S("StagingView_5D", sp[0], sp[1], sp[2], sp[3],sp[4]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3], sp[4]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], lb[3], sp[4]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                for(int i2=0; i2<sp[2]; i2++) {
                    for(int i3=0; i3<sp[3]; i3++) {
                        for(int i4=0; i4<sp[4]; i4++) {
                            v_P(i0,i1,i2,i3,i4) = (((i0*sp[1]+i1)*sp[2]+i2)*sp[3]+i3)*sp[4]+i4 + 0.01*ts;
                        }
                    }
                }
            }
        });

        sleep(delay);

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 6, Layout> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    int delay, std::string log_name, bool terminate)
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

    ViewHost_t v_P("PutView", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5]);
    ViewStaging_t v_S("StagingView_6D", sp[0], sp[1], sp[2], sp[3],sp[4], sp[5]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3], sp[4], sp[5]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], lb[3], sp[4], sp[5]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                for(int i2=0; i2<sp[2]; i2++) {
                    for(int i3=0; i3<sp[3]; i3++) {
                        for(int i4=0; i4<sp[4]; i4++) {
                            for(int i5=0; i5<sp[5]; i5++) {
                                v_P(i0,i1,i2,i3,i4,i5) = ((((i0*sp[1]+i1)*sp[2]+i2)*sp[3]+i3)*sp[4]+i4)
                                                            *sp[5]+i5 + 0.01*ts;
                            }
                        }
                    }
                }
            }
        });

        sleep(delay);

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 7, Layout> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    int delay, std::string log_name, bool terminate)
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

    ViewHost_t v_P("PutView", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5], sp[6]);
    ViewStaging_t v_S("StagingView_7D", sp[0], sp[1], sp[2], sp[3],sp[4], sp[5], sp[6]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3], sp[4], sp[5], sp[6]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], lb[3], sp[4], sp[5], sp[6]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                for(int i2=0; i2<sp[2]; i2++) {
                    for(int i3=0; i3<sp[3]; i3++) {
                        for(int i4=0; i4<sp[4]; i4++) {
                            for(int i5=0; i5<sp[5]; i5++) {
                                for(int i6=0; i6<sp[6]; i6++) {
                                    v_P(i0,i1,i2,i3,i4,i5,i6) = (((((i0*sp[1]+i1)*sp[2]+i2)*sp[3]+i3)*sp[4]
                                                                +i4)*sp[5]+i5)*sp[6]+i6 + 0.01*ts;
                                }
                            }
                        }
                    }
                }
            }
        });

        sleep(delay);

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

template <class Data_t, class Layout>
struct kokkos_run<Data_t, 8, Layout> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num,
                    int delay, std::string log_name, bool terminate)
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

    ViewHost_t v_P("PutView", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5], sp[6], sp[7]);
    ViewStaging_t v_S("StagingView_7D", sp[0], sp[1], sp[2], sp[3],sp[4], sp[5], sp[6], sp[7]);

    Kokkos::Staging::set_lower_bound(v_S, lb[0], lb[1], lb[2], lb[3], sp[4], sp[5], sp[6], sp[7]);
    Kokkos::Staging::set_upper_bound(v_S, ub[0], ub[1], ub[2], lb[3], sp[4], sp[5], sp[6], sp[7]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open(log_name, std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                for(int i2=0; i2<sp[2]; i2++) {
                    for(int i3=0; i3<sp[3]; i3++) {
                        for(int i4=0; i4<sp[4]; i4++) {
                            for(int i5=0; i5<sp[5]; i5++) {
                                for(int i6=0; i6<sp[6]; i6++) {
                                    for(int i7=0; i7<sp[7]; i7++) {
                                        v_P(i0,i1,i2,i3,i4,i5,i6,i7) = ((((((i0*sp[1]+i1)*sp[2]+i2)*sp[3]
                                                                        +i3)*sp[4]+i4)*sp[5]+i5)*sp[6]+i6)
                                                                        *sp[7]+i7 + 0.01*ts;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        sleep(delay);

        Kokkos::Staging::set_version(v_S, ts);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
            Kokkos::Staging::terminate();
        }
    }

    Kokkos::Staging::finalize();

    return 0;
};
};

#endif // TEST_WRITER_HPP