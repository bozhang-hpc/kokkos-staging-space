//#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_StagingSpace.hpp>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <typeinfo>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
    Kokkos::StagingSpace::initialize();

    using ViewHost_lr_t    = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;
    using ViewHost_ll_t    = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    using ViewStaging_lr_t = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::StagingSpace>;
    using ViewStaging_ll_t = Kokkos::View<double**, Kokkos::LayoutLeft,Kokkos::StagingSpace>;

    const int i1 = 8;
    const int i2 = 8;

    std::string v_s_label ="StagingView_2D_";
    
    v_s_label += "double_"+std::to_string(i1)+"_"+std::to_string(i2);

    ViewHost_lr_t v_P("PutView", i1, i2);
    ViewStaging_lr_t v_S(v_s_label, i1, i2);
    Kokkos::Staging::set_lower_bound(v_S, 1, 1);
    Kokkos::Staging::set_upper_bound(v_S, 8, 8);
    ViewStaging_ll_t v_S_ll("Whatever", i1, i2);
    Kokkos::Staging::set_lower_bound(v_S_ll, 1, 1);
    Kokkos::Staging::set_upper_bound(v_S_ll, 8, 8);
    ViewHost_ll_t v_G("GetView", i1, i2);

    Kokkos::Staging::view_bind_layout(v_S_ll, v_S);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++) {
            v_P(i1_,i2_) = i1_*i2+i2_;
            std::cout<<v_P(i1_, i2_)<<"\t";
        }
        std::cout<<std::endl;
    });

    Kokkos::deep_copy(v_S, v_P);

    Kokkos::deep_copy(v_G, v_S_ll);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++) {
            std::cout<<v_G(i1_, i2_)<<"\t";
        }
        std::cout<<std::endl;
    });

    sleep(5);
    Kokkos::StagingSpace::finalize();
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}

/*

template <class Data_t>
void test_deepcopy_layout(int i1, int i2)
{
    using ViewHost_t    = Kokkos::View<Data_t**, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t**, Kokkos::StagingSpace>;
    using ViewStaging_ll_t = Kokkos::View<Data_t**, Kokkos::LayoutLeft,Kokkos::StagingSpace>;

    std::string v_s_label ="StagingView_2D_";
    std::string type_name (typeid(Data_t).name());
    v_s_label += type_name+"_"+std::to_string(i1)+"_"+std::to_string(i2);

    ViewHost_t v_P("PutView", i1, i2);
    ViewStaging_t v_S(v_s_label, i1, i2);
    ViewStaging_ll_t v_S_ll(v_s_label, i1, i2);
    ViewHost_t v_G("GetView", i1, i2);

    Kokkos::Staging::view_bind_layout(v_S_ll, v_S);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++) 
            v_P(i1_,i2_) = i1_*i2+i2_;
    });

    Kokkos::deep_copy(v_S, v_P);
    
    Kokkos::deep_copy(v_G, v_S);
    
    for(int i1_=0; i1_<i1; i1_++) {
        for(int i2_=0; i2_<i2; i2_++){
            //std::cout<<"VP="<<v_P(i1_, i2_)<<"\tVG="<<v_G(i1_, i2_)<<std::endl;
            ASSERT_EQ(v_G(i1_, i2_), v_P(i1_, i2_));
        }
    }

}
*/

/*
TEST(TEST_CATEGORY, test_deepcopy_layout) {
    void test_deepcopy_layout(10, 10);
}
*/