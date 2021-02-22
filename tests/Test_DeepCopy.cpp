#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_StagingSpace.hpp>
#include <mpi.h>
#include <string.h>
#include <iostream>

//----------------------------------------------------------------------------
/** \brief  Test for 1D Deep Copy from ExecSpace to StagingSpace and then from
 * StagingSpace to ExecSpace.
 */
template <class Data_t>
void test_deepcopy(int i1)
{
    using ViewHost_t    = Kokkos::View<Data_t*, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t*, Kokkos::StagingSpace>;

    std::string v_s_label ="StagingView_1D_";
    v_s_label += std::to_string(i1);

    ViewHost_t v_P("PutView", i1);
    ViewStaging_t v_S(v_s_label, i1);
    ViewHost_t v_G("GetView", i1);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
                            v_P(i1_) = i1_;
    });

    Kokkos::deep_copy(v_S, v_P);
    
    Kokkos::deep_copy(v_G, v_S);

    for(int i1_=0; i1_<i1; i1_++) {
        ASSERT_EQ(v_G(i1_), v_P(i1_));
    }

}

//----------------------------------------------------------------------------
/** \brief  Test for 2D Deep Copy from ExecSpace to StagingSpace and then from
 * StagingSpace to ExecSpace.
 */
template <class Data_t>
void test_deepcopy(int i1, int i2)
{
    using ViewHost_t    = Kokkos::View<Data_t**, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t**, Kokkos::StagingSpace>;

    std::string v_s_label ="StagingView_2D_";
    v_s_label += std::to_string(i1)+"_"+std::to_string(i2);

    ViewHost_t v_P("PutView", i1, i2);
    ViewStaging_t v_S(v_s_label, i1, i2);
    ViewHost_t v_G("GetView", i1, i2);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++) 
            v_P(i1_,i2_) = i1_*i2+i2_;
    });

    Kokkos::deep_copy(v_S, v_P);
    
    Kokkos::deep_copy(v_G, v_S);

    for(int i1_=0; i1_<i1; i1_++) {
        for(int i2_=0; i2_<i2; i2_++)
            ASSERT_EQ(v_G(i1_, i2_), v_P(i1_, i2_));
    }

}

//----------------------------------------------------------------------------
/** \brief  Test for 3D Deep Copy from ExecSpace to StagingSpace and then from
 * StagingSpace to ExecSpace.
 */
template <class Data_t>
void test_deepcopy(int i1, int i2, int i3)
{
    using ViewHost_t    = Kokkos::View<Data_t***, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t***, Kokkos::StagingSpace>;

    std::string v_s_label ="StagingView_3D_";
    v_s_label += std::to_string(i1)+"_"+std::to_string(i2)+"_"+
                std::to_string(i3);

    ViewHost_t v_P("PutView", i1, i2, i3);
    ViewStaging_t v_S(v_s_label, i1, i2, i3);
    ViewHost_t v_G("GetView", i1, i2, i3);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                v_P(i1_,i2_,i3_) = (i1_*i2+i2_)*i3+i3_;
    });

    Kokkos::deep_copy(v_S, v_P);
    
    Kokkos::deep_copy(v_G, v_S);

    for(int i1_=0; i1_<i1; i1_++) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                ASSERT_EQ(v_G(i1_, i2_, i3_), v_P(i1_, i2_, i3_));
    }

}

//----------------------------------------------------------------------------
/** \brief  Test for 4D Deep Copy from ExecSpace to StagingSpace and then from
 * StagingSpace to ExecSpace.
 */
template <class Data_t>
void test_deepcopy(int i1, int i2, int i3, int i4)
{
    using ViewHost_t    = Kokkos::View<Data_t****, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t****, Kokkos::StagingSpace>;

    std::string v_s_label ="StagingView_4D_";
    v_s_label += std::to_string(i1)+"_"+std::to_string(i2)+"_"+
                std::to_string(i3)+"_"+std::to_string(i4);

    ViewHost_t v_P("PutView", i1, i2, i3, i4);
    ViewStaging_t v_S(v_s_label, i1, i2, i3, i4);
    ViewHost_t v_G("GetView", i1, i2, i3, i4);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    v_P(i1_,i2_,i3_,i4_) = ((i1_*i2+i2_)*i3+i3_)*i4+i4_;
    });

    Kokkos::deep_copy(v_S, v_P);
    
    Kokkos::deep_copy(v_G, v_S);

    for(int i1_=0; i1_<i1; i1_++) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    ASSERT_EQ(v_G(i1_, i2_, i3_, i4_), v_P(i1_, i2_, i3_, i4_));
    }

}

//----------------------------------------------------------------------------
/** \brief  Test for 5D Deep Copy from ExecSpace to StagingSpace and then from
 * StagingSpace to ExecSpace.
 */
template <class Data_t>
void test_deepcopy(int i1, int i2, int i3, int i4, int i5)
{
    using ViewHost_t    = Kokkos::View<Data_t*****, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t*****, Kokkos::StagingSpace>;

    std::string v_s_label ="StagingView_5D_";
    v_s_label += std::to_string(i1)+"_"+std::to_string(i2)+"_"+
                std::to_string(i3)+"_"+std::to_string(i4)+"_"+
                std::to_string(i5);

    ViewHost_t v_P("PutView", i1, i2, i3, i4, i5);
    ViewStaging_t v_S(v_s_label, i1, i2, i3, i4, i5);
    ViewHost_t v_G("GetView", i1, i2, i3, i4, i5);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    for(int i5_=0; i5_<i5; i5_++)
                        v_P(i1_,i2_,i3_,i4_,i5_) = 
                            (((i1_*i2+i2_)*i3+i3_)*i4+i4_)*i5+i5_;
    });

    Kokkos::deep_copy(v_S, v_P);
    
    Kokkos::deep_copy(v_G, v_S);

    for(int i1_=0; i1_<i1; i1_++) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    for(int i5_=0; i5_<i5; i5_++)
                        ASSERT_EQ(v_G(i1_, i2_, i3_, i4_, i5_), 
                                    v_P(i1_, i2_, i3_, i4_, i5_));
    }

}

//----------------------------------------------------------------------------
/** \brief  Test for 6D Deep Copy from ExecSpace to StagingSpace and then from
 * StagingSpace to ExecSpace.
 */
template <class Data_t>
void test_deepcopy(int i1, int i2, int i3, int i4, int i5, int i6)
{
    using ViewHost_t    = Kokkos::View<Data_t******, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t******, Kokkos::StagingSpace>;

    std::string v_s_label ="StagingView_6D_";
    v_s_label += std::to_string(i1)+"_"+std::to_string(i2)+"_"+
                std::to_string(i3)+"_"+std::to_string(i4)+"_"+
                std::to_string(i5)+"_"+std::to_string(i6);

    ViewHost_t v_P("PutView", i1, i2, i3, i4, i5, i6);
    ViewStaging_t v_S(v_s_label, i1, i2, i3, i4, i5, i6);
    ViewHost_t v_G("GetView", i1, i2, i3, i4, i5, i6);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    for(int i5_=0; i5_<i5; i5_++)
                        for(int i6_=0; i6_<i6; i6_++)
                            v_P(i1_,i2_,i3_,i4_,i5_,i6_) = 
                                ((((i1_*i2+i2_)*i3+i3_)*i4+i4_)*i5+i5_)*i6+i6_;
    });

    Kokkos::deep_copy(v_S, v_P);
    
    Kokkos::deep_copy(v_G, v_S);

    for(int i1_=0; i1_<i1; i1_++) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    for(int i5_=0; i5_<i5; i5_++)
                        for(int i6_=0; i6_<i6; i6_++)
                            ASSERT_EQ(v_G(i1_, i2_, i3_, i4_, i5_, i6_), 
                                        v_P(i1_, i2_, i3_, i4_, i5_, i6_));
    }

}

//----------------------------------------------------------------------------
/** \brief  Test for 7D Deep Copy from ExecSpace to StagingSpace and then from
 * StagingSpace to ExecSpace.
 */
template <class Data_t>
void test_deepcopy(int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
    using ViewHost_t    = Kokkos::View<Data_t*******, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t*******, Kokkos::StagingSpace>;

    std::string v_s_label ="StagingView_7D_";
    v_s_label += std::to_string(i1)+"_"+std::to_string(i2)+"_"+
                std::to_string(i3)+"_"+std::to_string(i4)+"_"+
                std::to_string(i5)+"_"+std::to_string(i6)+"_"+
                std::to_string(i7);

    ViewHost_t v_P("PutView", i1, i2, i3, i4, i5, i6, i7);
    ViewStaging_t v_S(v_s_label, i1, i2, i3, i4, i5, i6, i7);
    ViewHost_t v_G("GetView", i1, i2, i3, i4, i5, i6, i7);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    for(int i5_=0; i5_<i5; i5_++)
                        for(int i6_=0; i6_<i6; i6_++)
                            for(int i7_=0; i7_<i7; i7_++)
                                v_P(i1_,i2_,i3_,i4_,i5_,i6_,i7_) = 
                                    (((((i1_*i2+i2_)*i3+i3_)*i4+i4_)*i5+i5_)*i6+i6_)*i7+i7_;
    });

    Kokkos::deep_copy(v_S, v_P);
    
    Kokkos::deep_copy(v_G, v_S);

    for(int i1_=0; i1_<i1; i1_++) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    for(int i5_=0; i5_<i5; i5_++)
                        for(int i6_=0; i6_<i6; i6_++)
                            for(int i7_=0; i7_<i7; i7_++)
                                ASSERT_EQ(v_G(i1_, i2_, i3_, i4_, i5_, i6_, i7_), 
                                    v_P(i1_, i2_, i3_, i4_, i5_, i6_, i7_));
    }

}

//----------------------------------------------------------------------------
/** \brief  Test for 8D Deep Copy from ExecSpace to StagingSpace and then from
 * StagingSpace to ExecSpace.
 */
template <class Data_t>
void test_deepcopy(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8)
{
    using ViewHost_t    = Kokkos::View<Data_t********, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t********, Kokkos::StagingSpace>;

    std::string v_s_label ="StagingView_8D_";
    v_s_label += std::to_string(i1)+"_"+std::to_string(i2)+"_"+
                std::to_string(i3)+"_"+std::to_string(i4)+"_"+
                std::to_string(i5)+"_"+std::to_string(i6)+"_"+
                std::to_string(i7)+"_"+std::to_string(i8);

    ViewHost_t v_P("PutView", i1, i2, i3, i4, i5, i6, i7, i8);
    ViewStaging_t v_S(v_s_label, i1, i2, i3, i4, i5, i6, i7, i8);
    ViewHost_t v_G("GetView", i1, i2, i3, i4, i5, i6, i7, i8);

    Kokkos::parallel_for(i1, KOKKOS_LAMBDA(const int i1_) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    for(int i5_=0; i5_<i5; i5_++)
                        for(int i6_=0; i6_<i6; i6_++)
                            for(int i7_=0; i7_<i7; i7_++)
                                for(int i8_=0; i8_<i8; i8_++)
                                    v_P(i1_,i2_,i3_,i4_,i5_,i6_,i7_,i8_) = 
                                        ((((((i1_*i2+i2_)*i3+i3_)*i4+i4_)*i5+i5_)*i6+i6_)*i7+i7_)*i8+i8_;
    });

    Kokkos::deep_copy(v_S, v_P);
    
    Kokkos::deep_copy(v_G, v_S);

    for(int i1_=0; i1_<i1; i1_++) {
        for(int i2_=0; i2_<i2; i2_++)
            for(int i3_=0; i3_<i3; i3_++)
                for(int i4_=0; i4_<i4; i4_++)
                    for(int i5_=0; i5_<i5; i5_++)
                        for(int i6_=0; i6_<i6; i6_++)
                            for(int i7_=0; i7_<i7; i7_++)
                                for(int i8_; i8_<i8; i8_++)
                                    ASSERT_EQ(v_G(i1_, i2_, i3_, i4_, i5_, i6_, i7_, i8_), 
                                                v_P(i1_, i2_, i3_, i4_, i5_, i6_, i7_, i8_));
    }

}

TEST(TEST_CATEGORY, test_deepcopy) {
    
    //1D
    test_deepcopy<int>(10);
    test_deepcopy<int64_t>(10);
    test_deepcopy<double>(10);

    //2D
    test_deepcopy<int>(10, 10);
    test_deepcopy<int64_t>(10, 10);
    test_deepcopy<double>(10, 10);

    //3D
    test_deepcopy<int>(10,10,10);
    test_deepcopy<int64_t>(10,10,10);
    test_deepcopy<double>(10,10,10);

    //4D
    test_deepcopy<int>(10,10,10,10);
    test_deepcopy<int64_t>(10,10,10,10);
    test_deepcopy<double>(10,10,10,10);

    //5D
    test_deepcopy<int>(10,10,10,10,10);
    test_deepcopy<int64_t>(10,10,10,10,10);
    test_deepcopy<double>(10,10,10,10,10);

    //6D
    test_deepcopy<int>(10,10,10,10,10,10);
    test_deepcopy<int64_t>(10,10,10,10,10,10);
    test_deepcopy<double>(10,10,10,10,10,10);

    //7D
    test_deepcopy<int>(10,10,10,10,10,10,10);
    test_deepcopy<int64_t>(10,10,10,10,10,10,10);
    test_deepcopy<double>(10,10,10,10,10,10,10);

    //8D
    test_deepcopy<int>(10,10,10,10,10,10,10,10);
    test_deepcopy<int64_t>(10,10,10,10,10,10,10,10);
    test_deepcopy<double>(10,10,10,10,10,10,10,10);

}