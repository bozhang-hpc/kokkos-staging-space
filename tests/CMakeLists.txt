include(CTest)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           master
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR}) 
endif()

SET(NAME KokkosStaging_TestAll)
FILE(GLOB TEST_SRCS Test_Main.cpp Test_DeepCopy.cpp) 
add_executable(${NAME} ${TEST_SRCS})
target_link_libraries(${NAME} PUBLIC Kokkos::staging)
target_link_libraries(${NAME} PUBLIC gtest_main)

add_executable(Test_DeepCopy_Layout Test_DeepCopy_Layout.cpp)
target_link_libraries(Test_DeepCopy_Layout PUBLIC Kokkos::staging)