# 查找 GTest 库
find_package(GTest REQUIRED)
# GTest 的头文件
include_directories(${GTEST_INCLUDE_DIRS} )

add_library(codeToTest ../src/common.cpp ../src/VisualOdom.cpp ../include/VisualOdom.h)

add_executable(all_test main.cpp)

# 链接测试库
target_link_libraries( all_test
        ${GTEST_BOTH_LIBRARIES}
        pthread
        codeToTest)

# 添加到测试
gtest_discover_tests(all_test)
