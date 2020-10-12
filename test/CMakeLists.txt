# 查找 GTest 库
find_package(GTest REQUIRED)
# GTest 的头文件
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(double_camera_test double_camera_test.cpp)

# 链接测试库
target_link_libraries( double_camera_test
        ${GTEST_BOTH_LIBRARIES}
        pthread )

# 添加到测试
gtest_discover_tests(double_camera_test)
