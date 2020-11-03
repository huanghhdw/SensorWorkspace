cmake_minimum_required(VERSION 3.10)

project(HSlam)

# 添加c++ 17标准支持
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set( CMAKE_BUILD_TYPE "Release" )

# 寻找OpenCV,Eigen,Ceres库
find_package(OpenCV REQUIRED)
find_package(Eigen3)
find_package(Ceres)

# 添加头文件
include_directories(${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})

# add sourcce files: *.cpp
#aux_source_directory(./src  SRC_LIST)

#设置可执行文件目录
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../build/bin/) 

#生成库文件
add_library(VO src/VisualOdom.cpp)
add_library(CO src/common.cpp)
#生成可执行文件
#add_executable(${PROJECT_NAME}_main ./src/main.cpp)
#add_executable(${PROJECT_NAME}_line ./src/line.cpp)
add_executable(${PROJECT_NAME} src/start_node.cpp)

# 链接OpenCV,Ceres库
#target_link_libraries(${PROJECT_NAME}_main ${OpenCV_LIBS} ${CERES_LIBRARIES})
#target_link_libraries(${PROJECT_NAME}_line ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CERES_LIBRARIES} VO CO)
#message("---------------success in generate executeFile in path: ${PROJECT_SOURCE_DIR}/build/bin/-------------------")

enable_testing()
add_subdirectory(test)

