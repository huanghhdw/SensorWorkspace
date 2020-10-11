cmake_minimum_required(VERSION 3.10)

project(double_camera)

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

MESSAGE( STATUS "========================== ${PROJECT_NAME} is about to construct==================================")
MESSAGE( STATUS "========================== ${PROJECT_NAME} is about to construct==================================")
MESSAGE( STATUS "========================== ${PROJECT_NAME} is about to construct==================================")

# 添加头文件
include_directories(./include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})

# add sourcce files: *.cpp
aux_source_directory(./src  SRC_LIST)

#设置可执行文件目录
SET(EXECUTABLE_OUTPUT_PATH ./bin/) 


#生成可执行文件
add_executable(${PROJECT_NAME} ${SRC_LIST})

# 链接OpenCV,Ceres库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CERES_LIBRARIES})

MESSAGE( STATUS "====succeed  in generating binary file  ${PROJECT_NAME} in path :  ${EXECUTABLE_OUTPUT_PATH}=======")
