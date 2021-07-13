project(hhh-slam)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

add_subdirectory(perception)
add_subdirectory(mapping)
add_subdirectory(locolization)
add_subdirectory(common)
