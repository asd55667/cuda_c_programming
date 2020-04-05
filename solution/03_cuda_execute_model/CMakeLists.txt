cmake_minimum_required(VERSION 3.10)

project(sumArray)

message(${CMAKE_CURRENT_SOURCE_DIR})

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++14")
set(CMAKE_CXX_STANDARD 14)

set (src_dir "${PROJECT_SOURCE_DIR}")
file (GLOB src_files "${src_dir}/*.cpp")

add_executable (sumArray ${src_files})

