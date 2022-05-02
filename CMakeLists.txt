project(LMTool)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

option(BUILD_APPS "Build APPs For LM" YES)
option(BUILD_TESTS "Build test For LM" No)

#添加Eigen&OpenCV依赖
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

#添加对应包含文件
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(backend)
#add_subdirectory(utils)

if (BUILD_APPS)
    add_subdirectory(app)
endif ()


