cmake_minimum_required(VERSION 3.10)
project(orb_slam3)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# ORB-SLAM3 headers and libs
include_directories(/home/keshawa/amr_safety/orb_slam3/ORB_SLAM3/include)
include_directories(/home/keshawa/amr_safety/orb_slam3/ORB_SLAM3/Thirdparty)
include_directories(/home/keshawa/amr_safety/orb_slam3/ORB_SLAM3/Thirdparty/Sophus)

include_directories(/home/keshawa/amr_safety/orb_slam3/ORB_SLAM3/src)
link_directories(/home/keshawa/amr_safety/orb_slam3/ORB_SLAM3/lib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# RealSense
find_package(realsense2 REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Source
add_executable(orb_slam3  localisation.cpp)
#add_executable(orb_slam3 main.cpp)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})


# Link libraries
target_link_libraries(orb_slam3
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    realsense2
    ORB_SLAM3
)
