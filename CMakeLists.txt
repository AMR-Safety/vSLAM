cmake_minimum_required(VERSION 3.10)
project(vslam_navigation)  

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Include RealSense manually
include_directories(/usr/include/librealsense2)
link_directories(/usr/lib)

# Include headers
include_directories(include)

# Source files
file(GLOB SOURCES "src/*.cpp")

add_executable(main_exec ${SOURCES})
target_link_libraries(main_exec ${OpenCV_LIBS} realsense2 Eigen3::Eigen)

