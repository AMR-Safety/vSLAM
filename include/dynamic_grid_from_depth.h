#pragma once

#include <librealsense2/rs.hpp>
#include <Eigen/Dense>
#include "occupancy_grid.h"

void updateDynamicGrid(rs2::depth_frame& depth,
                       const rs2_intrinsics& intrin,
                       const Eigen::Matrix3f& Rwc,
                       const Eigen::Vector3f& twc,
                       OccupancyGrid& dynGrid,
                       float max_range = 3.0);
