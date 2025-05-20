#include "occupancy_grid.h"
#include "dynamic_grid_from_depth.h"

#include <librealsense2/rs.hpp>
#include <Eigen/Dense>

void updateDynamicGrid(rs2::depth_frame& depth, const rs2_intrinsics& intrin,
                       const Eigen::Matrix3f& Rwc, const Eigen::Vector3f& twc,
                       OccupancyGrid& dynGrid, float max_range = 3.0) {

    const int step = 4;  // Downsample factor
    for (int y = 0; y < depth.get_height(); y += step) {
        for (int x = 0; x < depth.get_width(); x += step) {
            float dist = depth.get_distance(x, y);
            if (dist < 0.3 || dist > max_range) continue;

            // Pixel (x, y) to camera 3D
            float X = (x - intrin.ppx) * dist / intrin.fx;
            float Y = (y - intrin.ppy) * dist / intrin.fy;
            float Z = dist;

            Eigen::Vector3f Pc(X, Y, Z);
            Eigen::Vector3f Pw = Rwc * Pc + twc;
            dynGrid.markOccupied(Pw[0], Pw[1]);
        }
    }

    dynGrid.inflate(1);  // Optional
}
