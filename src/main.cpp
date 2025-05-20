#include "occupancy_grid.h"
#include "dynamic_grid_from_depth.h"

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <Eigen/Dense>




void run() {
    // Setup grid
    GridMeta meta = {0.05, -5, -5, 200, 200}; // 10x10m @ 5cm
    OccupancyGrid staticGrid(meta);
    OccupancyGrid dynamicGrid(meta);

    // Inflate static layer as example
    staticGrid.inflate(2);

    // Initialize RealSense
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    // Get depth intrinsics
    rs2::frameset frames = pipe.wait_for_frames();
    rs2::depth_frame depth = frames.get_depth_frame();
    auto profile = depth.get_profile().as<rs2::video_stream_profile>();
    rs2_intrinsics intrin = profile.get_intrinsics();

    // Identity pose for now (replace with ORB-SLAM3 camera pose)
    Eigen::Matrix3f Rwc = Eigen::Matrix3f::Identity();
    Eigen::Vector3f twc = Eigen::Vector3f(0, 0, 0);

    while (cv::waitKey(1) != 27) { // ESC to exit
        frames = pipe.wait_for_frames();
        depth = frames.get_depth_frame();

        dynamicGrid.clear();

        updateDynamicGrid(depth, intrin, Rwc, twc, dynamicGrid);

        // Fuse layers
        cv::Mat combined;
        cv::bitwise_or(staticGrid.getGrid(), dynamicGrid.getGrid(), combined);

        // Show result
        //cv::imshow("Occupancy Grid", combined);
        //cv::imshow("Dynamic Grid Only", dynamicGrid.getGrid());

        cv::Mat colorGrid;
        cv::applyColorMap(dynamicGrid.getGrid(), colorGrid, cv::COLORMAP_JET);
        cv::imshow("Dynamic Grid Color", colorGrid);


    }

    pipe.stop();
}

int main() {
    run();
    return 0;
}
