#include <librealsense2/rs.hpp> // camera access and frame handling
#include <opencv2/opencv.hpp> // creating and displaying the occupancy grid
#include <iostream>

const float cell_size = 0.05f; // Each grid cell represents 5 cm × 5 cm in the real world.
const int grid_size = 500; //  a 500 × 500 grid (25m × 25m).
const float max_range = 3.0f; // Only consider obstacles up to 3 meters from the camera.
const int robot_cell = grid_size / 2; // Robot is at the center of the grid.
const float fx = 390.8694f; // Focal length in pixels (for a RealSense camera).
const float fy = 390.8694f; // Focal length in pixels (for a RealSense camera).
// The robot's position is at the center of the grid (250, 250) in grid coordinates.
// The robot's position is at (0, 0) in the camera frame, so we need to adjust the grid coordinates accordingly.

int main() {
    rs2::pipeline pipe;
    pipe.start(); // Starts the RealSense data pipeline — automatically enables the depth stream.

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::depth_frame depth = frames.get_depth_frame(); // extracting the depth frame from the frameset

        rs2::video_stream_profile depth_profile = depth.get_profile().as<rs2::video_stream_profile>(); // Get the video stream profile of the depth frame
        rs2_intrinsics intr = depth_profile.get_intrinsics(); // Get the intrinsics of the depth frame

        // // Check frameset  data
        // std::cout << "FrameSet contains " << frames.size() << " frames." << std::endl;
        // for (auto&& f : frames) {
        //     std::cout << "Stream: " << f.get_profile().stream_name()
        //             << ", Format: " << f.get_profile().format()
        //             << std::endl;
        // }


        int width = depth.get_width();
        int height = depth.get_height();

        //Visualize depth image
        // Convert depth data to a CV_16U Mat (RealSense depth is 16-bit unsigned)
        cv::Mat depth_image(cv::Size(width, height), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

        // Normalize for display (convert to 8-bit)
        cv::Mat depth_display;
        depth_image.convertTo(depth_display, CV_8U, 255.0 / max_range / 1000.0); // assuming depth in mm

        cv::imshow("Depth Stream", depth_display);

        cv::Mat grid = cv::Mat::zeros(grid_size, grid_size, CV_8UC1); // Create a grid of zeros (empty cells)
        // 8UC1 means 8-bit single-channel image (grayscale).

        cv::circle(grid, cv::Point(robot_cell, robot_cell), 3, 128, -1); // Draw the robot's position in the grid

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float dist = depth.get_distance(x, y);

                if (dist > 0.2f && dist < max_range) {
                    // Deproject to 3D point (camera frame) for pinhole cameras
                    // float dx = (x - width / 2.0f) * dist / intr.fx; // real world horizontal position
                    // float dy = (y - height / 2.0f) * dist / intr.fy; // real world vertical position
                    // float dz = dist; 

                    float pixel[2] = { static_cast<float>(x), static_cast<float>(y) };
                    float point[3];
                    rs2_deproject_pixel_to_point(point, &intr, pixel, dist);

                    float dx = point[0]; // X
                    float dy = point[1]; // Y
                    float dz = point[2]; // Z (depth)



                    // Only use points near the floor plane (optional)
                    if (dy > -1.0f && dy < 1.0f) { // Check if the point is within a certain height range +-30 cm
                        // Convert to grid coordinates
                        int gx = robot_cell + static_cast<int>(dx / cell_size);
                        int gy = robot_cell - static_cast<int>(dz / cell_size);

                        if (gx >= 0 && gx < grid_size && gy >= 0 && gy < grid_size) {
                            grid.at<uchar>(gy, gx) = 255; // mark as occupied
                        }
                    }
                }
            }
        }

        cv::imshow("Dynamic Grid", grid);
        cv::imwrite("dynamic_grid.png", grid);

        if (cv::waitKey(1) == 27) break; // ESC to exit
    }

    return 0;
}
