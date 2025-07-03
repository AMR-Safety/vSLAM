// Includes
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <limits>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>

#include "System.h"
#include "Map.h"
#include "Atlas.h"
#include "MapPoint.h"

#define OBSTACLE_INFLATION_CELLS 2
#define TEXT_FILE 0
#define BINARY_FILE 1


bool runningFlag = true;
std::mutex gridMutex, slamMutex;

struct RobotState {
    float x = 0, y = 0, theta = 0;
};

struct OccupancyGrid {
    float originX, originY;
    int width, height;
    float resolution = 0.05f;
    cv::Mat staticGrid, dynamicGrid, combinedGrid;

    bool worldToGrid(float wx, float wy, int& gx, int& gy) {
        gx = static_cast<int>((wx - originX) / resolution);
        gy = static_cast<int>((wy - originY) / resolution);
        return gx >= 0 && gx < width && gy >= 0 && gy < height;
    }

    void updateCombinedGrid() {
        combinedGrid = staticGrid.clone();
        for (int y = 0; y < dynamicGrid.rows; y++) {
            for (int x = 0; x < dynamicGrid.cols; x++) {
                if (dynamicGrid.at<uint8_t>(y, x) == 255)
                    combinedGrid.at<uint8_t>(y, x) = 255;
            }
        }
    }
};

ORB_SLAM3::System* slamPtr = nullptr;

void loadORBSLAMMap(OccupancyGrid& grid) {
    ORB_SLAM3::System* slam = new ORB_SLAM3::System(
        "ORB_SLAM3/Vocabulary/ORBvoc.txt",
        "ORB_SLAM3/Examples/Stereo-Inertial/RealSense_D435i.yaml",
        ORB_SLAM3::System::STEREO, false);

    slam->mStrLoadAtlasFromFile = "/home/keshawa/amr_safety/orb_slam3/d435i_map";
    if (!slam->LoadAtlas(BINARY_FILE)) throw std::runtime_error("Failed to load .osa map");

    slamPtr = slam;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto* atlas = slam->GetAtlas();
    auto* map = atlas->GetCurrentMap();
    auto mapPoints = map->GetAllMapPoints();

    float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
    std::vector<Eigen::Vector3f> validPoints;
    for (auto* mp : mapPoints) {
        if (!mp || mp->isBad()) continue;
        Eigen::Vector3f pos = mp->GetWorldPos();
        if (pos[2] < 0.1f || pos[2] > 1.5f) continue;
        validPoints.push_back(pos);
        minX = std::min(minX, pos[0]); minY = std::min(minY, pos[1]);
        maxX = std::max(maxX, pos[0]); maxY = std::max(maxY, pos[1]);
    }

    minX -= 1.0f; maxX += 1.0f; minY -= 1.0f; maxY += 1.0f;
    grid.originX = minX; grid.originY = minY;
    grid.width = static_cast<int>((maxX - minX) / grid.resolution) + 1;
    grid.height = static_cast<int>((maxY - minY) / grid.resolution) + 1;

    grid.staticGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
    grid.dynamicGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
    grid.combinedGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);

    for (const auto& pos : validPoints) {
        int gx, gy;
        if (grid.worldToGrid(pos[0], pos[1], gx, gy))
            grid.staticGrid.at<uint8_t>(gy, gx) = 255;
    }

    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(2 * OBSTACLE_INFLATION_CELLS + 1, 2 * OBSTACLE_INFLATION_CELLS + 1));
    cv::dilate(grid.staticGrid, grid.staticGrid, element);
    grid.updateCombinedGrid();
    cv::imwrite("static_grid.png", grid.staticGrid);
}

void slamTrackingLoop(RobotState& robot) {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
    pipe.start(cfg);

    while (runningFlag) {
        rs2::frameset frames = pipe.wait_for_frames();
        auto left = frames.get_infrared_frame(1);
        auto right = frames.get_infrared_frame(2);

        double timestamp = frames.get_timestamp() * 1e-3;
        cv::Mat imLeft(cv::Size(640, 480), CV_8U, (void*)left.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat imRight(cv::Size(640, 480), CV_8U, (void*)right.get_data(), cv::Mat::AUTO_STEP);

        if (!slamPtr) continue;
        Sophus::SE3f Tcw = slamPtr->TrackStereo(imLeft, imRight, timestamp);
        if (Tcw.translation().array().isFinite().all()) {
            slamMutex.lock();
            Sophus::SE3f Twc = Tcw.inverse();
            robot.x = Twc.translation().x();
            robot.y = Twc.translation().y();
            robot.theta = atan2f(Twc.rotationMatrix()(1, 0), Twc.rotationMatrix()(0, 0));
            slamMutex.unlock();
        }
    }
    pipe.stop();
}

void processDynamicObstacles(OccupancyGrid& grid, RobotState& robot) {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    rs2::pointcloud pc;
    while (runningFlag) {
        auto start = std::chrono::steady_clock::now();
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::depth_frame depth = frames.get_depth_frame();
        if (!depth) continue;

        auto stream = depth.get_profile().as<rs2::video_stream_profile>();
        auto intrinsics = stream.get_intrinsics();
        rs2::points points = pc.calculate(depth);
        auto vertices = points.get_vertices();
        size_t n = points.size();

        slamMutex.lock();
        Eigen::Matrix3f Rwc = Eigen::AngleAxisf(robot.theta, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        Eigen::Vector3f twc(robot.x, robot.y, 0.3f);
        slamMutex.unlock();

        gridMutex.lock();
        grid.dynamicGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
        for (size_t i = 0; i < n; i += 10) {
            float x = vertices[i].x, y = vertices[i].y, z = vertices[i].z;
            if (z <= 0 || z > 5.0f || y < -0.5f || y > 0.5f) continue;

            Eigen::Vector3f pc(x, y, z);
            Eigen::Vector3f pw = Rwc * pc + twc;

            int gx, gy;
            if (grid.worldToGrid(pw[0], pw[1], gx, gy) &&
                grid.staticGrid.at<uint8_t>(gy, gx) == 0)
                grid.dynamicGrid.at<uint8_t>(gy, gx) = 255;
        }

        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
            cv::Size(2 * OBSTACLE_INFLATION_CELLS + 1, 2 * OBSTACLE_INFLATION_CELLS + 1));
        cv::dilate(grid.dynamicGrid, grid.dynamicGrid, element);
        grid.updateCombinedGrid();
        gridMutex.unlock();

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, 200 - (int)duration)));
    }
    pipe.stop();
}

int main() {
    OccupancyGrid grid;
    RobotState robot;

    std::cout << "Loading static map..." << std::endl;
    loadORBSLAMMap(grid);
    std::cout << "Static map loaded. Starting threads..." << std::endl;

    std::thread slamThread(slamTrackingLoop, std::ref(robot));
    std::thread dynamicThread(processDynamicObstacles, std::ref(grid), std::ref(robot));

    while (true) {
        gridMutex.lock();
        cv::imshow("Combined Occupancy Grid", grid.combinedGrid);
        cv::waitKey(1);
        gridMutex.unlock();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    runningFlag = false;
    slamThread.join();
    dynamicThread.join();
    return 0;
}
