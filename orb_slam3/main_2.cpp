// Includes
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <limits>
#include <atomic>
#include <csignal>
#include <string>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>
#include "System.h"
#include "Map.h"
#include "Atlas.h"
#include "MapPoint.h"

// Configuration constants
#define OBSTACLE_INFLATION_CELLS 2
#define TEXT_FILE 0
#define BINARY_FILE 1
#define POINT_CLOUD_SUBSAMPLE_RATE 5  // Sample every nth point
#define MIN_OBSTACLE_Z 0.0f
#define MAX_OBSTACLE_Z 5.0f
#define MIN_OBSTACLE_Y -0.5f
#define MAX_OBSTACLE_Y 0.5f
#define CAMERA_HEIGHT 0.3f
#define GRID_RESOLUTION 0.05f
#define GRID_MARGIN 1.0f
#define MIN_HEIGHT_THRESHOLD 0.1f
#define MAX_HEIGHT_THRESHOLD 1.5f
#define DYNAMIC_OBSTACLE_UPDATE_RATE 200  // ms

// Global variables
std::atomic<bool> runningFlag(true);
std::mutex gridMutex, slamMutex, mapPointsMutex;
std::string mapFilePath;

// Handle Ctrl+C gracefully
void signalHandler(int signum) {
    std::cout << "Interrupt signal received. Shutting down..." << std::endl;
    runningFlag = false;
}

struct RobotState {
    float x = 0, y = 0, z = 0, theta = 0;
    bool validPosition = false;
};

struct OccupancyGrid {
    float originX, originY;
    int width, height;
    float resolution = GRID_RESOLUTION;
    cv::Mat staticGrid, dynamicGrid, combinedGrid;

    bool worldToGrid(float wx, float wy, int& gx, int& gy) const {
        gx = static_cast<int>((wx - originX) / resolution);
        gy = static_cast<int>((wy - originY) / resolution);
        return gx >= 0 && gx < width && gy >= 0 && gy < height;
    }

    bool gridToWorld(int gx, int gy, float& wx, float& wy) const {
        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
            wx = originX + gx * resolution;
            wy = originY + gy * resolution;
            return true;
        }
        return false;
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

class SLAMSystem {
private:
    ORB_SLAM3::System* slamPtr = nullptr;
    std::vector<Eigen::Vector3f> mapPoints;

public:
    SLAMSystem() = default;
    
    ~SLAMSystem() {
        if (slamPtr) {
            slamPtr->Shutdown();
            delete slamPtr;
            slamPtr = nullptr;
        }
    }

    bool loadMap(const std::string& mapFile, OccupancyGrid& grid) {
        try {
            slamPtr = new ORB_SLAM3::System(
                "ORB_SLAM3/Vocabulary/ORBvoc.txt",
                "ORB_SLAM3/Examples/Stereo-Inertial/RealSense_D435i.yaml",
                ORB_SLAM3::System::STEREO, false);

            slamPtr->mStrLoadAtlasFromFile = mapFile;
            
            std::cout << "Loading ORB-SLAM3 map from: " << mapFile << std::endl;
            if (!slamPtr->LoadAtlas(BINARY_FILE)) {
                std::cerr << "Failed to load .osa map from: " << mapFile << std::endl;
                return false;
            }
            
            std::cout << "Map loaded successfully. Processing map points..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            auto* atlas = slamPtr->GetAtlas();
            if (!atlas) {
                std::cerr << "Failed to get Atlas from SLAM system" << std::endl;
                return false;
            }
            
            auto* map = atlas->GetCurrentMap();
            if (!map) {
                std::cerr << "Failed to get current Map from Atlas" << std::endl;
                return false;
            }
            
            auto mapPointsVec = map->GetAllMapPoints();
            std::cout << "Total map points: " << mapPointsVec.size() << std::endl;
            
            mapPoints.clear();
            float minX = std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest();
            float maxY = std::numeric_limits<float>::lowest();
            
            for (auto* mp : mapPointsVec) {
                if (!mp || mp->isBad()) continue;
                
                Eigen::Vector3f pos = mp->GetWorldPos();
                // Filter points based on height (Z coordinate)
                if (pos[2] < MIN_HEIGHT_THRESHOLD || pos[2] > MAX_HEIGHT_THRESHOLD) continue;
                
                std::lock_guard<std::mutex> lock(mapPointsMutex);
                mapPoints.push_back(pos);
                
                minX = std::min(minX, pos[0]); 
                minY = std::min(minY, pos[1]);
                maxX = std::max(maxX, pos[0]); 
                maxY = std::max(maxY, pos[1]);
            }
            
            std::cout << "Filtered map points: " << mapPoints.size() << std::endl;
            
            // Add margin around the map
            minX -= GRID_MARGIN; 
            maxX += GRID_MARGIN; 
            minY -= GRID_MARGIN; 
            maxY += GRID_MARGIN;
            
            grid.originX = minX; 
            grid.originY = minY;
            grid.width = static_cast<int>((maxX - minX) / grid.resolution) + 1;
            grid.height = static_cast<int>((maxY - minY) / grid.resolution) + 1;
            
            std::cout << "Creating occupancy grid of size " << grid.width << "x" << grid.height 
                     << " with resolution " << grid.resolution << "m/cell" << std::endl;
            
            // Initialize grid matrices
            grid.staticGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
            grid.dynamicGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
            grid.combinedGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
            
            // Add map points to static grid
            {
                std::lock_guard<std::mutex> lock(mapPointsMutex);
                for (const auto& pos : mapPoints) {
                    int gx, gy;
                    if (grid.worldToGrid(pos[0], pos[1], gx, gy))
                        grid.staticGrid.at<uint8_t>(gy, gx) = 255;
                }
            }
            
            // Dilate static obstacles for safety
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                cv::Size(2 * OBSTACLE_INFLATION_CELLS + 1, 2 * OBSTACLE_INFLATION_CELLS + 1));
            cv::dilate(grid.staticGrid, grid.staticGrid, element);
            
            grid.updateCombinedGrid();
            cv::imwrite("static_grid.png", grid.staticGrid);
            
            std::cout << "Static map processing complete" << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in loadMap: " << e.what() << std::endl;
            return false;
        }
    }

    const std::vector<Eigen::Vector3f>& getMapPoints() const {
        return mapPoints;
    }

    Sophus::SE3f trackStereo(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp) {
        if (!slamPtr) {
            return Sophus::SE3f();
        }
        return slamPtr->TrackStereo(imLeft, imRight, timestamp);
    }
};

void visualizeMapPoints(const SLAMSystem& slam, const RobotState& robot) {
    try {
        pangolin::CreateWindowAndBind("ORB-SLAM3 Map & Robot Viewer", 1024, 768);
        glEnable(GL_DEPTH_TEST);

        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -10, -10, 0, 0, 0, 0, -1, 0)
        );

        pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

        while (runningFlag) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);

            // Draw map points
            glPointSize(2.0);
            glBegin(GL_POINTS);
            glColor3f(1.0, 0.0, 0.0);
            
            std::lock_guard<std::mutex> lock(mapPointsMutex);
            const auto& points = slam.getMapPoints();
            for (const auto& p : points) {
                glVertex3f(p[0], p[1], p[2]);
            }
            glEnd();

            // Draw robot position and orientation
            if (robot.validPosition) {
                std::lock_guard<std::mutex> lock(slamMutex);
                
                // Draw robot as a triangle
                glColor3f(0.0, 1.0, 0.0);
                glBegin(GL_TRIANGLES);
                const float robotSize = 0.3f;
                float x = robot.x;
                float y = robot.y;
                float z = robot.z;
                float c = cos(robot.theta);
                float s = sin(robot.theta);
                
                // Front point
                glVertex3f(x + robotSize * c, y + robotSize * s, z);
                // Back left
                glVertex3f(x - robotSize * c + 0.5f * robotSize * s, 
                           y - robotSize * s - 0.5f * robotSize * c, z);
                // Back right
                glVertex3f(x - robotSize * c - 0.5f * robotSize * s, 
                           y - robotSize * s + 0.5f * robotSize * c, z);
                glEnd();
            }

            pangolin::FinishFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in visualizeMapPoints: " << e.what() << std::endl;
    }
}

bool initializeRealSensePipeline(rs2::pipeline& pipe, rs2::config& cfg, 
                                const rs2::stream_profile& stream1, 
                                const rs2::stream_profile& stream2, 
                                int width = 640, int height = 480, int fps = 30) {
    try {
        cfg.enable_stream(stream1.stream_type(), stream1.stream_index(), 
                         width, height, stream1.format(), fps);
        cfg.enable_stream(stream2.stream_type(), stream2.stream_index(), 
                         width, height, stream2.format(), fps);
        pipe.start(cfg);
        return true;
    }
    catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return false;
    }
}

void slamTrackingLoop(const SLAMSystem& slam, RobotState& robot) {
    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        
        std::cout << "Initializing stereo cameras for SLAM tracking..." << std::endl;
        if (!initializeRealSensePipeline(pipe, cfg, 
                          rs2::stream_profile(RS2_STREAM_INFRARED, 1, 0, 0, 0, 0),
                          rs2::stream_profile(RS2_STREAM_INFRARED, 2, 0, 0, 0, 0))) {
            std::cerr << "Failed to initialize stereo camera pipeline" << std::endl;
            return;
        }
        
        std::cout << "Stereo camera initialized. Starting tracking..." << std::endl;
        while (runningFlag) {
            try {
                rs2::frameset frames = pipe.wait_for_frames(1000); // Timeout of 1000ms
                
                if (!frames) {
                    std::cerr << "No frames received from camera" << std::endl;
                    continue;
                }
                
                auto left = frames.get_infrared_frame(1);
                auto right = frames.get_infrared_frame(2);
                
                if (!left || !right) {
                    std::cerr << "Invalid stereo frames received" << std::endl;
                    continue;
                }

                double timestamp = frames.get_timestamp() * 1e-3;
                cv::Mat imLeft(cv::Size(640, 480), CV_8U, (void*)left.get_data(), cv::Mat::AUTO_STEP);
                cv::Mat imRight(cv::Size(640, 480), CV_8U, (void*)right.get_data(), cv::Mat::AUTO_STEP);

                Sophus::SE3f Tcw = slam.trackStereo(imLeft, imRight, timestamp);
                
                if (Tcw.translation().array().isFinite().all()) {
                    std::lock_guard<std::mutex> lock(slamMutex);
                    Sophus::SE3f Twc = Tcw.inverse();
                    robot.x = Twc.translation().x();
                    robot.y = Twc.translation().y();
                    robot.z = Twc.translation().z();
                    robot.theta = atan2f(Twc.rotationMatrix()(1, 0), Twc.rotationMatrix()(0, 0));
                    robot.validPosition = true;
                }
            }
            catch (const rs2::error& e) {
                std::cerr << "RealSense error in tracking loop: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        pipe.stop();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in slamTrackingLoop: " << e.what() << std::endl;
    }
}

void processDynamicObstacles(OccupancyGrid& grid, const RobotState& robot) {
    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        
        std::cout << "Initializing depth camera for dynamic obstacle detection..." << std::endl;
        if (!initializeRealSensePipeline(pipe, cfg, 
                          rs2::stream_profile(RS2_STREAM_DEPTH, 0, 0, 0, 0, 0),
                          rs2::stream_profile(RS2_STREAM_DEPTH, 0, 0, 0, 0, 0),
                          640, 480, 30)) {
            std::cerr << "Failed to initialize depth camera pipeline" << std::endl;
            return;
        }
        
        rs2::pointcloud pc;
        std::cout << "Depth camera initialized. Starting obstacle detection..." << std::endl;
        
        while (runningFlag) {
            try {
                auto start = std::chrono::steady_clock::now();
                
                rs2::frameset frames = pipe.wait_for_frames(1000); // Timeout of 1000ms
                if (!frames) {
                    std::cerr << "No frames received from depth camera" << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
                
                rs2::depth_frame depth = frames.get_depth_frame();
                if (!depth) {
                    std::cerr << "Invalid depth frame received" << std::endl;
                    continue;
                }

                auto stream = depth.get_profile().as<rs2::video_stream_profile>();
                auto intrinsics = stream.get_intrinsics();
                rs2::points points = pc.calculate(depth);
                auto vertices = points.get_vertices();
                size_t n = points.size();

                if (!robot.validPosition) {
                    std::cout << "Waiting for valid robot position..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    continue;
                }

                // Get current robot pose
                Eigen::Matrix3f Rwc;
                Eigen::Vector3f twc;
                {
                    std::lock_guard<std::mutex> lock(slamMutex);
                    Rwc = Eigen::AngleAxisf(robot.theta, Eigen::Vector3f::UnitZ()).toRotationMatrix();
                    twc = Eigen::Vector3f(robot.x, robot.y, robot.z);
                }

                // Create new dynamic grid
                cv::Mat newDynamicGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
                
                // Process point cloud and mark dynamic obstacles
                for (size_t i = 0; i < n; i += POINT_CLOUD_SUBSAMPLE_RATE) {
                    float x = vertices[i].x;
                    float y = vertices[i].y;
                    float z = vertices[i].z;
                    
                    // Skip invalid points and points outside our region of interest
                    if (z <= MIN_OBSTACLE_Z || z > MAX_OBSTACLE_Z || 
                        y < MIN_OBSTACLE_Y || y > MAX_OBSTACLE_Y) continue;

                    // Transform point from camera to world frame
                    Eigen::Vector3f pc(x, y, z);
                    Eigen::Vector3f pw = Rwc * pc + twc;

                    // Convert to grid coordinates
                    int gx, gy;
                    if (grid.worldToGrid(pw[0], pw[1], gx, gy)) {
                        // Only mark as obstacle if not already in static map
                        if (grid.staticGrid.at<uint8_t>(gy, gx) == 0) {
                            newDynamicGrid.at<uint8_t>(gy, gx) = 255;
                        }
                    }
                }

                // Dilate dynamic obstacles for safety
                cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                    cv::Size(2 * OBSTACLE_INFLATION_CELLS + 1, 2 * OBSTACLE_INFLATION_CELLS + 1));
                cv::dilate(newDynamicGrid, newDynamicGrid, element);
                
                // Update the grid with mutex protection
                {
                    std::lock_guard<std::mutex> lock(gridMutex);
                    grid.dynamicGrid = newDynamicGrid;
                    grid.updateCombinedGrid();
                }

                // Control the update rate
                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                int sleepTime = std::max(0, DYNAMIC_OBSTACLE_UPDATE_RATE - static_cast<int>(duration));
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
            }
            catch (const rs2::error& e) {
                std::cerr << "RealSense error in obstacle detection: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        pipe.stop();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in processDynamicObstacles: " << e.what() << std::endl;
    }
}

void displayGridWithRobot(const OccupancyGrid& grid, const RobotState& robot) {
    try {
        while (runningFlag) {
            cv::Mat display;
            bool validPos = false;
            float rx = 0, ry = 0, rtheta = 0;
            
            // Get current robot position
            {
                std::lock_guard<std::mutex> lock(slamMutex);
                if (robot.validPosition) {
                    rx = robot.x;
                    ry = robot.y;
                    rtheta = robot.theta;
                    validPos = true;
                }
            }
            
            // Get current grid
            {
                std::lock_guard<std::mutex> lock(gridMutex);
                cv::cvtColor(grid.combinedGrid, display, cv::COLOR_GRAY2BGR);
            }
            
            // Draw robot on grid if position is valid
            if (validPos) {
                int gx, gy;
                if (grid.worldToGrid(rx, ry, gx, gy)) {
                    // Draw robot as a circle with orientation line
                    cv::circle(display, cv::Point(gx, gy), 5, cv::Scalar(0, 0, 255), -1);
                    cv::line(display, cv::Point(gx, gy), 
                            cv::Point(gx + 10 * cos(rtheta), gy + 10 * sin(rtheta)),
                            cv::Scalar(0, 0, 255), 2);
                }
            }
            
            // Display the grid
            cv::imshow("Navigation Grid with Robot Position", display);
            cv::waitKey(100);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in displayGridWithRobot: " << e.what() << std::endl;
    }
}

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " <path_to_map_file.osa>" << std::endl;
}

int main(int argc, char** argv) {
    // Register signal handler
    signal(SIGINT, signalHandler);
    
    // Parse command line arguments
    if (argc != 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    mapFilePath = argv[1];
    
    // Initialize robot state and occupancy grid
    OccupancyGrid grid;
    RobotState robot;
    SLAMSystem slamSystem;

    std::cout << "Loading static map from: " << mapFilePath << std::endl;
    if (!slamSystem.loadMap(mapFilePath, grid)) {
        std::cerr << "Failed to load map. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Static map loaded successfully." << std::endl;

    // Start tracking thread
    std::cout << "Starting SLAM tracking thread..." << std::endl;
    std::thread slamThread(slamTrackingLoop, std::ref(slamSystem), std::ref(robot));
    
    // Start dynamic obstacle detection thread
    std::cout << "Starting dynamic obstacle detection thread..." << std::endl;
    std::thread dynamicThread(processDynamicObstacles, std::ref(grid), std::ref(robot));
    
    // Start map point visualization thread
    std::cout << "Starting map visualization thread..." << std::endl;
    std::thread viewerThread(visualizeMapPoints, std::ref(slamSystem), std::ref(robot));
    
    // Start grid display thread
    std::cout << "Starting grid visualization thread..." << std::endl;
    std::thread displayThread(displayGridWithRobot, std::ref(grid), std::ref(robot));

    std::cout << "All threads started. Press Ctrl+C to exit." << std::endl;
    
    // Main thread waits for threads to finish
    slamThread.join();
    dynamicThread.join();
    viewerThread.join();
    displayThread.join();
    
    std::cout << "All threads have terminated. Exiting." << std::endl;
    return 0;
}