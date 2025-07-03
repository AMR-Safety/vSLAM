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
#include <iomanip>
#include <sstream>

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
    bool initialized = false;
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
        if (!initialized) return;
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
            // Explicitly activate localization-only mode before shutdown if needed
            if (!slamPtr->GetTrackingState())
                slamPtr->ActivateLocalizationMode();
            slamPtr->Shutdown();
            delete slamPtr;
            slamPtr = nullptr;
        }
    }

    // Struct to store map validation results
    struct MapValidationResult {
        bool isValid = false;
        std::string errorMessage;
        
        // Map statistics
        size_t totalMapPoints = 0;
        size_t validMapPoints = 0;
        size_t filteredMapPoints = 0;
        
        // Bounds information
        float minX = 0, maxX = 0, minY = 0, maxY = 0, minZ = 0, maxZ = 0;
        
        // Grid information
        int gridWidth = 0, gridHeight = 0;
        float gridResolution = 0;
        
        // SLAM system status
        bool slamInitialized = false;
        bool atlasLoaded = false;
        bool mapLoaded = false;
        bool localizationModeActive = false;
        
        void print() const {
            std::cout << "\n========== MAP VALIDATION REPORT ==========" << std::endl;
            std::cout << "Overall Status: " << (isValid ? "VALID" : "INVALID") << std::endl;
            
            if (!isValid) {
                std::cout << "Error: " << errorMessage << std::endl;
            }
            
            std::cout << "\n--- SLAM System Status ---" << std::endl;
            std::cout << "SLAM Initialized: " << (slamInitialized ? "YES" : "NO") << std::endl;
            std::cout << "Atlas Loaded: " << (atlasLoaded ? "YES" : "NO") << std::endl;
            std::cout << "Map Loaded: " << (mapLoaded ? "YES" : "NO") << std::endl;
            std::cout << "Localization Mode: " << (localizationModeActive ? "ACTIVE" : "INACTIVE") << std::endl;
            
            std::cout << "\n--- Map Points Statistics ---" << std::endl;
            std::cout << "Total Map Points: " << totalMapPoints << std::endl;
            std::cout << "Valid Map Points: " << validMapPoints << std::endl;
            std::cout << "Filtered Map Points: " << filteredMapPoints << std::endl;
            std::cout << "Filter Rate: " << std::fixed << std::setprecision(2) 
                    << (totalMapPoints > 0 ? (100.0 * filteredMapPoints / totalMapPoints) : 0) << "%" << std::endl;
            
            std::cout << "\n--- Map Bounds ---" << std::endl;
            std::cout << "X Range: [" << std::fixed << std::setprecision(3) << minX << ", " << maxX 
                    << "] = " << (maxX - minX) << "m" << std::endl;
            std::cout << "Y Range: [" << minY << ", " << maxY 
                    << "] = " << (maxY - minY) << "m" << std::endl;
            std::cout << "Z Range: [" << minZ << ", " << maxZ 
                    << "] = " << (maxZ - minZ) << "m" << std::endl;
            
            std::cout << "\n--- Occupancy Grid ---" << std::endl;
            std::cout << "Grid Size: " << gridWidth << " x " << gridHeight << " cells" << std::endl;
            std::cout << "Grid Resolution: " << gridResolution << " m/cell" << std::endl;
            std::cout << "Grid Coverage: " << std::fixed << std::setprecision(2) 
                    << (gridWidth * gridResolution) << "m x " 
                    << (gridHeight * gridResolution) << "m" << std::endl;
            
            std::cout << "==========================================\n" << std::endl;
        }
    };

    // Enhanced map validation function - add this to your SLAMSystem class
    MapValidationResult validateMapLoading(const std::string& mapFile, OccupancyGrid& grid) {
        MapValidationResult result;
        
        try {
            std::cout << "=== Starting Map Validation ===" << std::endl;
            
            // Step 1: Check if map file exists and is readable
            std::cout << "1. Checking map file: " << mapFile << std::endl;
            std::ifstream file(mapFile, std::ios::binary);
            if (!file.good()) {
                result.errorMessage = "Map file does not exist or is not readable: " + mapFile;
                return result;
            }
            file.close();
            std::cout << "   ✓ Map file exists and is readable" << std::endl;
            
            // Step 2: Initialize SLAM system
            std::cout << "2. Initializing ORB-SLAM3 system..." << std::endl;
            if (slamPtr) {
                delete slamPtr;
                slamPtr = nullptr;
            }
            
            slamPtr = new ORB_SLAM3::System(
                "/home/keshawa/amr_safety/orb_slam3/ORB_SLAM3/Vocabulary/ORBvoc.txt",
                "/home/keshawa/amr_safety/orb_slam3/ORB_SLAM3/Examples/Stereo-Inertial/RealSense_D435i_localize.yaml",
                ORB_SLAM3::System::STEREO, true);
            
            if (!slamPtr) {
                result.errorMessage = "Failed to initialize ORB-SLAM3 system";
                return result;
            }
            result.slamInitialized = true;
            std::cout << "   ✓ ORB-SLAM3 system initialized" << std::endl;
            
            // Step 3: Load the atlas
            std::cout << "3. Loading atlas from map file..." << std::endl;
            slamPtr->mStrLoadAtlasFromFile = mapFile;
            
            if (!slamPtr->LoadAtlas(BINARY_FILE)) {
                result.errorMessage = "Failed to load atlas from map file";
                return result;
            }
            result.atlasLoaded = true;
            std::cout << "   ✓ Atlas loaded successfully" << std::endl;
            
            // Step 4: Activate localization mode
            std::cout << "4. Activating localization mode..." << std::endl;
            slamPtr->ActivateLocalizationMode();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            slamPtr->ActivateLocalizationMode();
            std::this_thread::sleep_for(std::chrono::seconds(1));  // Let it settle
            std::cout << "[SLAM] Localization mode requested." << std::endl;

            result.localizationModeActive = true;
            std::cout << "   ✓ Localization mode activated" << std::endl;
            
            // Step 5: Validate Atlas and Map
            std::cout << "5. Validating atlas and map structure..." << std::endl;
            auto* atlas = slamPtr->GetAtlas();
            if (!atlas) {
                result.errorMessage = "Failed to get Atlas from SLAM system";
                return result;
            }
            
            auto* map = atlas->GetCurrentMap();
            if (!map) {
                result.errorMessage = "Failed to get current Map from Atlas";
                return result;
            }
            result.mapLoaded = true;
            std::cout << "   ✓ Atlas and map structure validated" << std::endl;
            
            // Step 6: Analyze map points
            std::cout << "6. Analyzing map points..." << std::endl;
            auto mapPointsVec = map->GetAllMapPoints();
            result.totalMapPoints = mapPointsVec.size();
            
            if (result.totalMapPoints == 0) {
                result.errorMessage = "Map contains no map points";
                return result;
            }
            
            mapPoints.clear();
            float minX = std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float minZ = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest();
            float maxY = std::numeric_limits<float>::lowest();
            float maxZ = std::numeric_limits<float>::lowest();
            
            size_t validCount = 0;
            size_t filteredCount = 0;
            
            for (auto* mp : mapPointsVec) {
                if (!mp || mp->isBad()) continue;
                validCount++;
                
                Eigen::Vector3f pos = mp->GetWorldPos();
                
                // Update overall bounds
                minX = std::min(minX, pos[0]); 
                minY = std::min(minY, pos[1]);
                minZ = std::min(minZ, pos[2]);
                maxX = std::max(maxX, pos[0]); 
                maxY = std::max(maxY, pos[1]);
                maxZ = std::max(maxZ, pos[2]);
                
                // Apply height filter
                if (pos[2] >= MIN_HEIGHT_THRESHOLD && pos[2] <= MAX_HEIGHT_THRESHOLD) {
                    filteredCount++;
                    std::lock_guard<std::mutex> lock(mapPointsMutex);
                    mapPoints.push_back(pos);
                }
            }
            
            result.validMapPoints = validCount;
            result.filteredMapPoints = filteredCount;
            result.minX = minX; result.maxX = maxX;
            result.minY = minY; result.maxY = maxY;
            result.minZ = minZ; result.maxZ = maxZ;
            
            if (filteredCount == 0) {
                result.errorMessage = "No map points remain after height filtering";
                return result;
            }
            
            std::cout << "   ✓ Map points analyzed: " << filteredCount << "/" << validCount 
                    << "/" << result.totalMapPoints << " (filtered/valid/total)" << std::endl;
            
            // Step 7: Create and validate occupancy grid
            std::cout << "7. Creating occupancy grid..." << std::endl;
            
            // Add margin around the map
            float gridMinX = minX - GRID_MARGIN; 
            float gridMaxX = maxX + GRID_MARGIN; 
            float gridMinY = minY - GRID_MARGIN; 
            float gridMaxY = maxY + GRID_MARGIN;
            
            grid.originX = gridMinX; 
            grid.originY = gridMinY;
            grid.width = static_cast<int>((gridMaxX - gridMinX) / grid.resolution) + 1;
            grid.height = static_cast<int>((gridMaxY - gridMinY) / grid.resolution) + 1;
            
            result.gridWidth = grid.width;
            result.gridHeight = grid.height;
            result.gridResolution = grid.resolution;
            
            // Validate grid dimensions
            if (grid.width <= 0 || grid.height <= 0) {
                result.errorMessage = "Invalid grid dimensions calculated";
                return result;
            }
            
            if (grid.width > 10000 || grid.height > 10000) {
                result.errorMessage = "Grid dimensions too large (may cause memory issues)";
                return result;
            }
            
            // Initialize grid matrices
            try {
                grid.staticGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
                grid.dynamicGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
                grid.combinedGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
            } catch (const cv::Exception& e) {
                result.errorMessage = "Failed to allocate grid memory: " + std::string(e.what());
                return result;
            }
            
            std::cout << "   ✓ Grid created: " << grid.width << "x" << grid.height << " cells" << std::endl;
            
            // Step 8: Populate static grid and validate
            std::cout << "8. Populating static grid..." << std::endl;
            size_t gridPointsAdded = 0;
            
            {
                std::lock_guard<std::mutex> lock(mapPointsMutex);
                for (const auto& pos : mapPoints) {
                    int gx, gy;
                    if (grid.worldToGrid(pos[0], pos[1], gx, gy)) {
                        grid.staticGrid.at<uint8_t>(gy, gx) = 255;
                        gridPointsAdded++;
                    }
                }
            }
            
            if (gridPointsAdded == 0) {
                result.errorMessage = "No map points were added to the grid";
                return result;
            }
            
            std::cout << "   ✓ Added " << gridPointsAdded << " points to static grid" << std::endl;
            
            // Step 9: Apply obstacle inflation
            std::cout << "9. Applying obstacle inflation..." << std::endl;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                cv::Size(2 * OBSTACLE_INFLATION_CELLS + 1, 2 * OBSTACLE_INFLATION_CELLS + 1));
            cv::dilate(grid.staticGrid, grid.staticGrid, element);
            
            // Step 10: Final validation
            std::cout << "10. Final validation..." << std::endl;
            grid.updateCombinedGrid();
            
            // Count occupied cells
            int occupiedCells = cv::countNonZero(grid.staticGrid);
            float occupancyRatio = static_cast<float>(occupiedCells) / (grid.width * grid.height);
            
            std::cout << "   ✓ Grid occupancy: " << occupiedCells << " cells (" 
                    << std::fixed << std::setprecision(2) << (occupancyRatio * 100) << "%)" << std::endl;
            
            // Save validation outputs
            cv::imwrite("validation_static_grid.png", grid.staticGrid);
            std::cout << "   ✓ Saved validation grid to 'validation_static_grid.png'" << std::endl;
            
            // All validations passed
            result.isValid = true;
            std::cout << "=== Map Validation PASSED ===" << std::endl;
            
            return result;
            
        } catch (const std::exception& e) {
            result.errorMessage = "Exception during validation: " + std::string(e.what());
            return result;
        }
    }

    // Additional helper function to test coordinate transformations
    bool testCoordinateTransformations(const OccupancyGrid& grid) {
        std::cout << "\n=== Testing Coordinate Transformations ===" << std::endl;
        
        // Test corner points
        std::vector<std::pair<float, float>> testPoints = {
            {grid.originX, grid.originY},  // Bottom-left corner
            {grid.originX + grid.width * grid.resolution, grid.originY},  // Bottom-right
            {grid.originX, grid.originY + grid.height * grid.resolution}, // Top-left
            {grid.originX + grid.width * grid.resolution, grid.originY + grid.height * grid.resolution}, // Top-right
            {grid.originX + (grid.width/2) * grid.resolution, grid.originY + (grid.height/2) * grid.resolution} // Center
        };
        
        std::vector<std::string> pointNames = {"Bottom-Left", "Bottom-Right", "Top-Left", "Top-Right", "Center"};
        
        bool allTestsPassed = true;
        
        for (size_t i = 0; i < testPoints.size(); i++) {
            float wx = testPoints[i].first;
            float wy = testPoints[i].second;
            
            int gx, gy;
            bool worldToGridOk = grid.worldToGrid(wx, wy, gx, gy);
            
            float wx2, wy2;
            bool gridToWorldOk = grid.gridToWorld(gx, gy, wx2, wy2);
            
            float error = sqrt((wx - wx2) * (wx - wx2) + (wy - wy2) * (wy - wy2));
            
            std::cout << pointNames[i] << ": World(" << std::fixed << std::setprecision(3) 
                    << wx << "," << wy << ") -> Grid(" << gx << "," << gy << ") -> World(" 
                    << wx2 << "," << wy2 << ") Error: " << error << "m" << std::endl;
            
            if (!worldToGridOk || !gridToWorldOk || error > grid.resolution) {
                std::cout << "  ❌ FAILED" << std::endl;
                allTestsPassed = false;
            } else {
                std::cout << "  ✓ PASSED" << std::endl;
            }
        }
        
        std::cout << "Coordinate transformation test: " << (allTestsPassed ? "PASSED" : "FAILED") << std::endl;
        return allTestsPassed;
    }

    // Replace the existing loadMap function in your SLAMSystem class with this enhanced version
    bool loadMapWithValidation(const std::string& mapFile, OccupancyGrid& grid) {
        MapValidationResult result = validateMapLoading(mapFile, grid);
        
        // Print detailed validation report
        result.print();
        
        // Test coordinate transformations
        if (result.isValid) {
            testCoordinateTransformations(grid);
        }
        
        return result.isValid;
    }



bool loadMap(const std::string& mapFile, OccupancyGrid& grid) {
        try {
            std::cout << "=== Starting Map Loading Process ===" << std::endl;
            
            // Check if map file exists
            std::ifstream file(mapFile, std::ios::binary);
            if (!file.good()) {
                std::cerr << "Error: Map file does not exist or is not readable: " << mapFile << std::endl;
                return false;
            }
            file.close();
            std::cout << "✓ Map file exists and is readable" << std::endl;

            // Initialize ORB-SLAM3 system
            std::cout << "Initializing ORB-SLAM3 system..." << std::endl;
            slamPtr = new ORB_SLAM3::System(
                "/home/keshawa/amr_safety/orb_slam3/ORB_SLAM3/Vocabulary/ORBvoc.txt",
                "/home/keshawa/amr_safety/orb_slam3/ORB_SLAM3/Examples/Stereo-Inertial/RealSense_D435i_localize.yaml",
                ORB_SLAM3::System::STEREO, 
                true  // Enable viewer
            );

            if (!slamPtr) {
                std::cerr << "Failed to initialize ORB-SLAM3 system" << std::endl;
                return false;
            }
            std::cout << "✓ ORB-SLAM3 system initialized" << std::endl;

            // Load the atlas
            std::cout << "Loading atlas from: " << mapFile << std::endl;
            slamPtr->mStrLoadAtlasFromFile = mapFile;
            
            if (!slamPtr->LoadAtlas(BINARY_FILE)) {
                std::cerr << "Failed to load atlas from map file: " << mapFile << std::endl;
                std::cerr << "Please check if the file is a valid .osa file" << std::endl;
                return false;
            }
            std::cout << "✓ Atlas loaded successfully" << std::endl;

            // Activate localization mode
            std::cout << "Activating localization mode..." << std::endl;
            slamPtr->ActivateLocalizationMode();
            std::this_thread::sleep_for(std::chrono::seconds(2));  // Give it time to activate
            std::cout << "✓ Localization mode activated" << std::endl;

            // Get map points
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
            std::cout << "Total map points in atlas: " << mapPointsVec.size() << std::endl;

            if (mapPointsVec.empty()) {
                std::cerr << "No map points found in the loaded map" << std::endl;
                return false;
            }

            // Process map points
            mapPoints.clear();
            float minX = std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest();
            float maxY = std::numeric_limits<float>::lowest();

            size_t validPoints = 0;
            for (auto* mp : mapPointsVec) {
                if (!mp || mp->isBad()) continue;

                Eigen::Vector3f pos = mp->GetWorldPos();
                
                // Filter points based on height
                if (pos[2] < MIN_HEIGHT_THRESHOLD || pos[2] > MAX_HEIGHT_THRESHOLD) 
                    continue;

                validPoints++;
                mapPoints.push_back(pos);

                minX = std::min(minX, pos[0]);
                minY = std::min(minY, pos[1]);
                maxX = std::max(maxX, pos[0]);
                maxY = std::max(maxY, pos[1]);
            }

            std::cout << "Valid map points after filtering: " << validPoints << "/" << mapPointsVec.size() << std::endl;

            if (validPoints == 0) {
                std::cerr << "No valid map points remain after height filtering" << std::endl;
                return false;
            }

            // Create occupancy grid
            std::cout << "Creating occupancy grid..." << std::endl;
            
            // Add margin
            minX -= GRID_MARGIN;
            maxX += GRID_MARGIN;
            minY -= GRID_MARGIN;
            maxY += GRID_MARGIN;

            grid.originX = minX;
            grid.originY = minY;
            grid.width = static_cast<int>((maxX - minX) / grid.resolution) + 1;
            grid.height = static_cast<int>((maxY - minY) / grid.resolution) + 1;

            std::cout << "Grid dimensions: " << grid.width << "x" << grid.height 
                     << " (resolution: " << grid.resolution << "m/cell)" << std::endl;

            // Validate grid size
            if (grid.width <= 0 || grid.height <= 0 || grid.width > 10000 || grid.height > 10000) {
                std::cerr << "Invalid grid dimensions calculated" << std::endl;
                return false;
            }

            // Initialize grid matrices
            grid.staticGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
            grid.dynamicGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
            grid.combinedGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
            grid.initialized = true;

            // Populate static grid
            std::cout << "Populating static grid..." << std::endl;
            size_t pointsAdded = 0;
            for (const auto& pos : mapPoints) {
                int gx, gy;
                if (grid.worldToGrid(pos[0], pos[1], gx, gy)) {
                    grid.staticGrid.at<uint8_t>(gy, gx) = 255;
                    pointsAdded++;
                }
            }

            std::cout << "Added " << pointsAdded << " points to static grid" << std::endl;

            // Apply obstacle inflation
            std::cout << "Applying obstacle inflation..." << std::endl;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                cv::Size(2 * OBSTACLE_INFLATION_CELLS + 1, 2 * OBSTACLE_INFLATION_CELLS + 1));
            cv::dilate(grid.staticGrid, grid.staticGrid, element);

            grid.updateCombinedGrid();

            // Save grid for verification
            cv::imwrite("loaded_static_grid.png", grid.staticGrid);
            std::cout << "✓ Static grid saved as 'loaded_static_grid.png'" << std::endl;

            std::cout << "=== Map Loading Complete ===" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Exception in loadMap: " << e.what() << std::endl;
            return false;
        }
    }    const std::vector<Eigen::Vector3f>& getMapPoints() const {
        return mapPoints;
    }

    // Ensure we stay in localization mode
    void ensureLocalizationMode() {
        if (slamPtr) {
            std::cout << "[SLAM] Re-activating localization mode..." << std::endl;
            slamPtr->ActivateLocalizationMode();
        }
    }

    Sophus::SE3f trackStereo(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp) const {
        if (!slamPtr) {
            return Sophus::SE3f();
        }
        return slamPtr->TrackStereo(imLeft, imRight, timestamp);
    }

    bool isInitialized() const {
        return slamPtr != nullptr;
    }
};

void visualizeMapPoints(const SLAMSystem& slam, const RobotState& robot) {
    if (!slam.isInitialized()) {
        std::cerr << "Cannot start visualization: SLAM system not initialized" << std::endl;
        return;
    }

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
            
            const auto& points = slam.getMapPoints();
            for (const auto& p : points) {
                glVertex3f(p[0], p[1], p[2]);
            }
            glEnd();

            // Draw robot if position is valid
            if (robot.validPosition) {
                glColor3f(0.0, 1.0, 0.0);
                glBegin(GL_TRIANGLES);
                const float robotSize = 0.3f;
                float x = robot.x;
                float y = robot.y;
                float z = robot.z;
                float c = cos(robot.theta);
                float s = sin(robot.theta);
                
                glVertex3f(x + robotSize * c, y + robotSize * s, z);
                glVertex3f(x - robotSize * c + 0.5f * robotSize * s, 
                           y - robotSize * s - 0.5f * robotSize * c, z);
                glVertex3f(x - robotSize * c - 0.5f * robotSize * s, 
                           y - robotSize * s + 0.5f * robotSize * c, z);
                glEnd();
            }

            pangolin::FinishFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in visualizeMapPoints: " << e.what() << std::endl;
    }
}


bool initializeRealSensePipeline(rs2::pipeline& pipe, rs2::config& cfg,
                                 rs2_stream stream1, int index1,
                                 rs2_stream stream2, int index2,
                                 int width = 640, int height = 480, int fps = 30) {
    try {
        cfg.enable_stream(stream1, index1, width, height, RS2_FORMAT_Y8, fps);
        cfg.enable_stream(stream2, index2, width, height, RS2_FORMAT_Y8, fps);
        pipe.start(cfg);
        return true;
    }
    catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return false;
    }
}

void slamTrackingLoop(const SLAMSystem& slam, RobotState& robot) {
    if (!slam.isInitialized()) {
        std::cerr << "Cannot start tracking: SLAM system not initialized" << std::endl;
        return;
    }

    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        
        std::cout << "Initializing stereo cameras for SLAM tracking..." << std::endl;
        if (!initializeRealSensePipeline(pipe, cfg, RS2_STREAM_INFRARED, 1, RS2_STREAM_INFRARED, 2)) {
            std::cerr << "Failed to initialize stereo camera pipeline" << std::endl;
            return;
        }
        
        std::cout << "Stereo camera initialized. Starting tracking..." << std::endl;
        
        while (runningFlag) {
            try {
                rs2::frameset frames = pipe.wait_for_frames(1000);
                
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
                
                // Periodically ensure localization mode
                static int frame_count = 0;
                if (++frame_count % 100 == 0) {
                    const_cast<SLAMSystem&>(slam).ensureLocalizationMode();
                }
                
                if (Tcw.translation().array().isFinite().all()) {
                    std::lock_guard<std::mutex> lock(slamMutex);
                    Sophus::SE3f Twc = Tcw.inverse();
                    robot.x = Twc.translation().x();
                    robot.y = Twc.translation().y();
                    robot.z = Twc.translation().z();
                    robot.theta = atan2f(Twc.rotationMatrix()(1, 0), Twc.rotationMatrix()(0, 0));
                    robot.validPosition = true;
                }
                
            } catch (const rs2::error& e) {
                std::cerr << "RealSense error in tracking loop: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        pipe.stop();
    } catch (const std::exception& e) {
        std::cerr << "Exception in slamTrackingLoop: " << e.what() << std::endl;
    }
}


void processDynamicObstacles(OccupancyGrid& grid, const RobotState& robot) {
    if (!grid.initialized) {
        std::cerr << "Cannot start dynamic obstacle processing: Grid not initialized" << std::endl;
        return;
    }

    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        
        std::cout << "Initializing depth camera for dynamic obstacles..." << std::endl;
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        pipe.start(cfg);

        rs2::pointcloud pc;
        std::cout << "Depth camera initialized. Starting obstacle detection..." << std::endl;
        
        while (runningFlag) {
            try {
                auto start = std::chrono::steady_clock::now();
                
                rs2::frameset frames = pipe.wait_for_frames(1000);
                if (!frames) continue;
                
                rs2::depth_frame depth = frames.get_depth_frame();
                if (!depth) continue;

                if (!robot.validPosition) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    continue;
                }

                auto stream = depth.get_profile().as<rs2::video_stream_profile>();
                rs2::points points = pc.calculate(depth);
                auto vertices = points.get_vertices();
                size_t n = points.size();

                // Get current robot pose
                Eigen::Matrix3f Rwc;
                Eigen::Vector3f twc;
                {
                    std::lock_guard<std::mutex> lock(slamMutex);
                    Rwc = Eigen::AngleAxisf(robot.theta, Eigen::Vector3f::UnitZ()).toRotationMatrix();
                    twc = Eigen::Vector3f(robot.x, robot.y, robot.z);
                }

                cv::Mat newDynamicGrid = cv::Mat::zeros(grid.height, grid.width, CV_8UC1);
                
                for (size_t i = 0; i < n; i += POINT_CLOUD_SUBSAMPLE_RATE) {
                    float x = vertices[i].x;
                    float y = vertices[i].y;
                    float z = vertices[i].z;
                    
                    if (z <= MIN_OBSTACLE_Z || z > MAX_OBSTACLE_Z || 
                        y < MIN_OBSTACLE_Y || y > MAX_OBSTACLE_Y) continue;

                    Eigen::Vector3f pc_point(x, y, z);
                    Eigen::Vector3f pw = Rwc * pc_point + twc;

                    int gx, gy;
                    if (grid.worldToGrid(pw[0], pw[1], gx, gy)) {
                        if (grid.staticGrid.at<uint8_t>(gy, gx) == 0) {
                            newDynamicGrid.at<uint8_t>(gy, gx) = 255;
                        }
                    }
                }

                // Apply inflation
                cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                    cv::Size(2 * OBSTACLE_INFLATION_CELLS + 1, 2 * OBSTACLE_INFLATION_CELLS + 1));
                cv::dilate(newDynamicGrid, newDynamicGrid, element);
                
                {
                    std::lock_guard<std::mutex> lock(gridMutex);
                    grid.dynamicGrid = newDynamicGrid;
                    grid.updateCombinedGrid();
                }

                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                int sleepTime = std::max(0, DYNAMIC_OBSTACLE_UPDATE_RATE - static_cast<int>(duration));
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
                
            } catch (const rs2::error& e) {
                std::cerr << "RealSense error in obstacle detection: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        pipe.stop();
    } catch (const std::exception& e) {
        std::cerr << "Exception in processDynamicObstacles: " << e.what() << std::endl;
    }
}

void displayGridWithRobot(const OccupancyGrid& grid, const RobotState& robot) {
    if (!grid.initialized) {
        std::cerr << "Cannot start grid display: Grid not initialized" << std::endl;
        return;
    }

    try {
        std::cout << "Starting grid visualization..." << std::endl;
        
        while (runningFlag) {
            cv::Mat display;
            bool validPos = false;
            float rx = 0, ry = 0, rtheta = 0;
            
            {
                std::lock_guard<std::mutex> lock(slamMutex);
                if (robot.validPosition) {
                    rx = robot.x;
                    ry = robot.y;
                    rtheta = robot.theta;
                    validPos = true;
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(gridMutex);
                if (!grid.combinedGrid.empty()) {
                    cv::cvtColor(grid.combinedGrid, display, cv::COLOR_GRAY2BGR);
                } else {
                    display = cv::Mat::zeros(grid.height, grid.width, CV_8UC3);
                }
            }
            
            if (validPos) {
                int gx, gy;
                if (grid.worldToGrid(rx, ry, gx, gy)) {
                    cv::circle(display, cv::Point(gx, gy), 5, cv::Scalar(0, 0, 255), -1);
                    cv::line(display, cv::Point(gx, gy), 
                            cv::Point(gx + 10 * cos(rtheta), gy + 10 * sin(rtheta)),
                            cv::Scalar(0, 0, 255), 2);
                }
            }
            
            cv::imshow("Navigation Grid with Robot Position", display);
            cv::waitKey(100);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (const std::exception& e) {
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

    // Check if file exists before proceeding
    std::ifstream testFile(mapFilePath);
    if (!testFile.good()) {
        std::cerr << "Error: Cannot access map file: " << mapFilePath << std::endl;
        std::cerr << "Please check the file path and permissions." << std::endl;
        return 1;
    }
    testFile.close();
    
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
