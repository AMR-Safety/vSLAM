#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

/**
 * @brief Class that detects dynamic obstacles using a RealSense D435i depth camera
 * and maintains a 2D occupancy grid for navigation.
 */
class DynamicObstacleDetector {
public:
    /**
     * @brief Constructor
     * @param resolution Grid resolution in meters per cell
     * @param width_meters Width of the grid in meters
     * @param height_meters Height of the grid in meters
     * @param x_origin_meters X coordinate of grid origin in meters
     * @param y_origin_meters Y coordinate of grid origin in meters
     */
    DynamicObstacleDetector(
        float resolution = 0.05f,
        float width_meters = 10.0f,
        float height_meters = 10.0f,
        float x_origin_meters = -5.0f,
        float y_origin_meters = -5.0f
    ) : resolution_(resolution),
        width_meters_(width_meters),
        height_meters_(height_meters),
        x_origin_(x_origin_meters),
        y_origin_(y_origin_meters),
        min_height_(0.1f),  // Minimum height to consider as obstacle (in meters)
        max_height_(1.5f),  // Maximum height to consider as obstacle (in meters)
        min_depth_(0.3f),   // Minimum depth to consider valid (in meters)
        max_depth_(5.0f),   // Maximum depth to consider valid (in meters)
        obstacle_radius_(0.2f),  // Radius to inflate obstacles (in meters)
        running_(false),
        pose_provider_(nullptr) {
        
        // Calculate grid dimensions
        width_cells_ = static_cast<int>(std::ceil(width_meters_ / resolution_));
        height_cells_ = static_cast<int>(std::ceil(height_meters_ / resolution_));
        
        // Initialize grid with all cells free
        grid_.resize(width_cells_ * height_cells_, 0);
        
        // Create inflation kernel for obstacle inflation
        int kernel_radius = static_cast<int>(std::ceil(obstacle_radius_ / resolution_));
        inflation_kernel_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * kernel_radius + 1, 2 * kernel_radius + 1)
        );
        
        std::cout << "Created dynamic obstacle grid with dimensions: " 
                  << width_cells_ << "x" << height_cells_ << " cells" << std::endl;
        std::cout << "Grid covers area: " << width_meters_ << "x" << height_meters_ 
                  << " meters at " << resolution_ << " meters/cell" << std::endl;
    }
    
    /**
     * @brief Destructor - ensures thread is stopped
     */
    ~DynamicObstacleDetector() {
        stop();
    }
    
    /**
     * @brief Set a callback function to provide camera poses
     * @param provider Function that returns the current camera pose as 4x4 matrix
     */
    void setPoseProvider(std::function<Eigen::Matrix4f()> provider) {
        pose_provider_ = provider;
    }
    
    /**
     * @brief Start the obstacle detection in a separate thread
     * @return True if successfully started
     */
    bool start() {
        if (running_) {
            std::cout << "Detection already running" << std::endl;
            return false;
        }
        
        running_ = true;
        detector_thread_ = std::thread(&DynamicObstacleDetector::detectionLoop, this);
        return true;
    }
    
    /**
     * @brief Stop the obstacle detection thread
     */
    void stop() {
        running_ = false;
        if (detector_thread_.joinable()) {
            detector_thread_.join();
        }
    }
    
    /**
     * @brief Get a copy of the current occupancy grid
     * @return Vector of grid cells (0 = free, 255 = occupied)
     */
    std::vector<uint8_t> getGrid() const {
        std::lock_guard<std::mutex> lock(grid_mutex_);
        return grid_;
    }
    
    /**
     * @brief Check if a world position is occupied
     * @param x X coordinate in world frame (meters)
     * @param y Y coordinate in world frame (meters)
     * @return True if occupied, false if free
     */
    bool isOccupied(float x, float y) const {
        int grid_x, grid_y;
        if (!worldToGrid(x, y, grid_x, grid_y)) {
            return true;  // Outside grid boundaries is considered occupied
        }
        
        std::lock_guard<std::mutex> lock(grid_mutex_);
        return grid_[grid_y * width_cells_ + grid_x] > 0;
    }
    
    /**
     * @brief Save the current grid as an image
     * @param filename Output filename
     * @return True if successfully saved
     */
    bool saveGridAsImage(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(grid_mutex_);
        cv::Mat image(height_cells_, width_cells_, CV_8UC1, const_cast<uint8_t*>(grid_.data()));
        return cv::imwrite(filename, image);
    }
    
    /**
     * @brief Convert world coordinates to grid cell indices
     * @param x X coordinate in world frame (meters)
     * @param y Y coordinate in world frame (meters)
     * @param grid_x Output grid x index
     * @param grid_y Output grid y index
     * @return True if the resulting indices are within grid bounds
     */
    bool worldToGrid(float x, float y, int& grid_x, int& grid_y) const {
        grid_x = static_cast<int>((x - x_origin_) / resolution_);
        grid_y = static_cast<int>((y - y_origin_) / resolution_);
        
        return grid_x >= 0 && grid_x < width_cells_ && grid_y >= 0 && grid_y < height_cells_;
    }
    
    /**
     * @brief Convert grid cell indices to world coordinates
     * @param grid_x Grid x index
     * @param grid_y Grid y index
     * @param x Output x coordinate in world frame (meters)
     * @param y Output y coordinate in world frame (meters)
     */
    void gridToWorld(int grid_x, int grid_y, float& x, float& y) const {
        x = x_origin_ + (grid_x + 0.5f) * resolution_;
        y = y_origin_ + (grid_y + 0.5f) * resolution_;
    }
    
    /**
     * @brief Get grid dimensions and parameters
     */
    int getWidthCells() const { return width_cells_; }
    int getHeightCells() const { return height_cells_; }
    float getResolution() const { return resolution_; }
    float getWidthMeters() const { return width_meters_; }
    float getHeightMeters() const { return height_meters_; }
    
    /**
     * @brief Process a single depth frame manually
     * @param depth_frame RealSense depth frame
     * @param camera_pose 4x4 transformation matrix from camera to world
     */
    void processDepthFrame(const rs2::depth_frame& depth_frame, const Eigen::Matrix4f& camera_pose) {
        // Skip if the frame is not valid
        if (!depth_frame)
            return;
        
        // Get depth frame dimensions
        const int width = depth_frame.get_width();
        const int height = depth_frame.get_height();
        
        // Get camera intrinsic parameters
        rs2_intrinsics intrinsics = depth_frame.get_profile()
                                   .as<rs2::video_stream_profile>()
                                   .get_intrinsics();
        
        // Reset the grid
        {
            std::lock_guard<std::mutex> lock(grid_mutex_);
            std::fill(grid_.begin(), grid_.end(), 0);
        }
        
        // Create a temporary OpenCV matrix for the grid
        cv::Mat grid_mat(height_cells_, width_cells_, CV_8UC1);
        grid_mat.setTo(0);
        
        // Extract camera pose components
        Eigen::Matrix3f R_camera_to_world = camera_pose.block<3, 3>(0, 0);
        Eigen::Vector3f t_camera_to_world = camera_pose.block<3, 1>(0, 3);
        
        // Process depth frame - subsample for performance
        const int stride = 4;  // Only process every 4th pixel
        for (int v = 0; v < height; v += stride) {
            for (int u = 0; u < width; u += stride) {
                // Get depth in meters
                float depth = depth_frame.get_distance(u, v);
                
                // Skip invalid depths or out of range
                if (depth <= 0 || depth < min_depth_ || depth > max_depth_)
                    continue;
                
                // Calculate 3D point in camera coordinates
                float x_c = (u - intrinsics.ppx) / intrinsics.fx * depth;
                float y_c = (v - intrinsics.ppy) / intrinsics.fy * depth;
                float z_c = depth;
                
                // Transform to world coordinates
                Eigen::Vector3f pt_camera(x_c, y_c, z_c);
                Eigen::Vector3f pt_world = R_camera_to_world * pt_camera + t_camera_to_world;
                
                // Extract world coordinates
                float x_w = pt_world[0];
                float y_w = pt_world[1];
                float z_w = pt_world[2];
                
                // Filter by height - skip floor/ceiling points
                if (z_w < min_height_ || z_w > max_height_)
                    continue;
                
                // Convert to grid coordinates
                int grid_x, grid_y;
                if (worldToGrid(x_w, y_w, grid_x, grid_y)) {
                    // Mark as occupied
                    grid_mat.at<uint8_t>(grid_y, grid_x) = 255;
                }
            }
        }
        
        // Inflate obstacles for robot clearance
        cv::dilate(grid_mat, grid_mat, inflation_kernel_);
        
        // Update the grid with mutex protection
        {
            std::lock_guard<std::mutex> lock(grid_mutex_);
            std::memcpy(grid_.data(), grid_mat.data, grid_.size());
        }
    }
    
    /**
     * @brief Set the height range for obstacle detection
     * @param min_height Minimum height in meters
     * @param max_height Maximum height in meters
     */
    void setHeightRange(float min_height, float max_height) {
        min_height_ = min_height;
        max_height_ = max_height;
    }
    
    /**
     * @brief Set the depth range for obstacle detection
     * @param min_depth Minimum depth in meters
     * @param max_depth Maximum depth in meters
     */
    void setDepthRange(float min_depth, float max_depth) {
        min_depth_ = min_depth;
        max_depth_ = max_depth;
    }
    
    /**
     * @brief Set the obstacle inflation radius
     * @param radius Inflation radius in meters
     */
    void setObstacleRadius(float radius) {
        obstacle_radius_ = radius;
        
        // Update inflation kernel
        int kernel_radius = static_cast<int>(std::ceil(obstacle_radius_ / resolution_));
        inflation_kernel_ = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * kernel_radius + 1, 2 * kernel_radius + 1)
        );
    }

private:
    /**
     * @brief Main detection loop that runs in a separate thread
     */
    void detectionLoop() {
        try {
            // Initialize RealSense pipeline
            rs2::pipeline pipe;
            rs2::config cfg;
            
            // Configure depth stream
            cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
            
            // Start streaming
            rs2::pipeline_profile profile = pipe.start(cfg);
            
            // Wait for frames
            for (int i = 0; i < 30; i++) {
                pipe.wait_for_frames();
            }
            
            std::cout << "RealSense depth stream started" << std::endl;
            
            while (running_) {
                // Get frameset
                rs2::frameset frames = pipe.wait_for_frames();
                rs2::depth_frame depth_frame = frames.get_depth_frame();
                
                // Skip if no pose provider
                if (!pose_provider_) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                
                // Get current camera pose
                Eigen::Matrix4f camera_pose = pose_provider_();
                
                // Process the depth frame
                processDepthFrame(depth_frame, camera_pose);
                
                // Process at ~10Hz
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // Stop streaming
            pipe.stop();
        }
        catch (const rs2::error& e) {
            std::cerr << "RealSense error: " << e.what() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in detection thread: " << e.what() << std::endl;
        }
    }

private:
    // Grid parameters
    float resolution_;          // meters per cell
    float width_meters_;        // width of the grid in meters
    float height_meters_;       // height of the grid in meters
    float x_origin_;            // x coordinate of grid origin in meters
    float y_origin_;            // y coordinate of grid origin in meters
    
    // Grid dimensions
    int width_cells_;           // width of the grid in cells
    int height_cells_;          // height of the grid in cells
    
    // Grid data
    std::vector<uint8_t> grid_; // Occupancy grid (0 = free, 255 = occupied)
    
    // Filtering parameters
    float min_height_;          // minimum height for obstacles
    float max_height_;          // maximum height for obstacles
    float min_depth_;           // minimum depth for valid readings
    float max_depth_;           // maximum depth for valid readings
    float obstacle_radius_;     // obstacle inflation radius
    
    // Thread management
    std::thread detector_thread_;
    std::atomic<bool> running_;
    
    // Mutex for thread safety
    mutable std::mutex grid_mutex_;
    
    // Obstacle inflation kernel
    cv::Mat inflation_kernel_;
    
    // Camera pose provider function
    std::function<Eigen::Matrix4f()> pose_provider_;
};

/**
 * @brief Simple application to demonstrate the dynamic obstacle detector
 */
class DynamicObstacleApp {
public:
    DynamicObstacleApp() 
        : detector_(0.05f, 10.0f, 10.0f, -5.0f, -5.0f),
          running_(false) {
        
        // Set up pose provider
        detector_.setPoseProvider([this]() { return getCurrentPose(); });
        
        // Set detection parameters
        detector_.setHeightRange(0.1f, 1.5f);
        detector_.setDepthRange(0.3f, 5.0f);
        detector_.setObstacleRadius(0.2f);
    }
    
    /**
     * @brief Start the application
     */
    void start() {
        running_ = true;
        
        // Start the detector
        detector_.start();
        
        // Start visualization in a separate thread
        viz_thread_ = std::thread(&DynamicObstacleApp::visualizationLoop, this);
    }
    
    /**
     * @brief Stop the application
     */
    void stop() {
        running_ = false;
        
        // Stop the detector
        detector_.stop();
        
        // Wait for visualization thread to finish
        if (viz_thread_.joinable()) {
            viz_thread_.join();
        }
    }
    
    /**
     * @brief Run the application with keyboard control
     */
    void run() {
        start();
        
        std::cout << "Press 'q' to quit, 's' to save current grid as image" << std::endl;
        
        char key = 0;
        while (key != 'q') {
            std::cin >> key;
            
            if (key == 's') {
                std::string filename = "dynamic_grid_" + 
                                     std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + 
                                     ".png";
                detector_.saveGridAsImage(filename);
                std::cout << "Saved grid as " << filename << std::endl;
            }
        }
        
        stop();
    }

private:
    /**
     * @brief Get current camera pose
     * @return 4x4 transformation matrix from camera to world
     */
    Eigen::Matrix4f getCurrentPose() {
        // In a real application, this would come from a SLAM system
        // For this example, we'll use a simple pose
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        
        // Get current time for a simple moving camera example
        auto now = std::chrono::steady_clock::now();
        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()
        ).count();
        
        // Simple circular motion example
        float angle = static_cast<float>(time_ms % 10000) / 10000.0f * 2.0f * M_PI;
        float radius = 2.0f;
        
        // Set position
        pose(0, 3) = std::cos(angle) * radius;
        pose(1, 3) = std::sin(angle) * radius;
        pose(2, 3) = 1.0f;  // Camera is 1m above ground
        
        // Set orientation (always looking at the center)
        Eigen::Vector3f camera_pos(pose(0, 3), pose(1, 3), pose(2, 3));
        Eigen::Vector3f target(0.0f, 0.0f, 0.0f);
        Eigen::Vector3f up(0.0f, 0.0f, 1.0f);
        
        Eigen::Vector3f z_axis = (camera_pos - target).normalized();
        Eigen::Vector3f x_axis = up.cross(z_axis).normalized();
        Eigen::Vector3f y_axis = z_axis.cross(x_axis);
        
        pose.block<3, 1>(0, 0) = x_axis;
        pose.block<3, 1>(0, 1) = y_axis;
        pose.block<3, 1>(0, 2) = z_axis;
        
        return pose;
    }
    
    /**
     * @brief Visualization loop to display the grid
     */
    void visualizationLoop() {
        cv::namedWindow("Dynamic Obstacle Grid", cv::WINDOW_NORMAL);
        cv::resizeWindow("Dynamic Obstacle Grid", 800, 800);
        
        while (running_) {
            // Get current grid
            auto grid = detector_.getGrid();
            
            // Create visualization image
            cv::Mat display(detector_.getHeightCells(), detector_.getWidthCells(), CV_8UC3);
            
            // Convert grid to RGB for display
            for (int y = 0; y < detector_.getHeightCells(); y++) {
                for (int x = 0; x < detector_.getWidthCells(); x++) {
                    int idx = y * detector_.getWidthCells() + x;
                    if (grid[idx] > 0) {
                        // Obstacle
                        display.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);  // Red
                    } else {
                        // Free
                        display.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);  // White
                    }
                }
            }
            
            // Draw current camera position
            Eigen::Matrix4f pose = getCurrentPose();
            float x_w = pose(0, 3);
            float y_w = pose(1, 3);
            
            int grid_x, grid_y;
            if (detector_.worldToGrid(x_w, y_w, grid_x, grid_y)) {
                // Draw camera position
                cv::circle(display, cv::Point(grid_x, grid_y), 5, cv::Scalar(0, 255, 0), -1);
                
                // Draw camera direction
                Eigen::Vector3f forward = -pose.block<3, 1>(0, 2);
                int end_x = grid_x + static_cast<int>(forward[0] * 20);
                int end_y = grid_y + static_cast<int>(forward[1] * 20);
                cv::line(display, cv::Point(grid_x, grid_y), cv::Point(end_x, end_y), 
                         cv::Scalar(0, 255, 0), 2);
            }
            
            // Show visualization
            cv::imshow("Dynamic Obstacle Grid", display);
            cv::waitKey(1);
            
            // Update at 10 Hz
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        cv::destroyWindow("Dynamic Obstacle Grid");
    }

private:
    DynamicObstacleDetector detector_;
    std::thread viz_thread_;
    std::atomic<bool> running_;
};

/**
 * @brief Main function
 */
int main(int argc, char** argv) {
    try {
        DynamicObstacleApp app;
        app.run();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}