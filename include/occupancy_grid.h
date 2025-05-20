#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

struct GridMeta {
    float resolution;      // meters per cell
    float x_min, y_min;    // world coords of (0,0) in grid
    int width, height;
};

class OccupancyGrid {
public:
    OccupancyGrid(GridMeta meta);
    void markOccupied(float x, float y);          // World coords
    void clear();                                 // Reset dynamic layer
    void inflate(int radius);                     // Obstacle inflation
    cv::Mat getGrid() const;                      // For visualization or fusion
    int worldToGridX(float x) const;
    int worldToGridY(float y) const;

private:
    GridMeta meta;
    cv::Mat grid; // 8-bit single channel grid (0 = free, 255 = occupied)
};
