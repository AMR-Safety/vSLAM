#include "occupancy_grid.h"
using namespace cv;

OccupancyGrid::OccupancyGrid(GridMeta meta_) : meta(meta_) {
    grid = Mat::zeros(meta.height, meta.width, CV_8U);
}

void OccupancyGrid::markOccupied(float x, float y) {
    int i = worldToGridY(y);
    int j = worldToGridX(x);
    if (i >= 0 && i < meta.height && j >= 0 && j < meta.width)
        grid.at<uchar>(i, j) = 255;
}

void OccupancyGrid::clear() {
    grid.setTo(Scalar(0));
}

void OccupancyGrid::inflate(int radius) {
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2*radius+1, 2*radius+1));
    dilate(grid, grid, element);
}

cv::Mat OccupancyGrid::getGrid() const {
    return grid;
}

int OccupancyGrid::worldToGridX(float x) const {
    return static_cast<int>((x - meta.x_min) / meta.resolution);
}

int OccupancyGrid::worldToGridY(float y) const {
    return static_cast<int>((y - meta.y_min) / meta.resolution);
}
