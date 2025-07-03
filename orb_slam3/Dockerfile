# Use base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Essential system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl unzip pkg-config \
    libeigen3-dev libboost-all-dev libtbb-dev \
    libyaml-cpp-dev libglew-dev libopencv-dev \
    qtbase5-dev qt5-qmake qtdeclarative5-dev \
    libqt5svg5-dev libpcl-dev libusb-1.0-0-dev \
    libgl1-mesa-dev libglu1-mesa-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libxkbcommon-dev libxi-dev libxxf86vm-dev \
    libxcursor-dev libxinerama-dev libxrandr-dev \
    libglfw3-dev libepoxy-dev libpython3-dev \
    python3 python3-pip && \
    apt-get clean

# Python dependencies
RUN pip3 install numpy opencv-python

# Clone and build RealSense SDK
RUN git clone https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    mkdir build && cd build && \
    cmake .. -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false && \
    make -j$(nproc) && make install && \
    ldconfig

# Clone and build Pangolin (with GUI support)
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake .. -DBUILD_PANGOLIN_GUI=ON -DBUILD_PANGOLIN_VIEWER=ON && \
    make -j$(nproc) && make install && \
    ldconfig

# Set working directory
WORKDIR /root

# Copy ORB_SLAM3 source code
COPY ./ORB_SLAM3 /root/ORB_SLAM3

# Build ORB_SLAM3
WORKDIR /root/ORB_SLAM3
RUN chmod +x build.sh && ./build.sh

# Set default command
CMD ["/bin/bash"]

