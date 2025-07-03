#!/bin/bash

# Directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to map a host path to a container path
hostToContainer() {
    local HOST_PATH=$(realpath "$1")
    local RELATIVE_PATH="${HOST_PATH#$SCRIPT_DIR/}"
    echo "/kalibr_ws/$RELATIVE_PATH"
}

# Parse command and arguments
COMMAND="$1"
shift
ARGS=""

# Map host paths to container paths for each argument
for arg in "$@"; do
    if [[ "$arg" == "--"* ]] && [[ "$arg" == *"="* ]]; then
        # Handle --option=path format
        option_name="${arg%%=*}"
        path_value="${arg#*=}"
        if [[ -e "$path_value" ]]; then
            # It's a file or directory that exists
            container_path=$(hostToContainer "$path_value")
            ARGS="$ARGS $option_name=$container_path"
        else
            # Not a path or doesn't exist
            ARGS="$ARGS $arg"
        fi
    elif [[ "$arg" == "--"* ]]; then
        # Regular option
        ARGS="$ARGS $arg"
    elif [[ -e "$arg" ]]; then
        # It's a file or directory
        container_path=$(hostToContainer "$arg")
        ARGS="$ARGS $container_path"
    else
        # Regular argument
        ARGS="$ARGS $arg"
    fi
done

# Execute the command in the container
case "$COMMAND" in
    "calibrate_cameras")
        docker exec -it kalibr_container bash -c "source /kalibr_ws/devel/setup.bash && cd /kalibr_ws && kalibr_calibrate_cameras $ARGS"
        ;;
    "calibrate_imu_camera")
        docker exec -it kalibr_container bash -c "source /kalibr_ws/devel/setup.bash && cd /kalibr_ws && kalibr_calibrate_imu_camera $ARGS"
        ;;
    "bagcreater")
        docker exec -it kalibr_container bash -c "source /kalibr_ws/devel/setup.bash && cd /kalibr_ws && kalibr_bagcreater $ARGS"
        ;;
    "bagextractor")
        docker exec -it kalibr_container bash -c "source /kalibr_ws/devel/setup.bash && cd /kalibr_ws && kalibr_bagextractor $ARGS"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Available commands: calibrate_cameras, calibrate_imu_camera, bagcreater, bagextractor"
        exit 1
        ;;
esac
