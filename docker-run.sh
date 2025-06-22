#!/bin/bash

# Simple Docker runtime script for blur API using python:3
# This script builds and runs the application in a Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Blur API Docker Runtime (Python 3) ===${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Build the Docker image
echo -e "${YELLOW}Building Docker image with python:3...${NC}"
docker build -t blur-api:latest .

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Docker image${NC}"
    exit 1
fi

echo -e "${GREEN}Docker image built successfully!${NC}"

# Function to run the container
run_container() {
    local command="$1"
    local input="$2"
    local output="$3"
    local additional_args="$4"
    
    echo -e "${YELLOW}Running container with:${NC}"
    echo -e "  Command: ${command}"
    echo -e "  Input: ${input}"
    echo -e "  Output: ${output}"
    echo -e "  Additional args: ${additional_args}"
    
    docker run --rm \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/wp-content:/app/wp-content" \
        -v "$(pwd)/screenshots:/app/screenshots" \
        -v "$(pwd)/models:/app/models" \
        blur-api:latest \
        python main.py \
        ${command} \
        ${input} \
        ${output} \
        ${additional_args}
}

# Function to run custom command
run_custom_command() {
    local full_command="$1"
    
    echo -e "${YELLOW}Running custom command: ${full_command}${NC}"
    
    docker run --rm \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/wp-content:/app/wp-content" \
        -v "$(pwd)/screenshots:/app/screenshots" \
        -v "$(pwd)/models:/app/models" \
        blur-api:latest \
        python main.py \
        ${full_command}
}

# Interactive mode
echo -e "${BLUE}Choose an option:${NC}"
echo "1) Process images from data/input to data/output"
echo "2) Process WordPress uploads"
echo "3) Process screenshots"
echo "4) Sliding window - single image"
echo "5) Sliding window - batch processing"
echo "6) Sliding window - WordPress processing"
echo "7) Sliding window - custom JSON"
echo "8) Custom JSON processing"
echo "9) Single image processing"
echo "10) List available parts"
echo "11) Show processing stats"
echo "12) Cleanup orphaned records"
echo "13) Custom command"
echo "14) Interactive shell"

read -p "Enter your choice (1-14): " choice

case $choice in
    1)
        echo -e "${YELLOW}Processing images from data/input...${NC}"
        run_container "batch" "data/input" "data/output" ""
        ;;
    2)
        echo -e "${YELLOW}Processing WordPress uploads...${NC}"
        run_container "batch" "wp-content/uploads" "data/output" "--wordpress"
        ;;
    3)
        echo -e "${YELLOW}Processing screenshots...${NC}"
        run_container "batch" "screenshots" "data/output" ""
        ;;
    4)
        read -p "Enter input image path: " input_image
        read -p "Enter output image path: " output_image
        read -p "Enter window size (default 640): " window_size
        read -p "Enter stride (default 320): " stride
        window_size=${window_size:-640}
        stride=${stride:-320}
        run_container "sliding-single" "${input_image}" "${output_image}" "--window-size ${window_size} --stride ${stride}"
        ;;
    5)
        read -p "Enter input directory (default data/input): " input_dir
        read -p "Enter output directory (default data/output): " output_dir
        read -p "Enter window size (default 640): " window_size
        read -p "Enter stride (default 320): " stride
        input_dir=${input_dir:-data/input}
        output_dir=${output_dir:-data/output}
        window_size=${window_size:-640}
        stride=${stride:-320}
        run_container "sliding-batch" "${input_dir}" "${output_dir}" "--window-size ${window_size} --stride ${stride}"
        ;;
    6)
        read -p "Enter JSON file path or URL: " json_input
        read -p "Enter output directory (default data/wordpress_processed): " output_dir
        read -p "Enter window size (default 640): " window_size
        read -p "Enter stride (default 320): " stride
        read -p "Enter base URL (optional): " base_url
        output_dir=${output_dir:-data/wordpress_processed}
        window_size=${window_size:-640}
        stride=${stride:-320}
        additional_args="--window-size ${window_size} --stride ${stride}"
        if [ ! -z "$base_url" ]; then
            additional_args="${additional_args} --base-url ${base_url}"
        fi
        run_container "sliding-wordpress" "${json_input}" "${output_dir}" "${additional_args}"
        ;;
    7)
        read -p "Enter JSON file path or URL: " json_input
        read -p "Enter output directory (default data/custom_processed): " output_dir
        read -p "Enter window size (default 640): " window_size
        read -p "Enter stride (default 320): " stride
        read -p "Enter base URL (optional): " base_url
        output_dir=${output_dir:-data/custom_processed}
        window_size=${window_size:-640}
        stride=${stride:-320}
        additional_args="--window-size ${window_size} --stride ${stride}"
        if [ ! -z "$base_url" ]; then
            additional_args="${additional_args} --base-url ${base_url}"
        fi
        run_container "sliding-json" "${json_input}" "${output_dir}" "${additional_args}"
        ;;
    8)
        read -p "Enter JSON file path or URL: " json_input
        read -p "Enter output directory (default data/custom_processed): " output_dir
        read -p "Enter base URL (optional): " base_url
        output_dir=${output_dir:-data/custom_processed}
        additional_args=""
        if [ ! -z "$base_url" ]; then
            additional_args="--base-url ${base_url}"
        fi
        run_container "json" "${json_input}" "${output_dir}" "${additional_args}"
        ;;
    9)
        read -p "Enter input image path: " input_image
        read -p "Enter output image path: " output_image
        run_container "single" "${input_image}" "${output_image}" ""
        ;;
    10)
        run_custom_command "list-parts"
        ;;
    11)
        run_custom_command "stats"
        ;;
    12)
        run_custom_command "cleanup"
        ;;
    13)
        read -p "Enter custom command (e.g., 'sliding-wordpress data/images.json data/output --window-size 512 --stride 256'): " custom_cmd
        run_custom_command "${custom_cmd}"
        ;;
    14)
        echo -e "${YELLOW}Starting interactive shell...${NC}"
        docker run --rm -it \
            -v "$(pwd)/data:/app/data" \
            -v "$(pwd)/wp-content:/app/wp-content" \
            -v "$(pwd)/screenshots:/app/screenshots" \
            -v "$(pwd)/models:/app/models" \
            blur-api:latest \
            /bin/bash
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Container execution completed!${NC}" 