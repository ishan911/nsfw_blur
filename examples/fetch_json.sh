#!/bin/bash

# JSON Fetching Script for Web Server
# This script provides various ways to fetch JSON data

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/fetch_json_examples.py"

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --url <URL>         Fetch JSON from a specific URL"
    echo "  --file <FILE>       Load JSON from a local file"
    echo "  --save <FILE>       Save JSON data to a file"
    echo "  --post <URL>        Send POST request with JSON data"
    echo "  --headers <HEADERS> Custom headers (JSON format)"
    echo "  --data <DATA>       JSON data for POST requests"
    echo ""
    echo "Examples:"
    echo "  $0 --url https://jsonplaceholder.typicode.com/posts/1"
    echo "  $0 --file data/config.json"
    echo "  $0 --post https://api.example.com/data --data '{\"key\":\"value\"}'"
    echo ""
}

# Function to check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is not installed or not in PATH"
        exit 1
    fi
}

# Function to check if required packages are installed
check_dependencies() {
    python3 -c "import requests, json" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing required Python packages..."
        pip3 install requests
    fi
}

# Function to fetch JSON from URL
fetch_from_url() {
    local url="$1"
    local headers="$2"
    
    if [ -n "$headers" ]; then
        python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
from fetch_json_examples import fetch_json_with_headers
import json
result = fetch_json_with_headers('$url', json.loads('$headers'))
if result:
    print(json.dumps(result, indent=2))
"
    else
        python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
from fetch_json_examples import fetch_json_from_url_requests
import json
result = fetch_json_from_url_requests('$url')
if result:
    print(json.dumps(result, indent=2))
"
    fi
}

# Function to load JSON from file
load_from_file() {
    local file="$1"
    python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
from fetch_json_examples import fetch_json_from_file
import json
result = fetch_json_from_file('$file')
if result:
    print(json.dumps(result, indent=2))
"
}

# Function to save JSON to file
save_to_file() {
    local file="$1"
    local data="$2"
    python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
from fetch_json_examples import save_json_to_file
import json
data = json.loads('$data')
save_json_to_file(data, '$file')
"
}

# Function to send POST request
send_post_request() {
    local url="$1"
    local data="$2"
    python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
from fetch_json_examples import fetch_json_post
import json
result = fetch_json_post('$url', json_data=json.loads('$data'))
if result:
    print(json.dumps(result, indent=2))
"
}

# Main script logic
main() {
    # Check dependencies
    check_python
    check_dependencies
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_usage
                exit 0
                ;;
            --url)
                if [ -z "$2" ]; then
                    echo "Error: URL is required for --url option"
                    exit 1
                fi
                URL="$2"
                shift 2
                ;;
            --file)
                if [ -z "$2" ]; then
                    echo "Error: File path is required for --file option"
                    exit 1
                fi
                FILE="$2"
                shift 2
                ;;
            --save)
                if [ -z "$2" ]; then
                    echo "Error: File path is required for --save option"
                    exit 1
                fi
                SAVE_FILE="$2"
                shift 2
                ;;
            --post)
                if [ -z "$2" ]; then
                    echo "Error: URL is required for --post option"
                    exit 1
                fi
                POST_URL="$2"
                shift 2
                ;;
            --headers)
                if [ -z "$2" ]; then
                    echo "Error: Headers JSON is required for --headers option"
                    exit 1
                fi
                HEADERS="$2"
                shift 2
                ;;
            --data)
                if [ -z "$2" ]; then
                    echo "Error: JSON data is required for --data option"
                    exit 1
                fi
                JSON_DATA="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute based on provided options
    if [ -n "$URL" ]; then
        echo "Fetching JSON from: $URL"
        fetch_from_url "$URL" "$HEADERS"
    elif [ -n "$FILE" ]; then
        echo "Loading JSON from file: $FILE"
        load_from_file "$FILE"
    elif [ -n "$POST_URL" ]; then
        if [ -z "$JSON_DATA" ]; then
            echo "Error: --data is required for POST requests"
            exit 1
        fi
        echo "Sending POST request to: $POST_URL"
        send_post_request "$POST_URL" "$JSON_DATA"
    elif [ -n "$SAVE_FILE" ]; then
        if [ -z "$JSON_DATA" ]; then
            echo "Error: --data is required for --save option"
            exit 1
        fi
        echo "Saving JSON to file: $SAVE_FILE"
        save_to_file "$SAVE_FILE" "$JSON_DATA"
    else
        echo "Running all examples..."
        python3 "$PYTHON_SCRIPT"
    fi
}

# Run main function with all arguments
main "$@" 