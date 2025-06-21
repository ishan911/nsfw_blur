#!/usr/bin/env python3
"""
Standalone JSON Fetcher Script for Web Servers
This script can be easily deployed and executed on web servers.
"""

import json
import sys
import os
from pathlib import Path

# Try to import requests, fall back to urllib if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    try:
        from urllib.request import urlopen
        from urllib.error import URLError
    except ImportError:
        print("Error: Neither requests nor urllib available")
        sys.exit(1)

def fetch_json_from_url(url, headers=None):
    """Fetch JSON from URL using available HTTP library."""
    try:
        if REQUESTS_AVAILABLE:
            response = requests.get(url, headers=headers or {}, timeout=30)
            response.raise_for_status()
            return response.json()
        else:
            # Fallback to urllib
            req = urlopen(url, timeout=30)
            data = req.read()
            return json.loads(data.decode('utf-8'))
    except Exception as e:
        print(f"Error fetching JSON from {url}: {e}")
        return None

def load_json_from_file(file_path):
    """Load JSON from local file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None

def save_json_to_file(data, file_path):
    """Save JSON data to file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"JSON saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False

def post_json_to_url(url, data, headers=None):
    """Send POST request with JSON data."""
    if not REQUESTS_AVAILABLE:
        print("Error: POST requests require the requests library")
        return None
    
    try:
        response = requests.post(url, json=data, headers=headers or {}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error posting to {url}: {e}")
        return None

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 json_fetcher.py get <URL> [output_file]")
        print("  python3 json_fetcher.py file <file_path>")
        print("  python3 json_fetcher.py post <URL> <json_data> [output_file]")
        print("  python3 json_fetcher.py save <json_data> <file_path>")
        print("")
        print("Examples:")
        print("  python3 json_fetcher.py get https://jsonplaceholder.typicode.com/posts/1")
        print("  python3 json_fetcher.py file data/config.json")
        print("  python3 json_fetcher.py post https://api.example.com/data '{\"key\":\"value\"}'")
        print("  python3 json_fetcher.py save '{\"name\":\"test\"}' output.json")
        return

    command = sys.argv[1].lower()
    
    if command == "get":
        if len(sys.argv) < 3:
            print("Error: URL required for get command")
            return
        
        url = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        result = fetch_json_from_url(url)
        if result:
            if output_file:
                save_json_to_file(result, output_file)
            else:
                print(json.dumps(result, indent=2))
    
    elif command == "file":
        if len(sys.argv) < 3:
            print("Error: File path required for file command")
            return
        
        file_path = sys.argv[2]
        result = load_json_from_file(file_path)
        if result:
            print(json.dumps(result, indent=2))
    
    elif command == "post":
        if len(sys.argv) < 4:
            print("Error: URL and JSON data required for post command")
            return
        
        url = sys.argv[2]
        json_data_str = sys.argv[3]
        output_file = sys.argv[4] if len(sys.argv) > 4 else None
        
        try:
            json_data = json.loads(json_data_str)
        except json.JSONDecodeError:
            print("Error: Invalid JSON data")
            return
        
        result = post_json_to_url(url, json_data)
        if result:
            if output_file:
                save_json_to_file(result, output_file)
            else:
                print(json.dumps(result, indent=2))
    
    elif command == "save":
        if len(sys.argv) < 4:
            print("Error: JSON data and file path required for save command")
            return
        
        json_data_str = sys.argv[2]
        file_path = sys.argv[3]
        
        try:
            json_data = json.loads(json_data_str)
        except json.JSONDecodeError:
            print("Error: Invalid JSON data")
            return
        
        save_json_to_file(json_data, file_path)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'get', 'file', 'post', or 'save'")

if __name__ == "__main__":
    main() 