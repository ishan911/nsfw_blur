#!/usr/bin/env python3
"""
Examples of how to fetch JSON data in Python.
"""

import json
import requests
from pathlib import Path
from urllib.request import urlopen
import urllib.parse

def fetch_json_from_url_requests(url):
    """
    Fetch JSON from a URL using the requests library (recommended).
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching JSON from {url}: {e}")
        return None

def fetch_json_from_url_urllib(url):
    """
    Fetch JSON from a URL using urllib (built-in, no external dependencies).
    """
    try:
        with urlopen(url) as response:
            data = response.read()
            return json.loads(data.decode('utf-8'))
    except Exception as e:
        print(f"Error fetching JSON from {url}: {e}")
        return None

def fetch_json_from_file(file_path):
    """
    Load JSON from a local file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {file_path}: {e}")
        return None

def fetch_json_with_headers(url, headers=None):
    """
    Fetch JSON with custom headers (useful for APIs requiring authentication).
    """
    try:
        response = requests.get(url, headers=headers or {})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching JSON from {url}: {e}")
        return None

def fetch_json_post(url, data=None, json_data=None):
    """
    Fetch JSON using POST request (useful for APIs that require POST).
    """
    try:
        response = requests.post(url, data=data, json=json_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error posting to {url}: {e}")
        return None

def save_json_to_file(data, file_path):
    """
    Save JSON data to a file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"JSON saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")

# Example usage
if __name__ == "__main__":
    # Example 1: Fetch from a public API
    print("=== Example 1: Fetching from JSONPlaceholder API ===")
    api_url = "https://jsonplaceholder.typicode.com/posts/1"
    json_data = fetch_json_from_url_requests(api_url)
    if json_data:
        print(f"Title: {json_data.get('title', 'N/A')}")
        print(f"Body: {json_data.get('body', 'N/A')[:50]}...")
    
    # Example 2: Fetch with headers (GitHub API example)
    print("\n=== Example 2: Fetching with headers ===")
    github_url = "https://api.github.com/users/octocat"
    headers = {
        'User-Agent': 'MyApp/1.0',
        'Accept': 'application/vnd.github.v3+json'
    }
    github_data = fetch_json_with_headers(github_url, headers)
    if github_data:
        print(f"Username: {github_data.get('login', 'N/A')}")
        print(f"Name: {github_data.get('name', 'N/A')}")
    
    # Example 3: Create and save sample JSON
    print("\n=== Example 3: Creating and saving JSON ===")
    sample_data = {
        "name": "Sample User",
        "email": "user@example.com",
        "settings": {
            "theme": "dark",
            "notifications": True
        },
        "tags": ["python", "json", "api"]
    }
    
    # Save to file
    output_file = Path("examples/sample_data.json")
    save_json_to_file(sample_data, output_file)
    
    # Read it back
    loaded_data = fetch_json_from_file(output_file)
    if loaded_data:
        print(f"Loaded data: {loaded_data['name']}")
    
    # Example 4: POST request example
    print("\n=== Example 4: POST request example ===")
    post_url = "https://jsonplaceholder.typicode.com/posts"
    post_data = {
        "title": "Test Post",
        "body": "This is a test post",
        "userId": 1
    }
    post_response = fetch_json_post(post_url, json_data=post_data)
    if post_response:
        print(f"Created post with ID: {post_response.get('id', 'N/A')}") 