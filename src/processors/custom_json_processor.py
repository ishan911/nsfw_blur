#!/usr/bin/env python3
"""
Custom JSON Image Processor Class

This module contains the CustomJSONImageProcessor class that provides
image processing functionality for images specified in JSON format.
"""

import os
import json
import requests
from urllib.parse import urljoin, urlparse

from .base_processor import ImageProcessor


class CustomJSONImageProcessor(ImageProcessor):
    def __init__(self, json_url=None, json_file=None, base_url=None, model_path='models/640m.onnx', database_path='data/processed_images.json'):
        """Initialize custom JSON image processor."""
        super().__init__(model_path, database_path)
        self.json_url = json_url
        self.json_file = json_file
        self.base_url = base_url.rstrip('/') if base_url else None
        
    def fetch_json_data(self):
        """Fetch JSON data from URL or load from file."""
        try:
            if self.json_url:
                print(f"Fetching JSON from: {self.json_url}")
                response = requests.get(self.json_url, timeout=30)
                response.raise_for_status()
                return response.json()
            elif self.json_file:
                print(f"Loading JSON from file: {self.json_file}")
                with open(self.json_file, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError("Either json_url or json_file must be provided")
        except Exception as e:
            print(f"Error fetching/loading JSON data: {e}")
            return []
    
    def extract_image_urls(self, item):
        """Extract image URLs from a JSON item."""
        image_urls = []
        
        # Handle screenshot_url (relative path)
        if item.get('screenshot_full_url'):
            screenshot_url = item['screenshot_full_url']
            if screenshot_url.startswith('/'):
                # Convert relative path to absolute URL
                if self.base_url:
                    full_url = urljoin(self.base_url, screenshot_url)
                    image_urls.append({
                        'type': 'screenshot_full_url',
                        'url': full_url,
                        'slug': item.get('slug', 'unknown')
                    })
            else:
                # Already absolute URL
                image_urls.append({
                    'type': 'screenshot_full_url',
                    'url': screenshot_url,
                    'slug': item.get('slug', 'unknown')
                })
        
        # Handle mobile_thumb (absolute URL)
        if item.get('mobile_thumb'):
            image_urls.append({
                'type': 'mobile_thumb',
                'url': item['mobile_thumb'],
                'slug': item.get('slug', 'unknown')
            })
        
        # Handle review_image (absolute URL)
        if item.get('review_full_image'):
            review_full_url = item['review_full_image']
            if review_full_url.startswith('/'):
                # Convert relative path to absolute URL
                if self.base_url:
                    full_url = urljoin(self.base_url, review_full_url)
                    image_urls.append({
                        'type': 'review_full_image',
                        'url': full_url,
                        'slug': item.get('slug', 'unknown')
                    })
            else:
                # Already absolute URL
                image_urls.append({
                    'type': 'review_full_image',
                    'url': review_full_url,
                    'slug': item.get('slug', 'unknown')
                })
        
        return image_urls
    
    def download_image(self, image_url, local_path):
        """Download an image from URL to local path."""
        try:
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
            return False
    
    def get_image_filename(self, image_url, slug, image_type):
        """Generate a filename for the downloaded image."""
        parsed_url = urlparse(image_url)
        original_filename = os.path.basename(parsed_url.path)
        
        if not original_filename or '.' not in original_filename:
            # Fallback to type-based filename
            original_filename = f"{image_type}.jpg"
        
        return f"{slug}_{image_type}_{original_filename}"
    
    def process_custom_json_images(self, output_dir="data/custom_processed", pixel_size=10, 
                                 force=False, download_only=False):
        """Process images from custom JSON format."""
        print("Fetching JSON data...")
        json_data = self.fetch_json_data()
        
        if not json_data:
            print("No data found or error loading JSON.")
            return
        
        print(f"Found {len(json_data)} items in JSON data")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        downloaded_count = 0
        skipped_count = 0
        error_count = 0
        total_images = 0
        
        for item in json_data:
            slug = item.get('slug', 'unknown')
            print(f"\nProcessing item: {slug}")
            
            # Extract all image URLs from this item
            image_urls = self.extract_image_urls(item)
            total_images += len(image_urls)
            
            for image_info in image_urls:
                image_url = image_info['url']
                image_type = image_info['type']
                filename = self.get_image_filename(image_url, slug, image_type)
                local_path = os.path.join("data/custom_downloads", filename)
                # Remove 'processed_' prefix from output filename
                output_filename = filename
                if output_filename.startswith('processed_'):
                    output_filename = output_filename[10:]  # Remove 'processed_' prefix
                output_path = os.path.join(output_dir, output_filename)
                
                print(f"  Processing {image_type}: {image_url}")
                
                # Download the image
                if not os.path.exists(local_path) or force:
                    print(f"    Downloading: {image_url}")
                    if self.download_image(image_url, local_path):
                        downloaded_count += 1
                        print(f"    Downloaded to: {local_path}")
                    else:
                        error_count += 1
                        continue
                else:
                    print(f"    Already downloaded: {local_path}")
                
                if download_only:
                    continue
                
                # Process the image
                try:
                    result = self.process_single_image(
                        input_path=local_path,
                        output_path=output_path,
                        pixel_size=pixel_size,
                        force=force,
                        image_type=image_type
                    )
                    if result:
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"    Error processing {image_type} for {slug}: {str(e)}")
                    error_count += 1
        
        print(f"\nCustom JSON processing complete:")
        print(f"  Total items: {len(json_data)}")
        print(f"  Total images found: {total_images}")
        print(f"  Downloaded: {downloaded_count} images")
        print(f"  Processed: {processed_count} images")
        print(f"  Skipped: {skipped_count} images")
        print(f"  Errors: {error_count} images") 