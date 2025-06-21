#!/usr/bin/env python3
"""
Image Processing Script with Record Keeping
This script demonstrates how to use the ImageBlurrer class with database tracking.
Can be used as a standalone script with command-line arguments.
"""

import os
import sys
import json
import hashlib
import argparse
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.blurrer import ImageBlurrer, SlidingWindowBlurrer

class ImageProcessor:
    def __init__(self, model_path='models/640m.onnx', database_path='data/processed_images.json'):
        """Initialize the image processor with record keeping."""
        self.model_path = model_path
        self.database_path = database_path
        self.processed_images = self.load_database()
        
        # Initialize the blurrer
        self.blurrer = ImageBlurrer(model_path=model_path, parts=[
            'FEMALE_BREAST_EXPOSED',
            'FEMALE_GENITALIA_EXPOSED',
            'FEMALE_BREAST_COVERED',
            'ANUS_EXPOSED',
            'MALE_GENITALIA_EXPOSED',
        ])
    
    def load_database(self):
        """Load the processed images database."""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r') as f:
                    return json.load(f)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
                return {}
        except Exception as e:
            print(f"Warning: Could not load database: {e}")
            return {}
    
    def save_database(self):
        """Save the processed images database."""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.processed_images, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save database: {e}")
    
    def get_file_hash(self, file_path):
        """Generate a hash for the file to detect changes."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Warning: Could not generate hash for {file_path}: {e}")
            return None
    
    def is_already_processed(self, input_path, output_path, pixel_size=10):
        """Check if an image has already been processed with the same settings."""
        if input_path not in self.processed_images:
            return False
        
        record = self.processed_images[input_path]
        
        # Check if output file still exists
        if not os.path.exists(record['output_path']):
            return False
        
        # Check if file hash has changed (file was modified)
        current_hash = self.get_file_hash(input_path)
        if current_hash != record['file_hash']:
            return False
        
        # Check if processing settings are the same
        if (record['pixel_size'] != pixel_size or 
            record['output_path'] != output_path or
            record['model_path'] != self.model_path):
            return False
        
        return True
    
    def record_processed_image(self, input_path, output_path, pixel_size=10):
        """Record a processed image in the database."""
        file_hash = self.get_file_hash(input_path)
        
        self.processed_images[input_path] = {
            'output_path': output_path,
            'file_hash': file_hash,
            'pixel_size': pixel_size,
            'model_path': self.model_path,
            'processed_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(input_path) if os.path.exists(input_path) else 0
        }
        
        self.save_database()
    
    def process_single_image(self, input_path, output_path, pixel_size=10, force=False):
        """Process a single image with record keeping."""
        # Check if already processed
        if not force and self.is_already_processed(input_path, output_path, pixel_size):
            print(f"Image already processed: {input_path}")
            print(f"  Output: {self.processed_images[input_path]['output_path']}")
            print(f"  Processed: {self.processed_images[input_path]['processed_at']}")
            return self.processed_images[input_path]['output_path']
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Process the image
            result = self.blurrer.process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=pixel_size
            )
            
            # Record the processed image
            self.record_processed_image(input_path, output_path, pixel_size)
            
            print(f"Image processed successfully. Saved to: {result}")
            return result
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def process_multiple_images(self, input_dir, output_dir, pixel_size=10, force=False):
        """Process multiple images in a directory with record keeping."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Get all images from input directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                
                # Check if already processed
                if not force and self.is_already_processed(input_path, output_path, pixel_size):
                    print(f"Skipped (already processed): {filename}")
                    skipped_count += 1
                    continue
                
                try:
                    result = self.process_single_image(
                        input_path=input_path,
                        output_path=output_path,
                        pixel_size=pixel_size,
                        force=force
                    )
                    if result:
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    error_count += 1
        
        print(f"\nBatch processing complete:")
        print(f"  Processed: {processed_count} images")
        print(f"  Skipped (already processed): {skipped_count} images")
        print(f"  Errors: {error_count} images")
    
    def get_processing_stats(self):
        """Get statistics about processed images."""
        total_images = len(self.processed_images)
        total_size = sum(record.get('file_size', 0) for record in self.processed_images.values())
        
        print(f"Processing Statistics:")
        print(f"  Total images processed: {total_images}")
        print(f"  Total size processed: {total_size / (1024*1024):.2f} MB")
        
        if total_images > 0:
            # Show recent processing
            recent = sorted(
                self.processed_images.items(),
                key=lambda x: x[1]['processed_at'],
                reverse=True
            )[:5]
            
            print(f"  Recent processing:")
            for input_path, record in recent:
                filename = os.path.basename(input_path)
                print(f"    {filename} -> {os.path.basename(record['output_path'])} ({record['processed_at'][:19]})")
    
    def cleanup_orphaned_records(self):
        """Remove records for files that no longer exist."""
        orphaned = []
        for input_path in list(self.processed_images.keys()):
            if not os.path.exists(input_path):
                orphaned.append(input_path)
        
        for input_path in orphaned:
            del self.processed_images[input_path]
            print(f"Removed orphaned record: {input_path}")
        
        if orphaned:
            self.save_database()
            print(f"Cleaned up {len(orphaned)} orphaned records")

class SlidingWindowImageProcessor(ImageProcessor):
    def __init__(self, model_path='models/640m.onnx', database_path='data/processed_images.json',
                 window_size=640, stride=320, overlap_threshold=0.3):
        """Initialize the sliding window image processor with record keeping."""
        self.model_path = model_path
        self.database_path = database_path
        self.processed_images = self.load_database()
        
        # Sliding window parameters
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        
        # Initialize the sliding window blurrer
        self.blurrer = SlidingWindowBlurrer(
            model_path=model_path, 
            parts=[
                'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED',
                'FEMALE_BREAST_COVERED',
                'ANUS_EXPOSED',
                'MALE_GENITALIA_EXPOSED',
            ],
            window_size=window_size,
            stride=stride,
            overlap_threshold=overlap_threshold
        )
    
    def is_already_processed(self, input_path, output_path, pixel_size=10, confidence_threshold=0.1):
        """Check if an image has already been processed with the same settings."""
        if input_path not in self.processed_images:
            return False
        
        record = self.processed_images[input_path]
        
        # Check if output file still exists
        if not os.path.exists(record['output_path']):
            return False
        
        # Check if file hash has changed (file was modified)
        current_hash = self.get_file_hash(input_path)
        if current_hash != record['file_hash']:
            return False
        
        # Check if processing settings are the same (including sliding window params)
        if (record['pixel_size'] != pixel_size or 
            record['output_path'] != output_path or
            record['model_path'] != self.model_path or
            record.get('window_size') != self.window_size or
            record.get('stride') != self.stride or
            record.get('overlap_threshold') != self.overlap_threshold or
            record.get('confidence_threshold') != confidence_threshold):
            return False
        
        return True
    
    def record_processed_image(self, input_path, output_path, pixel_size=10, confidence_threshold=0.1):
        """Record a processed image in the database with sliding window parameters."""
        file_hash = self.get_file_hash(input_path)
        
        self.processed_images[input_path] = {
            'output_path': output_path,
            'file_hash': file_hash,
            'pixel_size': pixel_size,
            'model_path': self.model_path,
            'window_size': self.window_size,
            'stride': self.stride,
            'overlap_threshold': self.overlap_threshold,
            'confidence_threshold': confidence_threshold,
            'processed_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(input_path) if os.path.exists(input_path) else 0
        }
        
        self.save_database()
    
    def process_single_image(self, input_path, output_path, pixel_size=10, confidence_threshold=0.1, force=False, image_type=None):
        """Process a single image with sliding window approach and record keeping."""
        # Check if already processed
        if not force and self.is_already_processed(input_path, output_path, pixel_size, confidence_threshold):
            print(f"Image already processed: {input_path}")
            print(f"  Output: {self.processed_images[input_path]['output_path']}")
            print(f"  Processed: {self.processed_images[input_path]['processed_at']}")
            return self.processed_images[input_path]['output_path']
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Process the image with sliding window approach
            result = self.blurrer.process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=pixel_size,
                confidence_threshold=confidence_threshold,
                create_wordpress_sizes=True,
                image_type=image_type
            )
            
            # Record the processed image
            self.record_processed_image(input_path, output_path, pixel_size, confidence_threshold)
            
            print(f"Image processed successfully with sliding window. Saved to: {result}")
            return result
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def process_multiple_images(self, input_dir, output_dir, pixel_size=10, confidence_threshold=0.1, force=False):
        """Process multiple images in a directory with sliding window approach and record keeping."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Get all images from input directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                # Remove 'processed_' prefix from output filename
                output_filename = filename
                if output_filename.startswith('processed_'):
                    output_filename = output_filename[10:]  # Remove 'processed_' prefix
                output_path = os.path.join(output_dir, output_filename)
                
                # Check if already processed
                if not force and self.is_already_processed(input_path, output_path, pixel_size, confidence_threshold):
                    print(f"Skipped (already processed): {filename}")
                    skipped_count += 1
                    continue
                
                try:
                    result = self.process_single_image(
                        input_path=input_path,
                        output_path=output_path,
                        pixel_size=pixel_size,
                        confidence_threshold=confidence_threshold,
                        force=force
                    )
                    if result:
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    error_count += 1
        
        print(f"\nSliding window batch processing complete:")
        print(f"  Processed: {processed_count} images")
        print(f"  Skipped (already processed): {skipped_count} images")
        print(f"  Errors: {error_count} images")
        print(f"  Window size: {self.window_size}x{self.window_size}")
        print(f"  Stride: {self.stride} ({(self.window_size - self.stride) / self.window_size * 100:.1f}% overlap)")

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
            image_urls.append({
                'type': 'review_full_image',
                'url': item['review_full_image'],
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

class SlidingWindowCustomJSONImageProcessor(CustomJSONImageProcessor):
    def __init__(self, json_url=None, json_file=None, base_url=None, model_path='models/640m.onnx', 
                 database_path='data/processed_images.json', window_size=640, stride=320, overlap_threshold=0.3):
        """Initialize sliding window custom JSON image processor."""
        # Initialize parent without creating the blurrer
        self.model_path = model_path
        self.database_path = database_path
        self.processed_images = self.load_database()
        self.json_url = json_url
        self.json_file = json_file
        self.base_url = base_url.rstrip('/') if base_url else None
        
        # Sliding window parameters
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        
        # Initialize the sliding window blurrer
        self.blurrer = SlidingWindowBlurrer(
            model_path=model_path, 
            parts=[
                'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED',
                'FEMALE_BREAST_COVERED',
                'ANUS_EXPOSED',
                'MALE_GENITALIA_EXPOSED',
            ],
            window_size=window_size,
            stride=stride,
            overlap_threshold=overlap_threshold
        )
    
    def is_already_processed(self, input_path, output_path, pixel_size=10, confidence_threshold=0.1):
        """Check if an image has already been processed with the same settings."""
        if input_path not in self.processed_images:
            return False
        
        record = self.processed_images[input_path]
        
        # Check if output file still exists
        if not os.path.exists(record['output_path']):
            return False
        
        # Check if file hash has changed (file was modified)
        current_hash = self.get_file_hash(input_path)
        if current_hash != record['file_hash']:
            return False
        
        # Check if processing settings are the same (including sliding window params)
        if (record['pixel_size'] != pixel_size or 
            record['output_path'] != output_path or
            record['model_path'] != self.model_path or
            record.get('window_size') != self.window_size or
            record.get('stride') != self.stride or
            record.get('overlap_threshold') != self.overlap_threshold or
            record.get('confidence_threshold') != confidence_threshold):
            return False
        
        return True
    
    def record_processed_image(self, input_path, output_path, pixel_size=10, confidence_threshold=0.1):
        """Record a processed image in the database with sliding window parameters."""
        file_hash = self.get_file_hash(input_path)
        
        self.processed_images[input_path] = {
            'output_path': output_path,
            'file_hash': file_hash,
            'pixel_size': pixel_size,
            'model_path': self.model_path,
            'window_size': self.window_size,
            'stride': self.stride,
            'overlap_threshold': self.overlap_threshold,
            'confidence_threshold': confidence_threshold,
            'processed_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(input_path) if os.path.exists(input_path) else 0
        }
        
        self.save_database()
    
    def process_single_image(self, input_path, output_path, pixel_size=10, confidence_threshold=0.1, force=False, image_type=None):
        """Process a single image with sliding window approach and record keeping."""
        # Check if already processed
        if not force and self.is_already_processed(input_path, output_path, pixel_size, confidence_threshold):
            print(f"Image already processed: {input_path}")
            print(f"  Output: {self.processed_images[input_path]['output_path']}")
            print(f"  Processed: {self.processed_images[input_path]['processed_at']}")
            return self.processed_images[input_path]['output_path']
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Process the image with sliding window approach
            result = self.blurrer.process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=pixel_size,
                confidence_threshold=confidence_threshold,
                create_wordpress_sizes=True,
                image_type=image_type
            )
            
            # Record the processed image
            self.record_processed_image(input_path, output_path, pixel_size, confidence_threshold)
            
            print(f"Image processed successfully with sliding window. Saved to: {result}")
            return result
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def process_custom_json_images(self, output_dir="data/custom_processed", pixel_size=10, 
                                 confidence_threshold=0.1, force=False, download_only=False):
        """Process images from custom JSON format with sliding window approach."""
        print("Fetching JSON data...")
        json_data = self.fetch_json_data()
        
        if not json_data:
            print("No data found or error loading JSON.")
            return
        
        print(f"Found {len(json_data)} items in JSON data")
        print(f"Using sliding window: {self.window_size}x{self.window_size}, stride={self.stride}")
        
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
                
                # Process the image with sliding window
                try:
                    result = self.process_single_image(
                        input_path=local_path,
                        output_path=output_path,
                        pixel_size=pixel_size,
                        confidence_threshold=confidence_threshold,
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
        
        print(f"\nSliding window custom JSON processing complete:")
        print(f"  Total items: {len(json_data)}")
        print(f"  Total images found: {total_images}")
        print(f"  Downloaded: {downloaded_count} images")
        print(f"  Processed: {processed_count} images")
        print(f"  Skipped: {skipped_count} images")
        print(f"  Errors: {error_count} images")
        print(f"  Window size: {self.window_size}x{self.window_size}")
        print(f"  Stride: {self.stride} ({(self.window_size - self.stride) / self.stride * 100:.1f}% overlap)")

class SlidingWindowWordPressImageProcessor(SlidingWindowCustomJSONImageProcessor):
    def __init__(self, json_url=None, json_file=None, base_url=None, model_path='models/640m.onnx', 
                 database_path='data/processed_images.json', window_size=640, stride=320, overlap_threshold=0.3):
        """Initialize sliding window WordPress image processor for existing images."""
        # Initialize parent without creating the blurrer
        self.model_path = model_path
        self.database_path = database_path
        self.processed_images = self.load_database()
        self.json_url = json_url
        self.json_file = json_file
        self.base_url = base_url.rstrip('/') if base_url else None
        
        # Sliding window parameters
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        
        # Initialize the sliding window blurrer
        self.blurrer = SlidingWindowBlurrer(
            model_path=model_path, 
            parts=[
                'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED',
                'FEMALE_BREAST_COVERED',
                'ANUS_EXPOSED',
                'MALE_GENITALIA_EXPOSED',
            ],
            window_size=window_size,
            stride=stride,
            overlap_threshold=overlap_threshold
        )
    
    def get_wordpress_image_path(self, image_url, slug, image_type):
        """Get the WordPress uploads path for an existing image based on relative URL."""
        # Extract filename from URL
        parsed_url = urlparse(image_url)
        original_filename = os.path.basename(parsed_url.path)
        
        if not original_filename or '.' not in original_filename:
            # Fallback to type-based filename
            original_filename = f"{image_type}.jpg"
        
        # Determine WordPress uploads path based on image type
        if image_type == 'review_full_image':
            # Look in wp-content/uploads/screenshots
            wp_path = os.path.join('wp-content', 'uploads', 'screenshots', original_filename)
        else:
            # Look in wp-content/uploads
            wp_path = os.path.join('wp-content', 'uploads', original_filename)
        
        return wp_path
    
    def process_wordpress_json_images(self, output_dir="data/wordpress_processed", pixel_size=10, 
                                    confidence_threshold=0.1, force=False):
        """Process existing images from WordPress uploads folder using JSON data."""
        print("Fetching JSON data...")
        json_data = self.fetch_json_data()
        
        if not json_data:
            print("No data found or error loading JSON.")
            return
        
        print(f"Found {len(json_data)} items in JSON data")
        print(f"Using sliding window: {self.window_size}x{self.window_size}, stride={self.stride}")
        print("Processing existing images from wp-content/uploads folder...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        found_count = 0
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
                
                # Get WordPress uploads path for existing image
                wp_image_path = self.get_wordpress_image_path(image_url, slug, image_type)
                
                print(f"  Processing {image_type}: {image_url}")
                print(f"    Looking for: {wp_image_path}")
                
                # Check if image exists in WordPress uploads
                if not os.path.exists(wp_image_path):
                    print(f"    ❌ Image not found: {wp_image_path}")
                    error_count += 1
                    continue
                
                print(f"    ✅ Found image: {wp_image_path}")
                found_count += 1
                
                # Generate output path
                filename = os.path.basename(wp_image_path)
                output_filename = filename
                if output_filename.startswith('processed_'):
                    output_filename = output_filename[10:]  # Remove 'processed_' prefix
                output_path = os.path.join(output_dir, output_filename)
                
                # Process the image with sliding window
                try:
                    result = self.process_single_image(
                        input_path=wp_image_path,
                        output_path=output_path,
                        pixel_size=pixel_size,
                        confidence_threshold=confidence_threshold,
                        force=force,
                        image_type=image_type
                    )
                    if result:
                        processed_count += 1
                        print(f"    ✅ Processed successfully")
                    else:
                        error_count += 1
                        print(f"    ❌ Processing failed")
                except Exception as e:
                    print(f"    ❌ Error processing {image_type} for {slug}: {str(e)}")
                    error_count += 1
        
        print(f"\nWordPress image processing complete:")
        print(f"  Total items: {len(json_data)}")
        print(f"  Total images found in JSON: {total_images}")
        print(f"  Images found in wp-content/uploads: {found_count}")
        print(f"  Processed: {processed_count} images")
        print(f"  Skipped: {skipped_count} images")
        print(f"  Errors: {error_count} images")
        print(f"  Window size: {self.window_size}x{self.window_size}")
        print(f"  Stride: {self.stride} ({(self.window_size - self.stride) / self.stride * 100:.1f}% overlap)")

def list_available_parts():
    """List available body parts for blurring."""
    parts = [
        'FEMALE_BREAST_EXPOSED',
        'FEMALE_GENITALIA_EXPOSED',
        'ANUS_EXPOSED',
        'MALE_GENITALIA_EXPOSED'
    ]
    
    print("Available body parts for blurring:")
    for part in parts:
        print(f"  - {part}")
    print("\nNote: You can specify multiple parts separated by commas")

def validate_paths(input_path, output_path, model_path):
    """Validate file and directory paths."""
    # Check if input file/directory exists
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return False
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file does not exist: {model_path}")
        return False
    
    # Check if input is an image file (for single mode)
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Error: Input file is not a supported image format (jpg, jpeg, png): {input_path}")
            return False
    
    return True

def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Image Processing Script with Record Keeping and Custom JSON Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s single input.jpg output.jpg
  %(prog)s batch data/input data/output
  %(prog)s single image.jpg result.jpg --pixel-size 15
  %(prog)s batch photos/ processed/ --force
  %(prog)s json https://example.com/api/images --base-url https://example.com
  %(prog)s json data/images.json --base-url https://example.com
  %(prog)s json data/images.json --download-only
  %(prog)s sliding-single input.jpg output.jpg --window-size 160 --stride 80
  %(prog)s sliding-batch data/input data/output --window-size 640 --stride 320
  %(prog)s sliding-json data/images.json --window-size 512 --stride 256
  %(prog)s sliding-wordpress data/images.json --window-size 512 --stride 256
  %(prog)s list-parts
  %(prog)s stats
  %(prog)s cleanup
  %(prog)s  # Run with default settings (batch processing)
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['single', 'batch', 'json', 'sliding-single', 'sliding-batch', 'sliding-json', 'sliding-wordpress', 'list-parts', 'stats', 'cleanup'],
        default='batch',
        help='Command to execute (default: batch)'
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Input file, directory path, JSON URL, or JSON file path'
    )
    
    parser.add_argument(
        'output',
        nargs='?',
        help='Output file or directory path'
    )
    
    parser.add_argument(
        '--model',
        default='models/640m.onnx',
        help='Path to the model file (default: models/640m.onnx)'
    )
    
    parser.add_argument(
        '--pixel-size',
        type=int,
        default=12,
        help='Pixelation size (default: 10)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if image was already processed'
    )
    
    parser.add_argument(
        '--database',
        default='data/processed_images.json',
        help='Path to the processing database (default: data/processed_images.json)'
    )
    
    # Sliding window specific arguments
    parser.add_argument(
        '--window-size',
        type=int,
        default=640,
        help='Sliding window size in pixels (default: 640)'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=320,
        help='Stride between sliding windows in pixels (default: 320)'
    )
    
    parser.add_argument(
        '--overlap-threshold',
        type=float,
        default=0.3,
        help='IoU threshold for merging overlapping detections (default: 0.3)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.1,
        help='Minimum confidence score for detections (default: 0.1)'
    )
    
    # Custom JSON specific arguments
    parser.add_argument(
        '--base-url',
        help='Base URL for converting relative paths to absolute URLs'
    )
    
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download images, do not process them'
    )
    
    args = parser.parse_args()
    
    # Handle non-processing commands
    if args.command == 'list-parts':
        list_available_parts()
        return
    elif args.command == 'stats':
        processor = ImageProcessor(database_path=args.database)
        processor.get_processing_stats()
        return
    elif args.command == 'cleanup':
        processor = ImageProcessor(database_path=args.database)
        processor.cleanup_orphaned_records()
        return
    
    # Handle sliding window custom JSON processing
    if args.command == 'sliding-json':
        if not args.input:
            parser.error("sliding-json command requires JSON URL or file path")
        
        # Determine if input is URL or file
        if args.input.startswith(('http://', 'https://')):
            json_url = args.input
            json_file = None
        else:
            json_url = None
            json_file = args.input
        
        json_processor = SlidingWindowCustomJSONImageProcessor(
            json_url=json_url,
            json_file=json_file,
            base_url=args.base_url,
            model_path=args.model,
            database_path=args.database,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold
        )
        print('Processing custom JSON images with sliding window... ', args.overlap_threshold)
        
        # Clean up orphaned records
        json_processor.cleanup_orphaned_records()
        
        # Show current statistics
        json_processor.get_processing_stats()
        
        # Process custom JSON images with sliding window
        output_dir = args.output if args.output else "data/custom_processed"
        json_processor.process_custom_json_images(
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            confidence_threshold=args.confidence_threshold,
            force=args.force,
            download_only=args.download_only
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        json_processor.get_processing_stats()
        return
    
    # Handle custom JSON processing (original)
    if args.command == 'json':
        if not args.input:
            parser.error("json command requires JSON URL or file path")
        
        # Determine if input is URL or file
        if args.input.startswith(('http://', 'https://')):
            json_url = args.input
            json_file = None
        else:
            json_url = None
            json_file = args.input
        
        json_processor = CustomJSONImageProcessor(
            json_url=json_url,
            json_file=json_file,
            base_url=args.base_url,
            model_path=args.model,
            database_path=args.database
        )
        
        # Clean up orphaned records
        json_processor.cleanup_orphaned_records()
        
        # Show current statistics
        json_processor.get_processing_stats()
        
        # Process custom JSON images
        output_dir = args.output if args.output else "data/custom_processed"
        json_processor.process_custom_json_images(
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            force=args.force,
            download_only=args.download_only
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        json_processor.get_processing_stats()
        return
    
    # Handle sliding window single image processing
    if args.command == 'sliding-single':
        if not args.input or not args.output:
            parser.error("sliding-single command requires input and output arguments")
        
        if not validate_paths(args.input, args.output, args.model):
            sys.exit(1)
        
        # Initialize sliding window processor
        processor = SlidingWindowImageProcessor(
            model_path=args.model, 
            database_path=args.database,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold
        )
        
        # Clean up orphaned records
        processor.cleanup_orphaned_records()
        
        # Show current statistics
        processor.get_processing_stats()
        
        print(f"Processing single image with sliding window: {args.input} -> {args.output}")
        print(f"Window size: {args.window_size}x{args.window_size}, Stride: {args.stride}")
        
        processor.process_single_image(
            input_path=args.input,
            output_path=args.output,
            pixel_size=args.pixel_size,
            confidence_threshold=args.confidence_threshold,
            force=args.force
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        processor.get_processing_stats()
        return
    
    # Handle sliding window batch processing
    if args.command == 'sliding-batch':
        # Use default paths if not provided
        input_dir = args.input if args.input else "data/input"
        output_dir = args.output if args.output else "data/output"
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        if not validate_paths(input_dir, output_dir, args.model):
            sys.exit(1)
        
        # Initialize sliding window processor
        processor = SlidingWindowImageProcessor(
            model_path=args.model, 
            database_path=args.database,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold
        )
        
        # Clean up orphaned records
        processor.cleanup_orphaned_records()
        
        # Show current statistics
        processor.get_processing_stats()
        
        print(f"Processing batch images with sliding window from: {input_dir} -> {output_dir}")
        print(f"Window size: {args.window_size}x{args.window_size}, Stride: {args.stride}")
        
        processor.process_multiple_images(
            input_dir=input_dir,
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            confidence_threshold=args.confidence_threshold,
            force=args.force
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        processor.get_processing_stats()
        return
    
    # Handle sliding window WordPress processing
    if args.command == 'sliding-wordpress':
        if not args.input:
            parser.error("sliding-wordpress command requires JSON URL or file path")
        
        # Determine if input is URL or file
        if args.input.startswith(('http://', 'https://')):
            json_url = args.input
            json_file = None
        else:
            json_url = None
            json_file = args.input
        
        wordpress_processor = SlidingWindowWordPressImageProcessor(
            json_url=json_url,
            json_file=json_file,
            base_url=args.base_url,
            model_path=args.model,
            database_path=args.database,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold
        )
        print('Processing existing WordPress images with sliding window...')
        
        # Clean up orphaned records
        wordpress_processor.cleanup_orphaned_records()
        
        # Show current statistics
        wordpress_processor.get_processing_stats()
        
        # Process WordPress images with sliding window
        output_dir = args.output if args.output else "data/wordpress_processed"
        wordpress_processor.process_wordpress_json_images(
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            confidence_threshold=args.confidence_threshold,
            force=args.force
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        wordpress_processor.get_processing_stats()
        return
    
    # Initialize processor for other commands (original functionality)
    processor = ImageProcessor(model_path=args.model, database_path=args.database)
    
    # Clean up orphaned records
    processor.cleanup_orphaned_records()
    
    # Show current statistics
    processor.get_processing_stats()
    
    # Handle processing commands (original functionality)
    if args.command == 'single':
        if not args.input or not args.output:
            parser.error("single command requires input and output arguments")
        
        if not validate_paths(args.input, args.output, args.model):
            sys.exit(1)
        
        print(f"Processing single image: {args.input} -> {args.output}")
        processor.process_single_image(
            input_path=args.input,
            output_path=args.output,
            pixel_size=args.pixel_size,
            force=args.force
        )
    
    elif args.command == 'batch':
        # Use default paths if not provided
        input_dir = args.input if args.input else "data/input"
        output_dir = args.output if args.output else "data/output"
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        if not validate_paths(input_dir, output_dir, args.model):
            sys.exit(1)
        
        print(f"Processing batch images from: {input_dir} -> {output_dir}")
        processor.process_multiple_images(
            input_dir=input_dir,
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            force=args.force
        )
    
    # Show updated statistics
    print("\n" + "="*50)
    processor.get_processing_stats()

if __name__ == "__main__":
    main() 