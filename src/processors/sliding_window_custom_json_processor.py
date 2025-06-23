#!/usr/bin/env python3
"""
Sliding Window Custom JSON Image Processor Class

This module contains the SlidingWindowCustomJSONImageProcessor class that provides
sliding window image processing functionality for images specified in JSON format.
"""

import os
from datetime import datetime

from .custom_json_processor import CustomJSONImageProcessor
from ..blurrer import SlidingWindowBlurrer


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
                'BUTTOCKS_EXPOSED',
                'BUTTOCKS_COVERED',
                'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED',
                'FEMALE_GENITALIA_COVERED',
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