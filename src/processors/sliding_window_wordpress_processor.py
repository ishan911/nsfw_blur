#!/usr/bin/env python3
"""
Sliding Window WordPress Image Processor Class

This module contains the SlidingWindowWordPressImageProcessor class that provides
sliding window image processing functionality for existing WordPress images.
"""

import os
import shutil
from datetime import datetime
from urllib.parse import urlparse

from .sliding_window_custom_json_processor import SlidingWindowCustomJSONImageProcessor
from ..blurrer import SlidingWindowBlurrer


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
    
    def extract_image_urls(self, item):
        """
        Extract image URLs from a JSON item for WordPress processing.
        Does not join base_url to keep relative paths for local file lookup.
        """
        image_urls = []
        
        # Handle screenshot_url (keep as relative path)
        if item.get('screenshot_full_url'):
            screenshot_url = item['screenshot_full_url']
            image_urls.append({
                'type': 'screenshot_full_url',
                'url': screenshot_url,
                'slug': item.get('slug', 'unknown')
            })
        
        # Handle mobile_thumb (keep as is)
        if item.get('mobile_thumb'):
            image_urls.append({
                'type': 'mobile_thumb',
                'url': item['mobile_thumb'],
                'slug': item.get('slug', 'unknown')
            })
        
        # Handle review_image (keep as relative path)
        if item.get('review_full_image'):
            review_full_url = item['review_full_image']
            image_urls.append({
                'type': 'review_full_image',
                'url': review_full_url,
                'slug': item.get('slug', 'unknown')
            })
        
        return image_urls
    
    def get_image_path(self, image_url, slug, image_type):
        """
        Get the local path for an image, skipping remote URLs.
        
        Args:
            image_url (str): Image URL (can be local path or remote URL)
            slug (str): Item slug
            image_type (str): Type of image
            
        Returns:
            str: Local path to the image if it exists, or None if remote or not found
        """
        # Check if it's a remote URL
        if image_url.startswith(('http://', 'https://')):
            # Skip remote URLs
            print(f"    ⏭️  Skipping remote URL: {image_url}")
            return None
        else:
            # It's a local path, check if it exists
            wp_image_path = self.get_wordpress_image_path(image_url, slug, image_type)
            if os.path.exists(wp_image_path):
                return wp_image_path
            else:
                print(f"    ❌ Local image not found: {wp_image_path}")
                return None
    
    def create_backup(self, file_path: str) -> str:
        """
        Create a backup of the original file before processing.
        Backups are created in wp-content/uploads/backup folder with original filename.
        
        Args:
            file_path (str): Path to the file to backup
            
        Returns:
            str: Path to the backup file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create backup directory inside wp-content/uploads
        backup_dir = "wp-content/uploads/backup"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Get the original filename
        filename = os.path.basename(file_path)
        
        # Create backup path in the backup directory with original filename
        backup_path = os.path.join(backup_dir, filename)
        
        # Copy the file
        shutil.copy2(file_path, backup_path)
        print(f"  ✅ Created backup: {backup_path}")
        
        return backup_path
    
    def get_wordpress_image_path(self, image_url, slug, image_type):
        """Get the WordPress uploads path for an existing image based on relative URL."""
        # Extract filename from URL
        parsed_url = urlparse(image_url)
        original_filename = os.path.basename(parsed_url.path)
        
        if not original_filename or '.' not in original_filename:
            # Fallback to simple filename without image type
            original_filename = "image.jpg"
        
        # Determine WordPress uploads path based on image type
        if image_type == 'review_full_image':
            # Look in wp-content/uploads/screenshots
            wp_path = os.path.join('wp-content', 'uploads', 'screenshots', original_filename)
        else:
            # Look in wp-content/uploads
            wp_path = os.path.join('wp-content', 'uploads', original_filename)
        
        return wp_path
    
    def process_wordpress_json_images(self, output_dir="data/wordpress_processed", pixel_size=10, 
                                    confidence_threshold=0.1, force=False, debug=False):
        """Process existing images from WordPress uploads folder using JSON data."""
        if debug:
            print("Fetching JSON data...")
        json_data = self.fetch_json_data()
        
        if not json_data:
            print("No data found or error loading JSON.")
            return
        
        if debug:
            print(f"Found {len(json_data)} items in JSON data")
            print(f"Using sliding window: {self.window_size}x{self.window_size}, stride={self.stride}")
            print("Processing existing images from wp-content/uploads folder (skipping remote URLs)...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        found_count = 0
        skipped_count = 0
        remote_skipped_count = 0
        error_count = 0
        total_images = 0
        
        for item in json_data:
            slug = item.get('slug', 'unknown')
            
            # Extract all image URLs from this item
            image_urls = self.extract_image_urls(item)
            total_images += len(image_urls)
            
            entry_processed = 0
            entry_found = 0
            entry_remote_skipped = 0
            entry_errors = 0
            
            for image_info in image_urls:
                image_url = image_info['url']
                image_type = image_info['type']
                
                if debug:
                    print(f"  Processing {image_type}: {image_url}")
                
                # Get local image path (skip remote URLs)
                local_image_path = self.get_image_path(image_url, slug, image_type)
                
                if not local_image_path:
                    if image_url.startswith(('http://', 'https://')):
                        remote_skipped_count += 1
                        entry_remote_skipped += 1
                        if debug:
                            print(f"    ⏭️  Skipped remote URL")
                    else:
                        error_count += 1
                        entry_errors += 1
                        if debug:
                            print(f"    ❌ Could not get image: {image_url}")
                    continue
                
                found_count += 1
                entry_found += 1
                if debug:
                    print(f"    ✅ Found local image: {local_image_path}")
                
                # Create backup before processing
                try:
                    backup_path = self.create_backup(local_image_path)
                    if debug:
                        print(f"    ✅ Created backup: {backup_path}")
                except Exception as e:
                    error_count += 1
                    entry_errors += 1
                    if debug:
                        print(f"    ❌ Failed to create backup: {e}")
                    continue
                
                # Generate output path (same as original for in-place processing)
                output_path = local_image_path
                
                # Process the image with sliding window
                try:
                    result = self.process_single_image(
                        input_path=local_image_path,
                        output_path=output_path,
                        pixel_size=pixel_size,
                        confidence_threshold=confidence_threshold,
                        force=force,
                        image_type=image_type
                    )
                    if result:
                        processed_count += 1
                        entry_processed += 1
                        if debug:
                            print(f"    ✅ Processed successfully")
                    else:
                        error_count += 1
                        entry_errors += 1
                        if debug:
                            print(f"    ❌ Processing failed")
                except Exception as e:
                    error_count += 1
                    entry_errors += 1
                    if debug:
                        print(f"    ❌ Error processing {image_type} for {slug}: {str(e)}")
            
            # Print one line summary for this entry
            summary_parts = []
            if entry_processed > 0:
                summary_parts.append(f"✅ {entry_processed} processed")
            if entry_found > 0:
                summary_parts.append(f"📁 {entry_found} found")
            if entry_remote_skipped > 0:
                summary_parts.append(f"⏭️ {entry_remote_skipped} remote skipped")
            if entry_errors > 0:
                summary_parts.append(f"❌ {entry_errors} errors")
            
            summary = " | ".join(summary_parts) if summary_parts else "⏭️ skipped"
            print(f"{slug}: {summary}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"WordPress image processing complete:")
        print(f"  Total items: {len(json_data)}")
        print(f"  Total images found in JSON: {total_images}")
        print(f"  Local images found: {found_count}")
        print(f"  Remote URLs skipped: {remote_skipped_count}")
        print(f"  Processed: {processed_count} images")
        print(f"  Skipped: {skipped_count} images")
        print(f"  Errors: {error_count} images")
        print(f"  Window size: {self.window_size}x{self.window_size}")
        print(f"  Stride: {self.stride} ({(self.window_size - self.stride) / self.stride * 100:.1f}% overlap)")
        print(f"  Backups created in: wp-content/uploads/backup/") 