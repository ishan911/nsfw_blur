#!/usr/bin/env python3
"""
Base Image Processor Class

This module contains the base ImageProcessor class that provides
standard image processing functionality with database tracking.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

from ..blurrer import ImageBlurrer


class ImageProcessor:
    def __init__(self, model_path='models/640m.onnx', database_path='data/processed_images.json'):
        """Initialize the image processor with record keeping."""
        self.model_path = model_path
        self.database_path = database_path
        self.processed_images = self.load_database()
        
        # Initialize the blurrer
        self.blurrer = ImageBlurrer(model_path=model_path, parts=[
            'BUTTOCKS_EXPOSED',
            'BUTTOCKS_COVERED',
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