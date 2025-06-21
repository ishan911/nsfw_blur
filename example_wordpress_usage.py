#!/usr/bin/env python3
"""
Example usage of WordPress Image Processor with Backup Functionality

This script demonstrates how to process existing WordPress images from wp-content/uploads
with automatic backup creation before processing.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.processors import SlidingWindowWordPressImageProcessor


def main():
    """Example usage of WordPress processor."""
    
    print("WordPress Image Processor with Backup Functionality")
    print("=" * 50)
    
    # Configuration
    json_file = "data/wordpress_images.json"  # Your JSON file with image data
    model_path = "models/640m.onnx"
    database_path = "data/wordpress_processed.json"
    
    # Sliding window parameters
    window_size = 640
    stride = 320
    overlap_threshold = 0.3
    
    # Processing parameters
    pixel_size = 10
    confidence_threshold = 0.1
    force = False  # Set to True to reprocess already processed images
    
    print(f"JSON file: {json_file}")
    print(f"Model: {model_path}")
    print(f"Window size: {window_size}x{window_size}")
    print(f"Stride: {stride}")
    print(f"Pixel size: {pixel_size}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Force reprocess: {force}")
    
    # Initialize WordPress processor
    processor = SlidingWindowWordPressImageProcessor(
        json_file=json_file,
        model_path=model_path,
        database_path=database_path,
        window_size=window_size,
        stride=stride,
        overlap_threshold=overlap_threshold
    )
    
    print("\nProcessing WordPress images...")
    print("This will:")
    print("1. Look for images in wp-content/uploads/ and wp-content/uploads/screenshots/")
    print("2. Create backups in backups/ directory before processing")
    print("3. Process images in-place (replace originals)")
    print("4. Track processing in the database")
    
    # Process WordPress images
    processor.process_wordpress_json_images(
        output_dir="data/wordpress_processed",  # Not used for in-place processing
        pixel_size=pixel_size,
        confidence_threshold=confidence_threshold,
        force=force
    )
    
    print("\nProcessing completed!")
    print("Check the backups/ directory for original image backups.")


if __name__ == "__main__":
    main() 