#!/usr/bin/env python3
"""
Test script for WordPress image processing with backup functionality.
This script demonstrates how to process existing WordPress images with automatic backups.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.processors import SlidingWindowWordPressImageProcessor


def test_wordpress_backup():
    """Test the WordPress processor with backup functionality."""
    
    # Test JSON data with WordPress image paths
    test_json_data = [
        {
            "slug": "theartporn",
            "screenshot_full_url": "/wp-content/uploads/theartporn-featured-image.jpg",
            "review_full_image": "/wp-content/uploads/screenshots/theartpornfeaturedimage_watermark.jpg"
        },
        {
            "slug": "testimage",
            "screenshot_full_url": "/wp-content/uploads/test-featured-image.jpg",
            "review_full_image": "/wp-content/uploads/screenshots/testfeaturedimage_watermark.jpg"
        }
    ]
    
    # Create test JSON file
    import json
    test_json_file = "test_wordpress_images.json"
    with open(test_json_file, 'w') as f:
        json.dump(test_json_data, f, indent=2)
    
    print("Created test JSON file:", test_json_file)
    
    # Create test WordPress directory structure
    wp_uploads_dir = "wp-content/uploads"
    wp_screenshots_dir = os.path.join(wp_uploads_dir, "screenshots")
    
    os.makedirs(wp_uploads_dir, exist_ok=True)
    os.makedirs(wp_screenshots_dir, exist_ok=True)
    
    # Create test images (you can replace these with actual test images)
    test_images = [
        os.path.join(wp_uploads_dir, "theartporn-featured-image.jpg"),
        os.path.join(wp_screenshots_dir, "theartpornfeaturedimage_watermark.jpg"),
        os.path.join(wp_uploads_dir, "test-featured-image.jpg"),
        os.path.join(wp_screenshots_dir, "testfeaturedimage_watermark.jpg")
    ]
    
    # Create placeholder images if they don't exist
    for image_path in test_images:
        if not os.path.exists(image_path):
            # Create a simple placeholder image using PIL
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (800, 600), color='lightblue')
                draw = ImageDraw.Draw(img)
                draw.text((400, 300), f"Test Image\n{os.path.basename(image_path)}", 
                         fill='black', anchor='mm')
                img.save(image_path)
                print(f"Created placeholder image: {image_path}")
            except ImportError:
                print(f"PIL not available, skipping image creation: {image_path}")
                # Create empty file as fallback
                with open(image_path, 'w') as f:
                    f.write("placeholder")
    
    # Initialize WordPress processor
    processor = SlidingWindowWordPressImageProcessor(
        json_file=test_json_file,
        model_path='models/640m.onnx',
        database_path='data/wordpress_processed.json',
        window_size=640,
        stride=320,
        overlap_threshold=0.3
    )
    
    print("\n" + "="*60)
    print("Testing WordPress Image Processing with Backup Functionality")
    print("="*60)
    
    # Test backup function directly
    print("\n1. Testing backup function...")
    for image_path in test_images:
        if os.path.exists(image_path):
            try:
                backup_path = processor.create_backup(image_path)
                print(f"   ✅ Backup created: {backup_path}")
            except Exception as e:
                print(f"   ❌ Backup failed for {image_path}: {e}")
    
    # Test WordPress processing
    print("\n2. Testing WordPress image processing...")
    try:
        processor.process_wordpress_json_images(
            output_dir="data/wordpress_processed",
            pixel_size=10,
            confidence_threshold=0.1,
            force=True
        )
        print("   ✅ WordPress processing completed")
    except Exception as e:
        print(f"   ❌ WordPress processing failed: {e}")
    
    # Show backup directory structure
    print("\n3. Backup directory structure:")
    if os.path.exists("backups"):
        for root, dirs, files in os.walk("backups"):
            level = root.replace("backups", "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = "  " * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("   No backups directory found")
    
    # Cleanup
    print("\n4. Cleanup...")
    if os.path.exists(test_json_file):
        os.remove(test_json_file)
        print(f"   Removed test JSON file: {test_json_file}")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)


if __name__ == "__main__":
    test_wordpress_backup() 