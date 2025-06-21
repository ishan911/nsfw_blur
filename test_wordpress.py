#!/usr/bin/env python3
"""
Test script for the new sliding-wordpress functionality.
This demonstrates how to process existing images from wp-content/uploads folder.
"""

import os
import json
from src.blurrer import SlidingWindowBlurrer

def create_test_json():
    """Create a test JSON file with sample data."""
    test_data = [
        {
            "slug": "test-game-1",
            "screenshot_full_url": "https://example.com/wp-content/uploads/game-screenshot.jpg",
            "review_full_image": "https://example.com/wp-content/uploads/screenshots/game-review.jpg"
        },
        {
            "slug": "test-game-2", 
            "screenshot_full_url": "https://example.com/wp-content/uploads/another-screenshot.png",
            "review_full_image": "https://example.com/wp-content/uploads/screenshots/another-review.png"
        }
    ]
    
    # Create test JSON file
    with open('test_wordpress_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("Created test JSON file: test_wordpress_data.json")
    return test_data

def create_test_images():
    """Create test image files in WordPress folder structure."""
    # Create WordPress uploads directories
    os.makedirs('wp-content/uploads', exist_ok=True)
    os.makedirs('wp-content/uploads/screenshots', exist_ok=True)
    
    # Create test images (you would replace these with actual images)
    test_images = [
        'wp-content/uploads/game-screenshot.jpg',
        'wp-content/uploads/another-screenshot.png', 
        'wp-content/uploads/screenshots/game-review.jpg',
        'wp-content/uploads/screenshots/another-review.png'
    ]
    
    for image_path in test_images:
        # Create a simple test image file (you can replace with actual images)
        with open(image_path, 'w') as f:
            f.write("# This is a placeholder for a test image\n")
        print(f"Created test image: {image_path}")
    
    print("Created test WordPress folder structure")

def test_wordpress_processor():
    """Test the WordPress image processor."""
    from main import SlidingWindowWordPressImageProcessor
    
    # Create test data and images
    create_test_json()
    create_test_images()
    
    print("\n" + "="*60)
    print("Testing SlidingWindowWordPressImageProcessor")
    print("="*60)
    
    # Initialize the processor
    processor = SlidingWindowWordPressImageProcessor(
        json_file='test_wordpress_data.json',
        model_path='models/640m.onnx',
        window_size=160,  # Small window for testing
        stride=120,
        overlap_threshold=0.8
    )
    
    # Process the images
    processor.process_wordpress_json_images(
        output_dir="test_output",
        pixel_size=15,
        confidence_threshold=0.15,
        force=True
    )

if __name__ == "__main__":
    print("WordPress Image Processing Test")
    print("This script demonstrates the new sliding-wordpress functionality.")
    print("It will:")
    print("1. Create a test JSON file with sample data")
    print("2. Create test WordPress folder structure")
    print("3. Test the SlidingWindowWordPressImageProcessor")
    print()
    
    test_wordpress_processor()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("To use this in production:")
    print("python main.py sliding-wordpress your-data.json --output processed_images")
    print("="*60) 