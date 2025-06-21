#!/usr/bin/env python3
"""
Test script to compare old vs new scaled detection approaches for WordPress sizes.
This demonstrates how scaled detections maintain consistent pixelation across all image sizes.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.blurrer import SlidingWindowBlurrer

def test_pixelation_comparison():
    """Test both pixelation approaches and compare results."""
    
    # Initialize the blurrer
    blurrer = SlidingWindowBlurrer(
        model_path='models/640m.onnx',
        parts=[ 'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED',
                'FEMALE_BREAST_COVERED',
                'ANUS_EXPOSED',
                'MALE_GENITALIA_EXPOSED',],
        window_size=160,
        stride=120,
        overlap_threshold=0.8
    )
    
    # Test image path (you'll need to provide a test image)
    test_image = "data/custom_downloads/fapvid-pussy-licking_screenshot_full_url_fapvid-featured-image.jpg"  # Change this to your test image
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        print("Please provide a test image path")
        return
    
    print("Testing pixelation approaches...")
    print("=" * 50)
    
    # Test 1: Original approach (pixelate first, then resize)
    print("\n1. Testing ORIGINAL approach (pixelate first, then resize)")
    print("-" * 50)
    
    try:
        blurrer.process_image(
            input_path=test_image,
            output_path="test_original_approach.jpg",
            pixel_size=15,
            confidence_threshold=0.1,
            create_wordpress_sizes=True,
            image_type='screenshot_full_url',
            use_scaled_detections=False  # Use original approach
        )
        print("✅ Original approach completed")
    except Exception as e:
        print(f"❌ Original approach failed: {e}")
    
    # Test 2: Scaled detection approach (resize first, then pixelate with scaled detections)
    print("\n2. Testing SCALED DETECTION approach (resize first, then pixelate with scaled detections)")
    print("-" * 50)
    
    try:
        blurrer.process_image(
            input_path=test_image,
            output_path="test_scaled_detection_approach.jpg",
            pixel_size=15,
            confidence_threshold=0.1,
            create_wordpress_sizes=True,
            image_type='screenshot_full_url',
            use_scaled_detections=True  # Use scaled detection approach
        )
        print("✅ Scaled detection approach completed")
    except Exception as e:
        print(f"❌ Scaled detection approach failed: {e}")
    
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS:")
    print("=" * 50)
    print("\nCheck the generated files:")
    print("- test_original_approach.jpg (main image)")
    print("- test_original_approach-170x145.jpg (WordPress thumbnail)")
    print("- test_original_approach-250x212.jpg (WordPress category thumb)")
    print("- test_scaled_detection_approach.jpg (main image)")
    print("- test_scaled_detection_approach-170x145.jpg (WordPress thumbnail)")
    print("- test_scaled_detection_approach-250x212.jpg (WordPress category thumb)")
    
    print("\nKey differences:")
    print("1. ORIGINAL: Pixelation becomes barely visible in small images")
    print("2. SCALED DETECTION: Uses same pixel size across all image sizes")
    print("3. SCALED DETECTION: Scales detection coordinates to match smaller images")
    print("4. SCALED DETECTION: Maintains consistent pixelation visibility")

if __name__ == "__main__":
    test_pixelation_comparison() 