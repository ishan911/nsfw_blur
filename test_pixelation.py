#!/usr/bin/env python3
"""
Test script for NudeNet detection with pixelation functionality.
This script demonstrates different pixelation settings and outputs.
"""

import os
from detect_all_parts import (
    run_nudenet_detection_enhanced, 
    draw_nudenet_rectangles, 
    create_pixelated_version
)

def test_pixelation_settings():
    """Test different pixelation settings on the same image."""
    
    # Configuration
    input_path = "data/reality-kings.jpg"  # Change this to your image path
    confidence_threshold = 0.05
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        print("Please update the input_path variable to point to your image file.")
        return
    
    print("=== Testing NudeNet Detection with Pixelation ===")
    print(f"Input image: {input_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print()
    
    try:
        # Run detection
        print("Running NudeNet detection...")
        detections = run_nudenet_detection_enhanced(input_path, confidence_threshold)
        
        if not detections:
            print("No detections found. Try lowering the confidence threshold.")
            return
        
        print(f"Found {len(detections)} detections")
        print()
        
        # Test different pixelation settings
        pixel_sizes = [5, 10, 15, 20, 25]
        
        for pixel_size in pixel_sizes:
            print(f"=== Testing Pixel Size: {pixel_size} ===")
            
            # 1. Create version with pixelation + rectangles + labels
            output_path = f"test_pixelation_{pixel_size}_with_labels.jpg"
            draw_nudenet_rectangles(
                input_path, 
                detections, 
                output_path, 
                f"NudeNet Detection (Pixel Size: {pixel_size})",
                pixelate=True,
                pixel_size=pixel_size,
                draw_rectangles=True,
                draw_labels=True
            )
            print(f"  Saved: {output_path}")
            
            # 2. Create pixelated-only version (no rectangles or labels)
            pixelated_only_path = f"test_pixelation_{pixel_size}_only.jpg"
            create_pixelated_version(input_path, detections, pixelated_only_path, pixel_size)
            print(f"  Saved: {pixelated_only_path}")
            
            print()
        
        # Test version with no pixelation (just rectangles)
        print("=== Testing No Pixelation (Rectangles Only) ===")
        no_pixelation_path = "test_no_pixelation.jpg"
        draw_nudenet_rectangles(
            input_path, 
            detections, 
            no_pixelation_path, 
            "NudeNet Detection (No Pixelation)",
            pixelate=False,
            draw_rectangles=True,
            draw_labels=True
        )
        print(f"  Saved: {no_pixelation_path}")
        
        # Test version with pixelation but no rectangles or labels
        print("=== Testing Pixelation Only (No Rectangles or Labels) ===")
        pixelation_only_path = "test_pixelation_only.jpg"
        draw_nudenet_rectangles(
            input_path, 
            detections, 
            pixelation_only_path, 
            "NudeNet Detection (Pixelation Only)",
            pixelate=True,
            pixel_size=15,
            draw_rectangles=False,
            draw_labels=False
        )
        print(f"  Saved: {pixelation_only_path}")
        
        print("\n=== Test Complete ===")
        print("Generated files:")
        for pixel_size in pixel_sizes:
            print(f"  - test_pixelation_{pixel_size}_with_labels.jpg (pixelation + labels)")
            print(f"  - test_pixelation_{pixel_size}_only.jpg (pixelation only)")
        print(f"  - test_no_pixelation.jpg (rectangles only)")
        print(f"  - test_pixelation_only.jpg (pixelation only)")
        print("\nCompare the files to see the different pixelation effects!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pixelation_settings() 