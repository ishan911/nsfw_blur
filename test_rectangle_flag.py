#!/usr/bin/env python3
"""
Test script to verify that the draw_rectangle_borders flag works correctly.
"""

import os
from detect_all_parts import run_nudenet_detection_enhanced, draw_nudenet_rectangles

def test_rectangle_flag():
    """Test that draw_rectangle_borders flag works correctly."""
    
    # Configuration
    input_path = "data/reality-kings.jpg"  # Change this to your image path
    confidence_threshold = 0.05
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        print("Please update the input_path variable to point to your image file.")
        return
    
    print("=== Testing Rectangle Border Flag ===")
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
        
        # Test 1: With rectangles and labels
        print("=== Test 1: With Rectangles and Labels ===")
        output_path_1 = "test_with_rectangles_and_labels.jpg"
        draw_nudenet_rectangles(
            input_path, 
            detections, 
            output_path_1, 
            "Test: With Rectangles and Labels",
            pixelate=True,
            pixel_size=15,
            draw_rectangles=True,
            draw_labels=True
        )
        print(f"  Saved: {output_path_1}")
        print()
        
        # Test 2: With rectangles but no labels
        print("=== Test 2: With Rectangles but No Labels ===")
        output_path_2 = "test_with_rectangles_no_labels.jpg"
        draw_nudenet_rectangles(
            input_path, 
            detections, 
            output_path_2, 
            "Test: With Rectangles, No Labels",
            pixelate=True,
            pixel_size=15,
            draw_rectangles=True,
            draw_labels=False
        )
        print(f"  Saved: {output_path_2}")
        print()
        
        # Test 3: No rectangles, no labels (pixelation only)
        print("=== Test 3: No Rectangles, No Labels (Pixelation Only) ===")
        output_path_3 = "test_no_rectangles_no_labels.jpg"
        draw_nudenet_rectangles(
            input_path, 
            detections, 
            output_path_3, 
            "Test: No Rectangles, No Labels",
            pixelate=True,
            pixel_size=15,
            draw_rectangles=False,
            draw_labels=False
        )
        print(f"  Saved: {output_path_3}")
        print()
        
        print("=== Test Complete ===")
        print("Generated files:")
        print(f"  - {output_path_1} (rectangles + labels)")
        print(f"  - {output_path_2} (rectangles only)")
        print(f"  - {output_path_3} (pixelation only)")
        print("\nCheck these files to verify the rectangle flag is working correctly!")
        print("The third file should have NO rectangles or labels, only pixelation.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rectangle_flag() 