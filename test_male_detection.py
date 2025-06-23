#!/usr/bin/env python3
"""
Test script to diagnose and improve male genitalia detection.
This script tests different confidence thresholds and sliding window parameters
to find the optimal settings for detecting partial male genitalia.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.blurrer import SlidingWindowBlurrer
from nudenet import NudeDetector

def test_detection_sensitivity():
    """Test detection with different confidence thresholds."""
    
    print("=== Testing Detection Sensitivity ===")
    
    # Test image path
    test_image = "data/Analvids.jpg"  # Change this to your test image
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    # Initialize detector
    detector = NudeDetector(model_path='models/640m.onnx')
    
    # Test different confidence thresholds
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    print(f"\nTesting image: {test_image}")
    print("=" * 60)
    
    for threshold in thresholds:
        print(f"\nConfidence threshold: {threshold}")
        print("-" * 40)
        
        # Get all detections
        results = detector.detect(test_image)
        
        # Filter by threshold
        filtered_results = [r for r in results if r['score'] >= threshold]
        
        # Count male genitalia detections
        male_detections = [r for r in filtered_results if 'MALE_GENITALIA' in r['class']]
        
        print(f"Total detections (≥{threshold}): {len(filtered_results)}")
        print(f"Male genitalia detections: {len(male_detections)}")
        
        # Show all detections
        for result in filtered_results:
            print(f"  {result['class']}: {result['score']:.3f}")
        
        # Show male genitalia specifically
        if male_detections:
            print("  Male genitalia details:")
            for result in male_detections:
                print(f"    {result['class']}: {result['score']:.3f} at {result['box']}")

def test_sliding_window_parameters():
    """Test different sliding window parameters for male genitalia detection."""
    
    print("\n\n=== Testing Sliding Window Parameters ===")
    
    # Test image path
    test_image = "input.png"  # Change this to your test image
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    # Test configurations
    configs = [
        {
            'name': 'Large windows, low overlap',
            'window_size': 800,
            'stride': 600,
            'overlap_threshold': 0.2,
            'confidence_threshold': 0.01
        },
        {
            'name': 'Medium windows, medium overlap',
            'window_size': 640,
            'stride': 320,
            'overlap_threshold': 0.3,
            'confidence_threshold': 0.01
        },
        {
            'name': 'Small windows, high overlap',
            'window_size': 480,
            'stride': 240,
            'overlap_threshold': 0.4,
            'confidence_threshold': 0.01
        },
        {
            'name': 'Very small windows, very high overlap',
            'window_size': 320,
            'stride': 160,
            'overlap_threshold': 0.5,
            'confidence_threshold': 0.01
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 50)
        print(f"Window size: {config['window_size']}x{config['window_size']}")
        print(f"Stride: {config['stride']}")
        print(f"Overlap: {((config['window_size'] - config['stride']) / config['window_size'] * 100):.1f}%")
        print(f"Confidence threshold: {config['confidence_threshold']}")
        
        try:
            # Initialize blurrer with current config
            blurrer = SlidingWindowBlurrer(
                model_path='models/640m.onnx',
                parts=['MALE_GENITALIA_EXPOSED'],  # Focus only on male genitalia
                window_size=config['window_size'],
                stride=config['stride'],
                overlap_threshold=config['overlap_threshold']
            )
            
            # Process image
            output_path = f"test_male_detection_{config['window_size']}.png"
            result = blurrer.process_image(
                input_path=test_image,
                output_path=output_path,
                pixel_size=15,
                confidence_threshold=config['confidence_threshold'],
                create_wordpress_sizes=False  # Don't create WordPress sizes for testing
            )
            
            print(f"✅ Processing completed: {output_path}")
            
            # Check if file was created and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"   Output file size: {file_size} bytes")
            else:
                print("   ❌ Output file not created")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_all_body_parts():
    """Test detection of all available body parts to understand model capabilities."""
    
    print("\n\n=== Testing All Body Parts ===")
    
    # Test image path
    test_image = "input.png"  # Change this to your test image
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    # Initialize detector
    detector = NudeDetector(model_path='models/640m.onnx')
    
    # Get all detections with very low threshold
    results = detector.detect(test_image)
    
    # Group by body part type
    body_parts = {}
    for result in results:
        part_type = result['class']
        if part_type not in body_parts:
            body_parts[part_type] = []
        body_parts[part_type].append(result)
    
    print(f"\nAll detected body parts in {test_image}:")
    print("=" * 60)
    
    for part_type, detections in sorted(body_parts.items()):
        print(f"\n{part_type}: {len(detections)} detection(s)")
        for detection in detections:
            print(f"  Score: {detection['score']:.3f}, Box: {detection['box']}")

def create_improved_male_detection_config():
    """Create an improved configuration for male genitalia detection."""
    
    print("\n\n=== Recommended Configuration for Male Genitalia Detection ===")
    
    # Based on testing, here's an improved configuration
    improved_config = {
        'window_size': 480,  # Smaller windows for better detail
        'stride': 240,       # 50% overlap for better coverage
        'overlap_threshold': 0.4,  # Higher threshold for better merging
        'confidence_threshold': 0.05,  # Lower threshold for partial detections
        'parts': [
            'MALE_GENITALIA_EXPOSED',
            'MALE_GENITALIA_COVERED',  # Add covered version if available
            'FEMALE_GENITALIA_EXPOSED',  # Keep other sensitive parts
            'FEMALE_BREAST_EXPOSED',
            'ANUS_EXPOSED'
        ]
    }
    
    print("Recommended settings:")
    print(f"  Window size: {improved_config['window_size']}x{improved_config['window_size']}")
    print(f"  Stride: {improved_config['stride']}")
    print(f"  Overlap: {((improved_config['window_size'] - improved_config['stride']) / improved_config['window_size'] * 100):.1f}%")
    print(f"  Confidence threshold: {improved_config['confidence_threshold']}")
    print(f"  Body parts: {improved_config['parts']}")
    
    print("\nUsage example:")
    print("```python")
    print("from src.blurrer import SlidingWindowBlurrer")
    print("")
    print("blurrer = SlidingWindowBlurrer(")
    print(f"    model_path='models/640m.onnx',")
    print(f"    parts={improved_config['parts']},")
    print(f"    window_size={improved_config['window_size']},")
    print(f"    stride={improved_config['stride']},")
    print(f"    overlap_threshold={improved_config['overlap_threshold']}")
    print(")")
    print("")
    print("blurrer.process_image(")
    print("    input_path='your_image.png',")
    print("    output_path='output.png',")
    print(f"    confidence_threshold={improved_config['confidence_threshold']}")
    print(")")
    print("```")

if __name__ == "__main__":
    print("Male Genitalia Detection Diagnostic Tool")
    print("=" * 50)
    
    # Run all tests
    test_detection_sensitivity()
    test_sliding_window_parameters()
    test_all_body_parts()
    create_improved_male_detection_config()
    
    print("\n\n=== Summary ===")
    print("If male genitalia detection is still poor:")
    print("1. Try even lower confidence thresholds (0.01-0.03)")
    print("2. Use smaller sliding windows (320x320 or 256x256)")
    print("3. Increase overlap between windows (75% overlap)")
    print("4. Consider if the image actually contains male genitalia")
    print("5. Test with different images that definitely contain male genitalia") 