from src.blurrer import SlidingWindowBlurrer

def main():
    # Example 1: Basic sliding window usage
    print("=== Example 1: Basic Sliding Window Processing ===")
    
    input_path = "data/input/TheyAreHugePiercing_watermark.jpg"
    output_path = "output_sliding_window.png"
    
    # Initialize with default parameters
    blurrer = SlidingWindowBlurrer(
        model_path='models/640m.onnx',
        parts=[
            'FEMALE_BREAST_EXPOSED',
            'FEMALE_GENITALIA_EXPOSED',
            'FEMALE_BREAST_COVERED',
            'ANUS_EXPOSED',
            'MALE_GENITALIA_EXPOSED',
        ],
        window_size=640,  # Window size (640x640 pixels)
        stride=320,       # Stride between windows (50% overlap)
        overlap_threshold=0.3  # IoU threshold for merging detections
    )
    
    # Process the image
    blurrer.process_image(
        input_path=input_path,
        output_path=output_path,
        pixel_size=10,
        confidence_threshold=0.1
    )
    
    print(f"Basic processing complete. Output saved to: {output_path}")
    
    # Example 2: High-resolution processing with smaller windows
    print("\n=== Example 2: High-Resolution Processing ===")
    
    output_path_hr = "output_high_res.png"
    
    # Initialize with smaller windows for high-resolution images
    blurrer_hr = SlidingWindowBlurrer(
        model_path='models/640m.onnx',
        parts=[
            'FEMALE_BREAST_EXPOSED',
            'FEMALE_GENITALIA_EXPOSED',
            'FEMALE_BREAST_COVERED',
            'ANUS_EXPOSED',
            'MALE_GENITALIA_EXPOSED',
        ],
        window_size=512,  # Smaller window size
        stride=256,       # Smaller stride for more overlap
        overlap_threshold=0.4  # Higher threshold for better merging
    )
    
    # Process with higher confidence threshold
    blurrer_hr.process_image(
        input_path=input_path,
        output_path=output_path_hr,
        pixel_size=8,  # Smaller pixels for finer detail
        confidence_threshold=0.15  # Higher confidence threshold
    )
    
    print(f"High-resolution processing complete. Output saved to: {output_path_hr}")
    
    # Example 3: Visualization mode to see sliding windows
    print("\n=== Example 3: Visualization Mode ===")
    
    output_path_viz = "output_with_windows.png"
    
    # Process with window visualization
    blurrer.process_image_with_visualization(
        input_path=input_path,
        output_path=output_path_viz,
        pixel_size=10,
        confidence_threshold=0.1,
        show_windows=True  # This will draw red rectangles around each window
    )
    
    print(f"Visualization complete. Output saved to: {output_path_viz}")
    print("Red rectangles show the sliding windows used for detection.")

def demonstrate_parameters():
    """Demonstrate different parameter configurations for sliding window processing."""
    
    print("\n=== Parameter Demonstration ===")
    
    # Configuration 1: Large windows, less overlap (faster processing)
    config1 = {
        'window_size': 800,
        'stride': 600,
        'overlap_threshold': 0.2,
        'description': 'Large windows, less overlap (faster)'
    }
    
    # Configuration 2: Medium windows, moderate overlap (balanced)
    config2 = {
        'window_size': 640,
        'stride': 320,
        'overlap_threshold': 0.3,
        'description': 'Medium windows, moderate overlap (balanced)'
    }
    
    # Configuration 3: Small windows, high overlap (more accurate)
    config3 = {
        'window_size': 480,
        'stride': 240,
        'overlap_threshold': 0.4,
        'description': 'Small windows, high overlap (more accurate)'
    }
    
    configs = [config1, config2, config3]
    
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config['description']}")
        print(f"  Window Size: {config['window_size']}x{config['window_size']}")
        print(f"  Stride: {config['stride']}")
        print(f"  Overlap: {((config['window_size'] - config['stride']) / config['window_size'] * 100):.1f}%")
        print(f"  IoU Threshold: {config['overlap_threshold']}")
        
        # Calculate number of windows for a sample image
        sample_width, sample_height = 1920, 1080
        windows = []
        for y in range(0, sample_height, config['stride']):
            for x in range(0, sample_width, config['stride']):
                window_w = min(config['window_size'], sample_width - x)
                window_h = min(config['window_size'], sample_height - y)
                if window_w >= config['window_size'] // 2 and window_h >= config['window_size'] // 2:
                    windows.append((x, y, window_w, window_h))
        
        print(f"  Windows for 1920x1080 image: {len(windows)}")

if __name__ == "__main__":
    try:
        main()
        demonstrate_parameters()
        print("\n=== All examples completed successfully! ===")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the input image exists and the model path is correct.") 