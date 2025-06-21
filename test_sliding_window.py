from src.blurrer import SlidingWindowBlurrer

# Use the same input as in complex.py
inputPath = "data/input/TheyAreHugePiercing_watermark.jpg"
outputPath = "output_sliding_window_test.png"

print("Testing Sliding Window Blurrer...")

# Initialize sliding window blurrer
blurrer = SlidingWindowBlurrer(
    model_path='models/640m.onnx', 
    parts=[
        'FEMALE_BREAST_EXPOSED',
        'FEMALE_GENITALIA_EXPOSED',
        'FEMALE_BREAST_COVERED',
        'ANUS_EXPOSED',
        'MALE_GENITALIA_EXPOSED',
    ],
    window_size=640,  # 640x640 pixel windows
    stride=320,       # 50% overlap between windows
    overlap_threshold=0.3  # Merge detections with >30% IoU
)

# Process the image with sliding window approach
blurrer.process_image(
    input_path=inputPath, 
    output_path=outputPath,
    pixel_size=10,
    confidence_threshold=0.1
)

print("Sliding window processing complete!")
print(f"Output saved to: {outputPath}") 