from src.blurrer import SlidingWindowBlurrer

inputPath = "data/RealityKings-Screenshot1-Jpg-2.png"
outputPath = "output_sliding_window.jpg"

# Initialize sliding window blurrer for better detection
blurrer = SlidingWindowBlurrer(
    model_path='models/640m.onnx', 
    parts=[
        'FEMALE_BREAST_EXPOSED',
        'FEMALE_GENITALIA_EXPOSED',
        'FEMALE_BREAST_COVERED',
        'ANUS_EXPOSED',
        'MALE_GENITALIA_EXPOSED',
    ],
    window_size=160,  # Small window size for detailed detection
    stride=120,        # 25% overlap between windows
    overlap_threshold=0.8  # Higher threshold for small windows to avoid duplicates
)

# Process the image with sliding window approach and WordPress sizing
try:
    # Test with screenshot_full_url type (should create 170x145 and 250x212)
    blurrer.process_image(
        input_path=inputPath, 
        output_path=outputPath,
        pixel_size=15,  # Updated pixel size
        confidence_threshold=0.15,  # Higher confidence for small windows
        create_wordpress_sizes=True,  # Create WordPress-sized images
        image_type='screenshot_full_url'  # This will create 170x145 and 250x212 sizes
    )
    print("Sliding window processing complete!")
    print("WordPress-sized images created for screenshot_full_url:")
    print("  Main image: wp-content/uploads/output_sliding_window.jpg")
    print("  - wp-content/uploads/output_sliding_window-170x145.jpg (blog-tn)")
    print("  - wp-content/uploads/output_sliding_window-250x212.jpg (category-thumb)")
    
    # Test with review_full_image type (should create only 590x504)
    # Uncomment the lines below to test review_full_image sizing
    # blurrer.process_image(
    #     input_path=inputPath, 
    #     output_path=outputPath.replace('.jpg', '_review.jpg'),
    #     pixel_size=15,
    #     confidence_threshold=0.15,
    #     create_wordpress_sizes=True,
    #     image_type='review_full_image'  # This will create only 590x504 size
    # )
    # print("WordPress-sized images created for review_full_image:")
    # print("  Main image: wp-content/uploads/screenshots/output_sliding_window_review.jpg")
    # print("  - wp-content/uploads/screenshots/output_sliding_window_review-590x504.jpg (swiper-desktop)")
    
except Exception as e:
    print(f"Error during processing: {e}")
    print("Try increasing window_size or decreasing stride if processing is too slow.")