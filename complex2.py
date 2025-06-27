from src.blurrer import SlidingWindowBlurrer
import cv2
import numpy as np
from ultralytics import YOLO
import os

inputPath = "data/brazzers.jpg"
outputPath = "output_sliding_window.jpg"

# Initialize sliding window blurrer for better detection
blurrer = SlidingWindowBlurrer(
    model_path='models/640m.onnx', 
    parts=[
        'BUTTOCKS_EXPOSED',
        'BUTTOCKS_COVERED',
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

def run_yolo_detection(image_path, yolo_model_path=None, confidence_threshold=0.1):
    """
    Run YOLO detection on an image and return detected rectangles.
    
    Args:
        image_path (str): Path to input image
        yolo_model_path (str): Path to YOLO model (defaults to custom trained model)
        confidence_threshold (float): Minimum confidence for detections
        
    Returns:
        List of detection rectangles [(x1, y1, x2, y2, confidence, class_name), ...]
    """
    try:
        # Use default custom model path if not specified
        if yolo_model_path is None:
            yolo_model_path = "yolo_v8_model/runs/detect/train15/weights/best.pt"
        
        model = YOLO(yolo_model_path)
        
        # Run detection
        results = model(image_path, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[class_id] if class_id in result.names else f"class_{class_id}"
                    
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(confidence), class_name))
        
        return detections
        
    except Exception as e:
        print(f"Error in YOLO detection: {str(e)}")
        return []

def draw_detection_rectangles(image_path, detections, output_path, title="Detections"):
    """
    Draw detection rectangles on an image and save it.
    
    Args:
        image_path (str): Path to input image
        detections (list): List of detection tuples (x1, y1, x2, y2, confidence, class_name)
        output_path (str): Path to save the output image
        title (str): Title for the image
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Draw each detection rectangle
        for x1, y1, x2, y2, confidence, class_name in detections:
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add title
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save the image
        cv2.imwrite(output_path, img)
        print(f"Detection image saved: {output_path}")
        print(f"Found {len(detections)} detections")
        
    except Exception as e:
        print(f"Error drawing detections: {str(e)}")

# Process the image with NudeNet first, then YOLO
try:
    print("=== Stage 1: NudeNet Processing ===")
    # Process the image with sliding window approach and WordPress sizing
    blurrer.process_image(
        input_path=inputPath, 
        output_path=outputPath,
        pixel_size=15,  # Updated pixel size
        confidence_threshold=0.15,  # Higher confidence for small windows
        create_wordpress_sizes=True,  # Create WordPress-sized images
        image_type='screenshot_full_url'  # This will create 170x145 and 250x212 sizes
    )
    print("NudeNet sliding window processing complete!")
    print("WordPress-sized images created for screenshot_full_url:")
    print("  Main image: wp-content/uploads/output_sliding_window.jpg")
    print("  - wp-content/uploads/output_sliding_window-170x145.jpg (blog-tn)")
    print("  - wp-content/uploads/output_sliding_window-250x212.jpg (category-thumb)")
    
    print("\n=== Stage 2: YOLO Detection (Rectangles Only) ===")
    # Run YOLO detection on the NudeNet processed image (not the original)
    yolo_detections = run_yolo_detection(outputPath, confidence_threshold=0.1)
    
    if yolo_detections:
        # Draw YOLO detections on the NudeNet processed image
        yolo_on_processed_path = "yolo_on_nudenet_processed.jpg"
        draw_detection_rectangles(outputPath, yolo_detections, yolo_on_processed_path, "YOLO on NudeNet Processed")
        
        # Also draw YOLO detections on the original image for comparison
        yolo_on_original_path = "yolo_on_original.jpg"
        draw_detection_rectangles(inputPath, yolo_detections, yolo_on_original_path, "YOLO on Original (for comparison)")
    else:
        print("No YOLO detections found.")
    
    print("\n=== Processing Summary ===")
    print("1. NudeNet sliding window processing completed")
    print("2. YOLO detection completed on the NudeNet processed image")
    print("3. Both detection methods applied to the same processed image")
    
except Exception as e:
    print(f"Error during processing: {e}")
    print("Try increasing window_size or decreasing stride if processing is too slow.")