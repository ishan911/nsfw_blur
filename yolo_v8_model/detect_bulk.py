import cv2
import os
from ultralytics import YOLO
import numpy as np

def process_image(input_path, output_path, model):
    """
    Process a single image: detect objects, blur them, and mark detection boxes.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save processed image
        model: YOLO model for detection
    """
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not load image: {input_path}")
        return False
    
    # Run detection
    results = model(input_path)[0]
    
    # Create a copy for drawing boxes
    img_with_boxes = img.copy()
    
    # Process each detection
    detection_count = 0
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        
        # Blur the detected region
        roi = img[y1:y2, x1:x2]
        roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
        img[y1:y2, x1:x2] = roi_blur
        
        # Draw detection box on the copy
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add detection label
        label = f"Detection {detection_count + 1}"
        cv2.putText(img_with_boxes, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detection_count += 1
    
    # Save both versions
    cv2.imwrite(output_path, img)  # Blurred version
    
    # Save version with boxes marked
    base_name = os.path.splitext(output_path)[0]
    boxes_path = f"{base_name}_with_boxes.jpg"
    cv2.imwrite(boxes_path, img_with_boxes)
    
    print(f"Processed: {os.path.basename(input_path)} - {detection_count} detections")
    return True

def main():
    # Initialize YOLO model
    model = YOLO("runs/detect/train8/weights/best.pt")
    
    # Define input and output paths
    input_folder = "../data/custom_downloads"
    output_folder = "../custom_processed"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        return
    
    # Get all image files from input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in: {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("-" * 50)
    
    # Process each image
    processed_count = 0
    error_count = 0
    
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        
        # Create output filename (preserve original extension)
        base_name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        output_filename = f"blurred_{base_name}{extension}"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            success = process_image(input_path, output_path, model)
            if success:
                processed_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            error_count += 1
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors: {error_count} images")
    print(f"Output saved to: {output_folder}")
    print(f"Each image has two versions:")
    print(f"  - blurred_[filename].jpg - Blurred version")
    print(f"  - blurred_[filename]_with_boxes.jpg - Version with detection boxes marked")

if __name__ == "__main__":
    main() 