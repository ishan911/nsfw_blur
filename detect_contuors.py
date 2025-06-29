#!/usr/bin/env python3
"""
Hybrid YOLO Segmentation + NudeNet Detection Script

This script combines YOLO11m-seg.pt segmentation with NudeNet detection
for more accurate body part detection and sensitive content filtering.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from nudenet import NudeDetector
from ultralytics import YOLO

def load_yolo_segmentation_model(model_path="yolo11m-seg.pt"):
    """
    Load YOLO segmentation model.
    
    Args:
        model_path (str): Path to YOLO segmentation model
        
    Returns:
        YOLO model instance
    """
    try:
        # Try to load the model
        model = YOLO(model_path)
        print(f"✅ YOLO segmentation model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        print("Please ensure yolo11m-seg.pt is available in the current directory")
        return None

def detect_body_parts_with_yolo(image_path, yolo_model):
    """
    Detect body parts using YOLO segmentation.
    
    Args:
        image_path (str): Path to input image
        yolo_model: YOLO segmentation model
        
    Returns:
        List of body part segments with masks and bounding boxes
    """
    try:
        if yolo_model is None:
            print("❌ YOLO model not available")
            return []
        
        print(f"🔍 Running YOLO segmentation on: {image_path}")
        
        # Run YOLO segmentation
        results = yolo_model(image_path, verbose=False)
        
        body_parts = []
        
        for result in results:
            if result.masks is not None:
                # Get masks and boxes
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                class_names = result.names
                
                print(f"📊 YOLO detected {len(masks)} segments")
                
                for i, (mask, box, conf, class_id) in enumerate(zip(masks, boxes, confidences, class_ids)):
                    class_name = class_names[int(class_id)]
                    
                    # Convert mask to contour
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Get the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        
                        if area > 100:  # Filter out very small segments
                            x1, y1, x2, y2 = box
                            bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                            
                            body_parts.append({
                                'contour': largest_contour,
                                'mask': mask_uint8,
                                'bbox': bbox,
                                'area': area,
                                'confidence': float(conf),
                                'class_name': class_name,
                                'class_id': int(class_id),
                                'method': 'yolo_segmentation'
                            })
                            
                            print(f"  Segment {i+1}: {class_name} (conf: {conf:.3f}, area: {area:.0f})")
        
        print(f"✅ YOLO segmentation found {len(body_parts)} body parts")
        return body_parts
        
    except Exception as e:
        print(f"❌ Error in YOLO segmentation: {e}")
        return []

def run_nudenet_on_body_parts(image_path, body_parts, confidence_threshold=0.05):
    """
    Run NudeNet detection specifically on detected body parts.
    
    Args:
        image_path (str): Path to input image
        body_parts (list): List of body part segments from YOLO
        confidence_threshold (float): Minimum confidence for NudeNet detections
        
    Returns:
        List of NudeNet detections that overlap with body parts
    """
    try:
        # Initialize NudeNet detector
        detector = NudeDetector()
        
        # Allowed labels for filtering
        allowed_labels = set([
            "BUTTOCKS_EXPOSED",
            "BUTTOCKS_COVERED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "FEMALE_GENITALIA_COVERED",
            "ANUS_COVERED",
            "ANUS_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
        ])
        
        print(f"🔍 Running NudeNet detection on {len(body_parts)} body parts...")
        
        # Run NudeNet on the full image
        nudenet_detections = detector.detect(image_path)
        
        # Filter by confidence and allowed labels
        filtered_detections = []
        for detection in nudenet_detections:
            if detection['score'] >= confidence_threshold and detection['class'] in allowed_labels:
                filtered_detections.append(detection)
        
        print(f"📊 NudeNet found {len(filtered_detections)} potential detections")
        
        # Check which NudeNet detections overlap with YOLO body parts
        enhanced_detections = []
        
        for nudenet_detection in filtered_detections:
            nudenet_bbox = nudenet_detection['box']
            nudenet_x, nudenet_y, nudenet_w, nudenet_h = nudenet_bbox
            
            best_overlap = 0
            best_body_part = None
            
            for body_part in body_parts:
                body_bbox = body_part['bbox']
                body_x, body_y, body_w, body_h = body_bbox
                
                # Calculate overlap
                overlap = calculate_bbox_overlap(
                    [nudenet_x, nudenet_y, nudenet_w, nudenet_h],
                    [body_x, body_y, body_w, body_h]
                )
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_body_part = body_part
            
            # If there's significant overlap with a body part, enhance the detection
            if best_overlap > 0.3:  # 30% overlap threshold
                enhanced_detection = nudenet_detection.copy()
                enhanced_detection['yolo_enhanced'] = True
                enhanced_detection['body_part_class'] = best_body_part['class_name']
                enhanced_detection['body_part_confidence'] = best_body_part['confidence']
                enhanced_detection['overlap_ratio'] = best_overlap
                enhanced_detection['body_part_contour'] = best_body_part['contour']
                enhanced_detection['body_part_mask'] = best_body_part['mask']
                enhanced_detections.append(enhanced_detection)
                
                print(f"  ✅ Enhanced: {nudenet_detection['class']} (conf: {nudenet_detection['score']:.3f}) "
                      f"overlaps with {best_body_part['class_name']} (overlap: {best_overlap:.2f})")
            else:
                # Keep the detection but mark it as not enhanced
                nudenet_detection['yolo_enhanced'] = False
                enhanced_detections.append(nudenet_detection)
        
        print(f"✅ YOLO-enhanced detections: {len([d for d in enhanced_detections if d.get('yolo_enhanced', False)])}")
        print(f"📊 Total detections: {len(enhanced_detections)}")
        
        return enhanced_detections
        
    except Exception as e:
        print(f"❌ Error in NudeNet detection: {e}")
        return []

def create_hybrid_output(image_path, body_parts, detections, output_path, pixel_size=15, draw_body_parts=True, draw_rectangles=True):
    """
    Create output image with YOLO body parts and NudeNet detections.
    
    Args:
        image_path (str): Path to input image
        body_parts (list): List of YOLO body part segments
        detections (list): List of NudeNet detections
        output_path (str): Path to save output image
        pixel_size (int): Size of pixel blocks for pixelation
        draw_body_parts (bool): Whether to draw YOLO body part outlines
        draw_rectangles (bool): Whether to draw detection rectangles
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        height, width = img.shape[:2]
        
        # Color mapping for different body parts
        color_map = {
            'FEMALE_BREAST_EXPOSED': (0, 0, 255),      # Red
            'FEMALE_BREAST_COVERED': (0, 165, 255),    # Orange
            'FEMALE_GENITALIA_EXPOSED': (255, 0, 0),   # Blue
            'FEMALE_GENITALIA_COVERED': (255, 0, 255), # Magenta
            'MALE_GENITALIA_EXPOSED': (0, 255, 0),     # Green
            'MALE_GENITALIA_COVERED': (0, 255, 255),   # Yellow
            'BUTTOCKS_EXPOSED': (128, 0, 128),         # Purple
            'BUTTOCKS_COVERED': (255, 192, 203),       # Pink
            'ANUS_EXPOSED': (165, 42, 42),             # Brown
            'ANUS_COVERED': (128, 128, 0),             # Olive
        }
        
        # YOLO body part colors
        yolo_colors = {
            'person': (0, 255, 255),      # Yellow
            'head': (255, 0, 255),        # Magenta
            'torso': (255, 0, 0),         # Blue
            'arm': (0, 255, 0),           # Green
            'leg': (0, 0, 255),           # Red
            'hand': (255, 165, 0),        # Orange
            'foot': (128, 0, 128),        # Purple
        }
        
        # First, draw YOLO body parts if requested
        if draw_body_parts:
            print("🎨 Drawing YOLO body parts...")
            for i, body_part in enumerate(body_parts):
                contour = body_part['contour']
                class_name = body_part['class_name']
                confidence = body_part['confidence']
                
                # Get color for this body part
                color = yolo_colors.get(class_name.lower(), (128, 128, 128))
                
                # Draw contour outline
                cv2.drawContours(img, [contour], -1, color, 2)
                
                # Draw bounding box
                x, y, w, h = body_part['bbox']
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Sort detections by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        valid_detections = 0
        
        # Process each detection
        for i, detection in enumerate(sorted_detections):
            try:
                box = detection['box']
                score = detection['score']
                class_name = detection['class']
                is_yolo_enhanced = detection.get('yolo_enhanced', False)
                
                # Validate bounding box coordinates
                if len(box) != 4:
                    continue
                
                x, y, w, h = box
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Check if coordinates are within image bounds
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                
                # Ensure rectangle doesn't go outside image bounds
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                
                # Skip if rectangle is too small or invalid
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Add padding to the detected rectangle
                padding = 5
                x1_padded = max(0, x1 - padding)
                y1_padded = max(0, y1 - padding)
                x2_padded = min(width, x2 + padding)
                y2_padded = min(height, y2 + padding)
                
                # Pixelate the detected region
                img = pixelate_region(img, x1_padded, y1_padded, x2_padded, y2_padded, pixel_size)
                
                # Draw rectangle border if requested
                if draw_rectangles:
                    # Get color for this class
                    color = color_map.get(class_name, (255, 255, 255))
                    
                    # Use different thickness for YOLO-enhanced detections
                    thickness = 3 if is_yolo_enhanced else 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add label
                    label = f"{class_name}: {score:.2f}"
                    if is_yolo_enhanced:
                        label += " (YOLO)"
                        body_part_class = detection.get('body_part_class', 'unknown')
                        label += f" [{body_part_class}]"
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img, 
                                 (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0], y1),
                                 color, -1)
                    
                    # Draw label text
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                valid_detections += 1
                
                # Print detection info
                enhancement_info = " (YOLO-Enhanced)" if is_yolo_enhanced else ""
                print(f"  Detection #{i+1}: {class_name} - {score:.3f}{enhancement_info}")
                
            except Exception as e:
                print(f"❌ Error processing detection {i}: {e}")
                continue
        
        # Add legend
        legend_y = 30
        cv2.putText(img, "YOLO Body Parts:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        
        for class_name, color in yolo_colors.items():
            cv2.putText(img, f"{class_name}: ", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            legend_y += 20
        
        # Save the image
        cv2.imwrite(output_path, img)
        print(f"✅ Hybrid output saved: {output_path}")
        print(f"✅ Successfully processed {valid_detections} out of {len(detections)} detections")
        
        # Print summary
        yolo_enhanced_count = len([d for d in detections if d.get('yolo_enhanced', False)])
        print(f"📊 YOLO-enhanced detections: {yolo_enhanced_count}")
        print(f"📊 Regular detections: {len(detections) - yolo_enhanced_count}")
        
    except Exception as e:
        print(f"❌ Error creating hybrid output: {str(e)}")
        import traceback
        traceback.print_exc()

def calculate_bbox_overlap(box1, box2):
    """
    Calculate overlap between two bounding boxes.
    
    Args:
        box1, box2: [x, y, w, h] format
        
    Returns:
        float: Overlap ratio
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Return intersection over union
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def pixelate_region(img, x1, y1, x2, y2, pixel_size=15):
    """
    Pixelate a region of an image.
    """
    try:
        # Extract the region to pixelate
        region = img[y1:y2, x1:x2]
        
        if region.size == 0:
            return img
        
        # Get dimensions of the region
        h, w = region.shape[:2]
        
        # Calculate new dimensions for pixelation
        new_h = h // pixel_size
        new_w = w // pixel_size
        
        if new_h == 0 or new_w == 0:
            # If region is too small, use a smaller pixel size
            pixel_size = min(h, w) // 2
            if pixel_size < 2:
                pixel_size = 2
            new_h = h // pixel_size
            new_w = w // pixel_size
        
        # Resize down to create pixelation effect
        if new_h > 0 and new_w > 0:
            # Use INTER_AREA for downsampling (better for pixelation)
            pixelated = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Resize back up to original size
            pixelated = cv2.resize(pixelated, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            # Fallback for very small regions
            pixelated = region
        
        # Replace the region in the original image
        img[y1:y2, x1:x2] = pixelated
        
        return img
        
    except Exception as e:
        print(f"❌ Error pixelating region: {e}")
        return img

def main():
    # Configuration
    input_path = "data/Brazzers-6.jpg"  # Change this to your image path
    confidence_threshold = 0.02  # Adjust this threshold as needed
    pixel_size = 15  # Size of pixel blocks for pixelation
    yolo_model_path = "yolo11m-seg.pt"  # Path to YOLO segmentation model
    
    print("=== Hybrid YOLO Segmentation + NudeNet Detection ===")
    print(f"Input image: {input_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Pixel size: {pixel_size}")
    print(f"YOLO model: {yolo_model_path}")
    print()
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        print("Please update the input_path variable to point to your image file.")
        return
    
    # Check if YOLO model exists
    if not os.path.exists(yolo_model_path):
        print(f"❌ Error: YOLO model not found: {yolo_model_path}")
        print("Please download yolo11m-seg.pt and place it in the current directory.")
        return
    
    try:
        # Step 1: Load YOLO segmentation model
        print("=== Loading YOLO Segmentation Model ===")
        yolo_model = load_yolo_segmentation_model(yolo_model_path)
        
        if yolo_model is None:
            print("❌ Cannot proceed without YOLO model")
            return
        
        # Step 2: Detect body parts with YOLO
        print("\n=== YOLO Body Part Detection ===")
        body_parts = detect_body_parts_with_yolo(input_path, yolo_model)
        
        if not body_parts:
            print("⚠️ No body parts detected by YOLO, proceeding with NudeNet only...")
        
        # Step 3: Run NudeNet detection on body parts
        print("\n=== NudeNet Detection on Body Parts ===")
        detections = run_nudenet_on_body_parts(input_path, body_parts, confidence_threshold)
        
        if detections:
            # Step 4: Create hybrid output
            print("\n=== Creating Hybrid Output ===")
            output_path = "hybrid_yolo_nudenet_detections.jpg"
            create_hybrid_output(
                input_path, 
                body_parts, 
                detections, 
                output_path, 
                pixel_size=pixel_size,
                draw_body_parts=True,
                draw_rectangles=True
            )
            
            # Show detection details
            print("\n=== Detection Details ===")
            yolo_enhanced = [d for d in detections if d.get('yolo_enhanced', False)]
            regular = [d for d in detections if not d.get('yolo_enhanced', False)]
            
            if yolo_enhanced:
                print(f"\nYOLO-Enhanced Detections ({len(yolo_enhanced)}):")
                for i, detection in enumerate(yolo_enhanced, 1):
                    body_part_class = detection.get('body_part_class', 'unknown')
                    overlap = detection.get('overlap_ratio', 0)
                    print(f"  #{i}. {detection['class']}: {detection['score']:.3f} "
                          f"(Body part: {body_part_class}, Overlap: {overlap:.2f})")
            
            if regular:
                print(f"\nRegular Detections ({len(regular)}):")
                for i, detection in enumerate(regular, 1):
                    print(f"  #{i}. {detection['class']}: {detection['score']:.3f}")
            
        else:
            print("❌ No detections found with the current confidence threshold.")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
