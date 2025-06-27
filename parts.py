#!/usr/bin/env python3
"""
NudeNet Detection Script - Detect and Draw Rectangles

This script uses NudeNet to detect all possible body parts in an image
and draws rectangles around them with labels showing part name and confidence.
Includes preprocessing techniques to improve detection on darker bodies.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from nudenet import NudeDetector

def preprocess_image_for_small_parts(image_path, enhancement_factor=1.5):
    """
    Preprocess image to improve detection of small body parts.
    
    Args:
        image_path (str): Path to input image
        enhancement_factor (float): Brightness enhancement factor
        
    Returns:
        List of preprocessed images with different scales and enhancements
    """
    try:
        # Load image with PIL for better enhancement control
        pil_image = Image.open(image_path)
        original_size = pil_image.size
        
        preprocessed_images = []
        
        # Original image
        preprocessed_images.append(('original', cv2.imread(image_path), original_size))
        
        # Upscaled versions for small part detection
        scale_factors = [1.5, 2.0, 2.5, 3.0]  # Different upscaling factors
        
        for scale in scale_factors:
            # Calculate new size
            new_width = int(original_size[0] * scale)
            new_height = int(original_size[1] * scale)
            new_size = (new_width, new_height)
            
            # Upscale image
            upscaled = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            upscaled_cv = cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGB2BGR)
            preprocessed_images.append((f'upscaled_{scale}x', upscaled_cv, new_size))
            
            # Upscaled + enhanced versions
            enhancer = ImageEnhance.Brightness(upscaled)
            brightened = enhancer.enhance(enhancement_factor)
            enhancer = ImageEnhance.Contrast(brightened)
            enhanced = enhancer.enhance(enhancement_factor)
            enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            preprocessed_images.append((f'upscaled_{scale}x_enhanced', enhanced_cv, new_size))
        
        # Add the original enhanced versions from the previous function
        # Brightness enhancement
        enhancer = ImageEnhance.Brightness(pil_image)
        brightened = enhancer.enhance(enhancement_factor)
        brightened_cv = cv2.cvtColor(np.array(brightened), cv2.COLOR_RGB2BGR)
        preprocessed_images.append(('brightened', brightened_cv, original_size))
        
        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_image)
        contrasted = enhancer.enhance(enhancement_factor)
        contrasted_cv = cv2.cvtColor(np.array(contrasted), cv2.COLOR_RGB2BGR)
        preprocessed_images.append(('contrasted', contrasted_cv, original_size))
        
        # Combined brightness and contrast
        enhancer = ImageEnhance.Brightness(pil_image)
        brightened = enhancer.enhance(enhancement_factor)
        enhancer = ImageEnhance.Contrast(brightened)
        combined = enhancer.enhance(enhancement_factor)
        combined_cv = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
        preprocessed_images.append(('combined', combined_cv, original_size))
        
        # Gamma correction for darker images
        gamma = 0.7  # Brighten dark areas
        pil_array = np.array(pil_image)
        gamma_corrected = np.power(pil_array / 255.0, gamma) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)
        gamma_cv = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2BGR)
        preprocessed_images.append(('gamma_corrected', gamma_cv, original_size))
        
        # Histogram equalization
        img_cv = cv2.imread(image_path)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        hist_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        preprocessed_images.append(('histogram_equalized', hist_eq, original_size))
        
        return preprocessed_images
        
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return [('original', cv2.imread(image_path), (0, 0))]

def run_nudenet_detection_enhanced(image_path, confidence_threshold=0.1, enhancement_factor=1.5):
    """
    Run NudeNet detection with multiple preprocessing techniques for better detection on darker bodies and small parts.
    
    Args:
        image_path (str): Path to input image
        confidence_threshold (float): Minimum confidence for detections
        enhancement_factor (float): Brightness/contrast enhancement factor
        
    Returns:
        List of detection dictionaries with 'box', 'score', 'class' keys
    """
    try:
        # Initialize NudeNet detector
        detector = NudeDetector()
        
        # Get preprocessed images (now includes upscaled versions)
        preprocessed_images = preprocess_image_for_small_parts(image_path, enhancement_factor)
        
        all_detections = []
        
        print(f"Running detection on {len(preprocessed_images)} preprocessed versions...")
        
        for i, (preprocess_type, img, img_size) in enumerate(preprocessed_images):
            print(f"  Processing {preprocess_type} version (size: {img_size})...")
            
            # Save temporary image for NudeNet
            temp_path = f"temp_{preprocess_type}.jpg"
            cv2.imwrite(temp_path, img)
            
            try:
                # Run detection on this preprocessed version
                detections = detector.detect(temp_path)
                
                # Filter by confidence threshold
                filtered_detections = []
                for detection in detections:
                    if detection['score'] >= confidence_threshold:
                        # Add preprocessing type info
                        detection['preprocess_type'] = preprocess_type
                        detection['image_size'] = img_size
                        
                        # Scale bounding boxes back to original image size if upscaled
                        if 'upscaled' in preprocess_type:
                            detection = scale_detection_to_original(detection, img_size, preprocessed_images[0][2])
                        
                        filtered_detections.append(detection)
                
                all_detections.extend(filtered_detections)
                print(f"    Found {len(filtered_detections)} detections")
                
            except Exception as e:
                print(f"    Error detecting on {preprocess_type}: {e}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Remove duplicate detections (same class and similar location)
        unique_detections = remove_duplicate_detections(all_detections)
        
        print(f"Total unique detections after deduplication: {len(unique_detections)}")
        
        return unique_detections
        
    except Exception as e:
        print(f"Error in enhanced NudeNet detection: {str(e)}")
        return []

def scale_detection_to_original(detection, current_size, original_size):
    """
    Scale detection bounding box from upscaled image back to original image size.
    
    Args:
        detection (dict): Detection dictionary with 'box' key
        current_size (tuple): Current image size (width, height)
        original_size (tuple): Original image size (width, height)
        
    Returns:
        dict: Detection with scaled bounding box
    """
    try:
        current_width, current_height = current_size
        original_width, original_height = original_size
        
        # Calculate scale factors
        scale_x = original_width / current_width
        scale_y = original_height / current_height
        
        # Scale the bounding box
        x, y, w, h = detection['box']
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        scaled_w = w * scale_x
        scaled_h = h * scale_y
        
        # Create new detection with scaled box
        scaled_detection = detection.copy()
        scaled_detection['box'] = [scaled_x, scaled_y, scaled_w, scaled_h]
        
        return scaled_detection
        
    except Exception as e:
        print(f"Error scaling detection: {e}")
        return detection

def remove_duplicate_detections(detections, iou_threshold=0.5):
    """
    Remove duplicate detections based on IoU (Intersection over Union).
    
    Args:
        detections (list): List of detection dictionaries
        iou_threshold (float): IoU threshold for considering detections as duplicates
        
    Returns:
        List of unique detections
    """
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    unique_detections = []
    
    for detection in sorted_detections:
        is_duplicate = False
        
        for unique_detection in unique_detections:
            # Check if same class
            if detection['class'] == unique_detection['class']:
                # Calculate IoU
                iou = calculate_iou(detection['box'], unique_detection['box'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_detections.append(detection)
    
    return unique_detections

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1, box2: [x, y, w, h] format
        
    Returns:
        float: IoU value
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1, y1, x2, y2] format
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2
    
    # Calculate intersection
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def run_nudenet_detection(image_path, confidence_threshold=0.1):
    """
    Run NudeNet detection on an image and return detected rectangles.
    
    Args:
        image_path (str): Path to input image
        confidence_threshold (float): Minimum confidence for detections
        
    Returns:
        List of detection dictionaries with 'box', 'score', 'class' keys
    """
    try:
        # Initialize NudeNet detector
        detector = NudeDetector()
        
        # Run detection
        detections = detector.detect(image_path)
        
        # Filter by confidence threshold
        filtered_detections = []
        for detection in detections:
            if detection['score'] >= confidence_threshold:
                filtered_detections.append(detection)
        
        return filtered_detections
        
    except Exception as e:
        print(f"Error in NudeNet detection: {str(e)}")
        return []

def draw_nudenet_rectangles(image_path, detections, output_path, title="NudeNet Detections"):
    """
    Draw NudeNet detection rectangles on an image and save it.
    
    Args:
        image_path (str): Path to input image
        detections (list): List of detection dictionaries from NudeNet
        output_path (str): Path to save the output image
        title (str): Title for the image
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return
        
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
            'FEET_EXPOSED': (255, 20, 147),            # Deep Pink
            'FEET_COVERED': (255, 105, 180),           # Hot Pink
            'ARMPITS_EXPOSED': (255, 215, 0),          # Gold
            'ARMPITS_COVERED': (218, 165, 32),         # Goldenrod
        }
        
        # Sort detections by confidence (highest first) to prioritize important ones
        sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        # Track used label positions to avoid overlap
        used_label_positions = []
        
        # Draw each detection rectangle
        for i, detection in enumerate(sorted_detections):
            box = detection['box']  # [x, y, w, h] format
            score = detection['score']
            class_name = detection['class']
            
            # Convert [x, y, w, h] to [x1, y1, x2, y2] for drawing
            x, y, w, h = box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # Get color for this class
            color = color_map.get(class_name, (255, 255, 255))  # White default
            
            # Draw rectangle with different thickness based on confidence
            thickness = max(1, int(score * 5))  # Higher confidence = thicker line
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Create label
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Find a good position for the label to avoid overlap
            label_positions = [
                (x1, y1 - 10),                    # Above top-left
                (x2 - label_size[0], y1 - 10),    # Above top-right
                (x1, y2 + label_size[1] + 10),    # Below bottom-left
                (x2 - label_size[0], y2 + label_size[1] + 10),  # Below bottom-right
                (x1, y1 + h//2),                  # Middle-left
                (x2 - label_size[0], y1 + h//2),  # Middle-right
            ]
            
            # Find the best position that doesn't overlap with existing labels
            best_position = None
            for pos in label_positions:
                pos_rect = (pos[0], pos[1], pos[0] + label_size[0], pos[1] + label_size[1])
                overlap = False
                for used_pos in used_label_positions:
                    if (pos_rect[0] < used_pos[2] and pos_rect[2] > used_pos[0] and
                        pos_rect[1] < used_pos[3] and pos_rect[3] > used_pos[1]):
                        overlap = True
                        break
                if not overlap:
                    best_position = pos
                    used_label_positions.append(pos_rect)
                    break
            
            # If no good position found, use the first one and add offset
            if best_position is None:
                best_position = (x1, y1 - 10 - i * 20)  # Stack labels vertically
                used_label_positions.append((best_position[0], best_position[1], 
                                           best_position[0] + label_size[0], 
                                           best_position[1] + label_size[1]))
            
            # Draw label background with semi-transparency effect
            overlay = img.copy()
            cv2.rectangle(overlay, 
                         (best_position[0], best_position[1] - label_size[1] - 5),
                         (best_position[0] + label_size[0], best_position[1] + 5),
                         color, -1)
            # Blend with original image for transparency effect
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            # Draw label text
            cv2.putText(img, label, best_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Add detection number for reference
            cv2.putText(img, f"#{i+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add title
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add detection count
        cv2.putText(img, f"Total Detections: {len(detections)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save the image
        cv2.imwrite(output_path, img)
        print(f"NudeNet detection image saved: {output_path}")
        print(f"Found {len(detections)} detections")
        
        # Print detection details
        print("\nDetection Details (sorted by confidence):")
        for i, detection in enumerate(sorted_detections, 1):
            print(f"  #{i}. {detection['class']}: {detection['score']:.3f}")
        
    except Exception as e:
        print(f"Error drawing detections: {str(e)}")

def preprocess_image_for_dark_bodies(image_path, enhancement_factor=1.5):
    """
    Preprocess image to improve detection on darker bodies.
    
    Args:
        image_path (str): Path to input image
        enhancement_factor (float): Brightness enhancement factor
        
    Returns:
        List of preprocessed images with different enhancements
    """
    try:
        # Load image with PIL for better enhancement control
        pil_image = Image.open(image_path)
        
        preprocessed_images = []
        
        # Original image
        preprocessed_images.append(('original', cv2.imread(image_path)))
        
        # Brightness enhancement
        enhancer = ImageEnhance.Brightness(pil_image)
        brightened = enhancer.enhance(enhancement_factor)
        brightened_cv = cv2.cvtColor(np.array(brightened), cv2.COLOR_RGB2BGR)
        preprocessed_images.append(('brightened', brightened_cv))
        
        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_image)
        contrasted = enhancer.enhance(enhancement_factor)
        contrasted_cv = cv2.cvtColor(np.array(contrasted), cv2.COLOR_RGB2BGR)
        preprocessed_images.append(('contrasted', contrasted_cv))
        
        # Combined brightness and contrast
        enhancer = ImageEnhance.Brightness(pil_image)
        brightened = enhancer.enhance(enhancement_factor)
        enhancer = ImageEnhance.Contrast(brightened)
        combined = enhancer.enhance(enhancement_factor)
        combined_cv = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
        preprocessed_images.append(('combined', combined_cv))
        
        # Gamma correction for darker images
        gamma = 0.7  # Brighten dark areas
        pil_array = np.array(pil_image)
        gamma_corrected = np.power(pil_array / 255.0, gamma) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)
        gamma_cv = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2BGR)
        preprocessed_images.append(('gamma_corrected', gamma_cv))
        
        # Histogram equalization
        img_cv = cv2.imread(image_path)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        hist_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        preprocessed_images.append(('histogram_equalized', hist_eq))
        
        return preprocessed_images
        
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return [('original', cv2.imread(image_path))]

def main():
    # Configuration
    input_path = "data/brazzers_2.png"  # Change this to your image path
    output_path = "nudenet_detections.jpg"
    confidence_threshold = 0.05  # Adjust this threshold as needed
    use_enhanced_detection = True  # Set to True for better detection on darker bodies and small parts
    enhancement_factor = 1.5  # Brightness/contrast enhancement factor
    
    print("=== NudeNet Detection Script (Enhanced for Dark Bodies & Small Parts) ===")
    print(f"Input image: {input_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Enhanced detection: {use_enhanced_detection}")
    if use_enhanced_detection:
        print(f"Enhancement factor: {enhancement_factor}")
        print("Processing includes:")
        print("  - Multiple upscaling factors (1.5x, 2.0x, 2.5x, 3.0x)")
        print("  - Enhanced versions of upscaled images")
        print("  - Brightness/contrast adjustments")
        print("  - Gamma correction")
        print("  - Histogram equalization")
        print("  - Automatic bounding box scaling")
    print()
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        print("Please update the input_path variable to point to your image file.")
        return
    
    try:
        # Run NudeNet detection
        print("Running NudeNet detection...")
        if use_enhanced_detection:
            detections = run_nudenet_detection_enhanced(input_path, confidence_threshold, enhancement_factor)
        else:
            detections = run_nudenet_detection(input_path, confidence_threshold)
        
        if detections:
            # Draw detection rectangles
            print("Drawing detection rectangles...")
            draw_nudenet_rectangles(input_path, detections, output_path, "NudeNet All Parts Detection")
            
            print(f"\n=== Detection Complete ===")
            print(f"Output saved: {output_path}")
            
            # Show preprocessing info if enhanced detection was used
            if use_enhanced_detection:
                preprocess_counts = {}
                upscaled_detections = 0
                for detection in detections:
                    preprocess_type = detection.get('preprocess_type', 'unknown')
                    preprocess_counts[preprocess_type] = preprocess_counts.get(preprocess_type, 0) + 1
                    if 'upscaled' in preprocess_type:
                        upscaled_detections += 1
                
                print("\nDetections by preprocessing method:")
                for preprocess_type, count in preprocess_counts.items():
                    print(f"  {preprocess_type}: {count} detections")
                
                if upscaled_detections > 0:
                    print(f"\nSmall part detection results:")
                    print(f"  Detections from upscaled images: {upscaled_detections}")
                    print(f"  This indicates small body parts were found by enlarging the image")
            
            # Show color legend
            print("\nColor Legend:")
            color_legend = {
                'FEMALE_BREAST_EXPOSED': 'Red',
                'FEMALE_BREAST_COVERED': 'Orange', 
                'FEMALE_GENITALIA_EXPOSED': 'Blue',
                'FEMALE_GENITALIA_COVERED': 'Magenta',
                'MALE_GENITALIA_EXPOSED': 'Green',
                'MALE_GENITALIA_COVERED': 'Yellow',
                'BUTTOCKS_EXPOSED': 'Purple',
                'BUTTOCKS_COVERED': 'Pink',
                'ANUS_EXPOSED': 'Brown',
                'ANUS_COVERED': 'Olive',
                'FEET_EXPOSED': 'Deep Pink',
                'FEET_COVERED': 'Hot Pink',
                'ARMPITS_EXPOSED': 'Gold',
                'ARMPITS_COVERED': 'Goldenrod'
            }
            
            for part, color in color_legend.items():
                print(f"  {part}: {color}")
                
        else:
            print("No detections found with the current confidence threshold.")
            print("Try lowering the confidence_threshold if you expect detections.")
            if not use_enhanced_detection:
                print("Consider enabling enhanced detection for better results on darker bodies and small parts.")
    
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
