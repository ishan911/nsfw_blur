#!/usr/bin/env python3
"""
NudeNet Detector Class - Generic wrapper for NudeNet detection and pixelation

This class provides a clean interface for detecting sensitive content in images
and applying pixelation for privacy protection.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from nudenet import NudeDetector

class NudeNetDetector:
    """
    A generic class for NudeNet detection and pixelation functionality.
    
    This class provides methods to:
    - Detect sensitive content in images
    - Apply pixelation to detected regions
    - Process images with sliding windows
    - Generate various output formats
    """
    
    def __init__(self, confidence_threshold=0.05, pixel_size=15, padding=5, disable_label_filter=False):
        """
        Initialize the NudeNet detector.
        
        Args:
            confidence_threshold (float): Minimum confidence for detections (0.01-1.0)
            pixel_size (int): Size of pixel blocks for pixelation (5-25 recommended)
            padding (int): Extra padding around detected regions in pixels
            disable_label_filter (bool): If True, process all detected labels instead of filtering
        """
        self.confidence_threshold = confidence_threshold
        self.pixel_size = pixel_size
        self.padding = padding
        self.detector = NudeDetector()
        
        # Color mapping for different body parts
        self.color_map = {
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
        
        # Allowed labels for filtering
        if disable_label_filter:
            # If label filtering is disabled, include all possible labels
            self.allowed_labels = set([
                "BUTTOCKS_EXPOSED",
                "BUTTOCKS_COVERED",
                "FEMALE_BREAST_EXPOSED",
                "FEMALE_BREAST_COVERED",
                "FEMALE_GENITALIA_EXPOSED",
                "FEMALE_GENITALIA_COVERED",
                "MALE_GENITALIA_EXPOSED",
                "MALE_GENITALIA_COVERED",
                "ANUS_COVERED",
                "ANUS_EXPOSED",
                "FEET_EXPOSED",
                "FEET_COVERED",
                "ARMPITS_EXPOSED",
                "ARMPITS_COVERED",
            ])
        else:
            # Default filtered set
            self.allowed_labels = set([
                "BUTTOCKS_EXPOSED",
                "BUTTOCKS_COVERED",
                "FEMALE_BREAST_EXPOSED",
                "FEMALE_GENITALIA_EXPOSED",
                "FEMALE_GENITALIA_COVERED",
                "ANUS_COVERED",
                "ANUS_EXPOSED",
                "MALE_GENITALIA_EXPOSED",
            ])
    
    def preprocess_image_for_small_parts(self, image_path, enhancement_factor=1.5):
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
            
            # Add the original enhanced versions
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
    
    def scale_detection_to_original(self, detection, current_size, original_size):
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
    
    def remove_duplicate_detections(self, detections, iou_threshold=0.5):
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
                    iou = self.calculate_iou(detection['box'], unique_detection['box'])
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_detections.append(detection)
        
        return unique_detections
    
    def calculate_iou(self, box1, box2):
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
    
    def pixelate_region(self, img, x1, y1, x2, y2, pixel_size=10):
        """
        Pixelate a region of an image with fixed pixel size.
        
        Args:
            img (np.ndarray): Input image
            x1, y1, x2, y2 (int): Bounding box coordinates
            pixel_size (int): Fixed size of each pixel block (higher = more pixelated)
            
        Returns:
            np.ndarray: Image with pixelated region
        """
        try:
            # Extract the region to pixelate
            region = img[y1:y2, x1:x2]
            
            if region.size == 0:
                return img
            
            # Get dimensions of the region
            h, w = region.shape[:2]
            
            # Use fixed pixel size - don't adapt based on region size
            # Calculate new dimensions for pixelation
            new_h = h // pixel_size
            new_w = w // pixel_size
            
            # If region is too small for the pixel size, use the smallest possible pixel size
            if new_h == 0 or new_w == 0:
                # Use minimum pixel size of 2, but don't change the original pixel_size
                min_pixel_size = max(2, min(h, w) // 2)
                new_h = h // min_pixel_size
                new_w = w // min_pixel_size
                print(f"    Warning: Region too small for pixel_size {pixel_size}, using {min_pixel_size} for this region")
            else:
                min_pixel_size = pixel_size
            
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
            print(f"Error pixelating region: {e}")
            return img
    
    def detect_enhanced(self, image_path, enhancement_factor=1.5):
        """
        Run enhanced NudeNet detection with multiple preprocessing techniques.
        
        Args:
            image_path (str): Path to input image
            enhancement_factor (float): Brightness/contrast enhancement factor
            
        Returns:
            List of detection dictionaries with 'box', 'score', 'class' keys
        """
        try:
            # Get preprocessed images
            preprocessed_images = self.preprocess_image_for_small_parts(image_path, enhancement_factor)
            
            all_detections = []
            
            # print(f"Running detection on {len(preprocessed_images)} preprocessed versions...")
            
            for i, (preprocess_type, img, img_size) in enumerate(preprocessed_images):
                # print(f"  Processing {preprocess_type} version (size: {img_size})...")
                
                # Save temporary image for NudeNet
                temp_path = f"temp_{preprocess_type}.jpg"
                cv2.imwrite(temp_path, img)
                
                try:
                    # Run detection on this preprocessed version
                    detections = self.detector.detect(temp_path)
                    
                    # Filter by confidence threshold
                    filtered_detections = []
                    for detection in detections:
                        if(detection['class'] in self.allowed_labels):
                            if detection['score'] >= self.confidence_threshold:
                                # Add preprocessing type info
                                detection['preprocess_type'] = preprocess_type
                                detection['image_size'] = img_size
                                
                                # Scale bounding boxes back to original image size if upscaled
                                if 'upscaled' in preprocess_type:
                                    detection = self.scale_detection_to_original(detection, img_size, preprocessed_images[0][2])
                                
                                filtered_detections.append(detection)
                    
                    all_detections.extend(filtered_detections)
                    # print(f"    Found {len(filtered_detections)} detections")
                    
                except Exception as e:
                    print(f"    Error detecting on {preprocess_type}: {e}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Remove duplicate detections
            unique_detections = self.remove_duplicate_detections(all_detections)
            
            # print(f"Total unique detections after deduplication: {len(unique_detections)}")
            
            return unique_detections
            
        except Exception as e:
            print(f"Error in enhanced NudeNet detection: {str(e)}")
            return []
    
    def detect_simple(self, image_path):
        """
        Run simple NudeNet detection on an image.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            List of detection dictionaries with 'box', 'score', 'class' keys
        """
        try:
            # Run detection
            detections = self.detector.detect(image_path)
            
            # Filter by confidence threshold
            filtered_detections = []
            for detection in detections:
                if detection['score'] >= self.confidence_threshold:
                    filtered_detections.append(detection)
            
            return filtered_detections
            
        except Exception as e:
            print(f"Error in NudeNet detection: {str(e)}")
            return []
    
    def sliding_window(self, image, step_size, window_size):
        """
        Slide a window across the image.
        
        Args:
            image (np.ndarray): The input image
            step_size (int): Step size for the sliding window
            window_size (tuple): (width, height) of the window
            
        Yields:
            (x, y, window): The top-left x, y and the window image
        """
        for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
            for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
    
    def detect_with_sliding_window(self, input_path, window_size=(256, 256), step_size=128, enhancement_factor=1.5):
        """
        Process the image with a sliding window for comprehensive detection.
        
        Args:
            input_path (str): Path to input image
            window_size (tuple): Size of sliding window (width, height)
            step_size (int): Step size for sliding window
            enhancement_factor (float): Enhancement factor for preprocessing
            
        Returns:
            List of detection dictionaries
        """
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Could not load image {input_path}")
            return []
        
        h, w = img.shape[:2]
        all_window_detections = []
        window_count = 0
        total_detections = 0
        
        print(f"Processing image with sliding window (size: {window_size}, step: {step_size})")
        print(f"Image size: {w}x{h}")
        
        for x, y, window in self.sliding_window(img, step_size, window_size):
            # Save temp window for processing
            temp_window_path = f"temp_window_{y}_{x}.jpg"
            cv2.imwrite(temp_window_path, window)
            
            # Run enhanced detection on the window
            detections = self.detect_enhanced(temp_window_path, enhancement_factor)
            
            # Filter for allowed labels
            filtered = [d for d in detections if d['class'] in self.allowed_labels]
            
            # Transform coordinates back to original image space
            for detection in filtered:
                # Get the bounding box coordinates relative to the window
                wx, wy, ww, wh = detection['box']
                
                # Transform to original image coordinates
                original_x = x + wx
                original_y = y + wy
                
                # Create new detection with original image coordinates
                transformed_detection = {
                    'box': [original_x, original_y, ww, wh],
                    'score': detection['score'],
                    'class': detection['class'],
                    'window_position': (x, y),
                    'preprocess_type': detection.get('preprocess_type', 'unknown')
                }
                
                all_window_detections.append(transformed_detection)
            
            total_detections += len(filtered)
            window_count += 1
            
            # Clean up temporary file
            os.remove(temp_window_path)
            
            # if len(filtered) > 0:
            #     print(f"  Window ({x},{y}): {len(filtered)} detections")
        
        print(f"\nSliding window processing complete:")
        print(f"  Windows processed: {window_count}")
        print(f"  Total detections found: {total_detections}")
        print(f"  Detections after coordinate transformation: {len(all_window_detections)}")
        
        # Remove duplicate detections from overlapping windows
        if all_window_detections:
            unique_detections = self.remove_duplicate_detections(all_window_detections, iou_threshold=0.3)
            # print(f"  Unique detections after deduplication: {len(unique_detections)}")
            return unique_detections
        else:
            print("  No detections found in any windows")
            return []
    
    def pixelate_image(self, image_path, detections, output_path, draw_rectangles=False, draw_labels=False):
        """
        Pixelate detected regions in an image.
        
        Args:
            image_path (str): Path to input image
            detections (list): List of detection dictionaries
            output_path (str): Path to save the output image
            draw_rectangles (bool): Whether to draw rectangle borders
            draw_labels (bool): Whether to draw labels (only if draw_rectangles=True)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return False
            
            height, width = img.shape[:2]
            
            # Sort detections by confidence (highest first)
            sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
            
            # Track used label positions to avoid overlap
            used_label_positions = []
            valid_detections = 0
            
            # Process each detection
            for i, detection in enumerate(sorted_detections):
                try:
                    box = detection['box']  # [x, y, w, h] format
                    score = detection['score']
                    class_name = detection['class']
                    
                    # Validate bounding box coordinates
                    if len(box) != 4:
                        print(f"Warning: Invalid box format for detection {i}: {box}")
                        continue
                    
                    x, y, w, h = box
                    
                    # Convert to integers and ensure they're valid
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # Check if coordinates are within image bounds
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        print(f"Warning: Invalid coordinates for detection {i}: x={x}, y={y}, w={w}, h={h}")
                        continue
                    
                    # Ensure rectangle doesn't go outside image bounds
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(width, x + w), min(height, y + h)
                    
                    # Skip if rectangle is too small or invalid
                    if x2 <= x1 or y2 <= y1:
                        print(f"Warning: Invalid rectangle for detection {i}: ({x1},{y1}) to ({x2},{y2})")
                        continue
                    
                    # Add padding to the detected rectangle
                    x1_padded = max(0, x1 - self.padding)
                    y1_padded = max(0, y1 - self.padding)
                    x2_padded = min(width, x2 + self.padding)
                    y2_padded = min(height, y2 + self.padding)
                    
                    # Pixelate the detected region (using padded coordinates)
                    img = self.pixelate_region(img, x1_padded, y1_padded, x2_padded, y2_padded, self.pixel_size)
                    print(f"  Pixelated detection #{i+1} ({class_name}) with pixel size {self.pixel_size} and {self.padding}px padding")
                    
                    # Draw rectangle border and labels if requested
                    if draw_rectangles:
                        # Get color for this class
                        color = self.color_map.get(class_name, (255, 255, 255))  # White default
                        
                        # Draw rectangle with different thickness based on confidence
                        thickness = max(1, int(score * 5))  # Higher confidence = thicker line
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw labels if requested
                        if draw_labels:
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
                    
                    valid_detections += 1
                    
                except Exception as e:
                    print(f"Error processing detection {i}: {e}")
                    continue
            
            # Save the image
            cv2.imwrite(output_path, img)
            print(f"Pixelated image saved: {output_path}")
            print(f"Successfully processed {valid_detections} out of {len(detections)} detections")
            
            return True
            
        except Exception as e:
            print(f"Error pixelating image: {str(e)}")
            return False
    
    def process_image(self, input_path, output_path, use_sliding_window=True, draw_rectangles=False, draw_labels=False):
        """
        Process an image with detection and pixelation.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save the output image
            use_sliding_window (bool): Whether to use sliding window detection
            draw_rectangles (bool): Whether to draw rectangle borders
            draw_labels (bool): Whether to draw labels (only if draw_rectangles=True)
            
        Returns:
            dict: Processing results with detection count and success status
        """
        try:
            print(f"Processing image: {input_path}")
            
            # Run detection with enhanced approach
            if use_sliding_window:
                # Use the new enhanced detection that processes full image first
                detections = self.detect_with_full_image_first(input_path)
            else:
                # Use simple enhanced detection for smaller images
                detections = self.detect_enhanced(input_path)
            
            if not detections:
                print("No detections found.")
                return {
                    'success': False,
                    'detection_count': 0,
                    'message': 'No detections found'
                }
            
            # Pixelate the image
            success = self.pixelate_image(input_path, detections, output_path, draw_rectangles, draw_labels)
            
            return {
                'success': success,
                'detection_count': len(detections),
                'detections': detections,
                'message': f'Processed {len(detections)} detections'
            }
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return {
                'success': False,
                'detection_count': 0,
                'message': f'Error: {str(e)}'
            }
    
    def detect_with_full_image_first(self, input_path, window_size=(256, 256), step_size=128, enhancement_factor=1.5):
        """
        Enhanced detection that first processes the full image for large body parts,
        then uses sliding window for smaller details.
        
        Args:
            input_path (str): Path to input image
            window_size (tuple): Size of sliding window (width, height)
            step_size (int): Step size for sliding window
            enhancement_factor (float): Enhancement factor for preprocessing
            
        Returns:
            List of detection dictionaries
        """
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Could not load image {input_path}")
            return []
        
        h, w = img.shape[:2]
        print(f"Processing image: {w}x{h}")
        
        all_detections = []
        
        # Step 1: Process full image for large body parts
        print("Step 1: Processing full image for large body parts...")
        full_image_detections = self.detect_enhanced(input_path, enhancement_factor)
        
        # Filter for allowed labels
        full_image_filtered = [d for d in full_image_detections if d['class'] in self.allowed_labels]
        
        print(f"  Full image detections: {len(full_image_filtered)}")
        for detection in full_image_filtered:
            detection['source'] = 'full_image'
            all_detections.append(detection)
        
        # Step 2: Determine if sliding window is needed based on image size and existing detections
        # Use sliding window if image is large (> 800x600) or if no large detections found
        image_area = w * h
        large_image_threshold = 800 * 600
        
        # Check if we have large detections (covering significant area)
        large_detections_found = False
        for detection in full_image_filtered:
            x, y, w_det, h_det = detection['box']
            detection_area = w_det * h_det
            area_ratio = detection_area / image_area
            
            # Consider it large if it covers more than 5% of the image
            if area_ratio > 0.05:
                large_detections_found = True
                break
        
        # Step 3: Use sliding window for smaller details if needed
        if image_area > large_image_threshold or not large_detections_found:
            print("Step 2: Using sliding window for smaller details...")
            
            # Adjust window size based on image size
            if image_area > 2000 * 1500:  # Very large images
                window_size = (512, 512)
                step_size = 256
            elif image_area > 1500 * 1000:  # Large images
                window_size = (384, 384)
                step_size = 192
            else:  # Medium images
                window_size = (256, 256)
                step_size = 128
            
            print(f"  Adjusted window size: {window_size}, step size: {step_size}")
            
            window_detections = self.detect_with_sliding_window(
                input_path, 
                window_size=window_size, 
                step_size=step_size, 
                enhancement_factor=enhancement_factor
            )
            
            # Mark sliding window detections
            for detection in window_detections:
                detection['source'] = 'sliding_window'
            
            all_detections.extend(window_detections)
            print(f"  Sliding window detections: {len(window_detections)}")
        else:
            print("Step 2: Skipping sliding window (large detections found)")
        
        # Step 4: Remove duplicates between full image and sliding window detections
        if len(all_detections) > 1:
            print("Step 3: Removing duplicate detections...")
            unique_detections = self.remove_duplicate_detections(all_detections, iou_threshold=0.4)
            print(f"  Unique detections after deduplication: {len(unique_detections)}")
            
            # Count by source
            full_image_count = len([d for d in unique_detections if d.get('source') == 'full_image'])
            sliding_window_count = len([d for d in unique_detections if d.get('source') == 'sliding_window'])
            
            print(f"  Final breakdown:")
            print(f"    Full image detections: {full_image_count}")
            print(f"    Sliding window detections: {sliding_window_count}")
            
            return unique_detections
        else:
            print(f"Step 3: No duplicates to remove (total: {len(all_detections)})")
            return all_detections 