#!/usr/bin/env python3
"""
Two-Stage Sliding Window Detection and Blurring Script
This script implements a comprehensive two-stage detection system:
1. NudeNet detection and blurring (first stage)
2. YOLO sliding window detection and blurring (second stage)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageFilter
from nudenet import NudeDetector
import os
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
import argparse
import sys


class TwoStageSlidingWindowDetector:
    def __init__(self, nudenet_model_path='models/classifier_model.onnx', yolo_model_path='models/80_epoch.pt',
                 window_size=640, stride=320, overlap_threshold=0.3,
                 nudenet_enabled=True, yolo_enabled=True,
                 nudenet_confidence_threshold=0.5, yolo_confidence_threshold=0.5,
                 blur_method='pixelate', pixel_size=10):
        """
        Initialize the two-stage sliding window detector.
        
        Args:
            nudenet_model_path (str): Path to NudeNet model
            yolo_model_path (str): Path to YOLO model
            window_size (int): Size of sliding windows
            stride (int): Stride between windows
            overlap_threshold (float): IoU threshold for merging detections
            nudenet_enabled (bool): Enable NudeNet detection
            yolo_enabled (bool): Enable YOLO detection
            nudenet_confidence_threshold (float): NudeNet confidence threshold
            yolo_confidence_threshold (float): YOLO confidence threshold
            blur_method (str): 'pixelate' or 'blur'
            pixel_size (int): Pixel size for pixelation
        """
        self.nudenet_model_path = nudenet_model_path
        self.yolo_model_path = yolo_model_path
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        self.nudenet_enabled = nudenet_enabled
        self.yolo_enabled = yolo_enabled
        self.nudenet_confidence_threshold = nudenet_confidence_threshold
        self.yolo_confidence_threshold = yolo_confidence_threshold
        self.blur_method = blur_method
        self.pixel_size = pixel_size
        
        # Initialize NudeNet detector
        if self.nudenet_enabled:
            try:
                from nudenet import NudeDetector
                self.nudenet_detector = NudeDetector(self.nudenet_model_path)
            except ImportError:
                print("Warning: nudenet not available. NudeNet detection will be skipped.")
                self.nudenet_enabled = False
        
        # Initialize YOLO model
        if self.yolo_enabled:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.yolo_model_path)
            except ImportError:
                print("Warning: ultralytics not available. YOLO detection will be skipped.")
                self.yolo_enabled = False
        
        # NudeNet body parts to detect
        self.nudenet_parts = [
            'BUTTOCKS_EXPOSED',
            'BUTTOCKS_COVERED',
            'FEMALE_BREAST_EXPOSED',
            'FEMALE_GENITALIA_EXPOSED',
            'FEMALE_BREAST_COVERED',
            'ANUS_EXPOSED',
            'MALE_GENITALIA_EXPOSED',
        ]
        
    def create_sliding_windows(self, image_width: int, image_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Create sliding windows for the given image dimensions.
        
        Args:
            image_width (int): Width of the input image
            image_height (int): Height of the input image
            
        Returns:
            List of (x, y, width, height) tuples for each window
        """
        windows = []
        
        for y in range(0, image_height, self.stride):
            for x in range(0, image_width, self.stride):
                # Calculate window boundaries
                window_x = x
                window_y = y
                window_w = min(self.window_size, image_width - x)
                window_h = min(self.window_size, image_height - y)
                
                # Only add windows that are large enough
                if window_w >= self.window_size // 2 and window_h >= self.window_size // 2:
                    windows.append((window_x, window_y, window_w, window_h))
                    
        return windows
    
    def crop_window(self, image: np.ndarray, window: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop a window from the image.
        
        Args:
            image (np.ndarray): Input image
            window (tuple): (x, y, width, height) of the window
            
        Returns:
            Cropped image window
        """
        x, y, w, h = window
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    def translate_detection_coordinates(self, detection: Dict[str, Any], window: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Translate detection coordinates from window coordinates to original image coordinates.
        
        Args:
            detection (Dict): Detection result from YOLO
            window (tuple): (x, y, width, height) of the window
            
        Returns:
            Detection with translated coordinates
        """
        window_x, window_y, window_w, window_h = window
        
        # Get the bounding box coordinates
        if hasattr(detection, 'boxes') and detection.boxes is not None:
            boxes = detection.boxes.xyxy.cpu().numpy()
            translated_boxes = []
            
            for box in boxes:
                x1, y1, x2, y2 = box
                # Translate coordinates
                translated_x1 = x1 + window_x
                translated_y1 = y1 + window_y
                translated_x2 = x2 + window_x
                translated_y2 = y2 + window_y
                translated_boxes.append([translated_x1, translated_y1, translated_x2, translated_y2])
            
            # Create a new detection object with translated coordinates
            # Instead of trying to create a new Results object, we'll create a simple dict
            # that contains the translated boxes and other necessary information
            translated_detection = {
                'boxes': {
                    'xyxy': np.array(translated_boxes),
                    'conf': detection.boxes.conf,
                    'cls': detection.boxes.cls
                }
            }
            
            return translated_detection
        
        return detection
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1 (List[float]): [x1, y1, x2, y2] of first box
            box2 (List[float]): [x1, y1, x2, y2] of second box
            
        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def merge_overlapping_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping detections based on IoU threshold.
        
        Args:
            detections (List[Dict]): List of detection results
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
        
        # Extract all bounding boxes and their metadata
        all_boxes = []
        for detection in detections:
            # Handle both YOLO Results objects and our custom dict format
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                # YOLO Results object
                boxes = detection.boxes.xyxy.cpu().numpy()
                confs = detection.boxes.conf.cpu().numpy()
                classes = detection.boxes.cls.cpu().numpy()
            elif isinstance(detection, dict) and 'boxes' in detection:
                # Our custom dict format
                boxes = detection['boxes']['xyxy']
                confs = detection['boxes']['conf'].cpu().numpy()
                classes = detection['boxes']['cls'].cpu().numpy()
            else:
                continue
                
            for i, box in enumerate(boxes):
                all_boxes.append({
                    'box': box.tolist(),
                    'conf': confs[i],
                    'cls': classes[i],
                    'detection': detection
                })
        
        if not all_boxes:
            return []
        
        # Sort by confidence (highest first)
        all_boxes.sort(key=lambda x: x['conf'], reverse=True)
        
        merged_boxes = []
        used_indices = set()
        
        for i, box_info in enumerate(all_boxes):
            if i in used_indices:
                continue
            
            current_box = box_info['box']
            current_conf = box_info['conf']
            current_cls = box_info['cls']
            
            # Find overlapping boxes
            overlapping_indices = []
            for j, other_box_info in enumerate(all_boxes):
                if j <= i or j in used_indices:
                    continue
                
                other_box = other_box_info['box']
                iou = self.calculate_iou(current_box, other_box)
                
                if iou > self.overlap_threshold:
                    overlapping_indices.append(j)
            
            # Merge overlapping boxes
            if overlapping_indices:
                # Use the box with highest confidence
                best_box_info = box_info
                best_conf = current_conf
                
                for idx in overlapping_indices:
                    other_box_info = all_boxes[idx]
                    if other_box_info['conf'] > best_conf:
                        best_box_info = other_box_info
                        best_conf = other_box_info['conf']
                
                merged_boxes.append(best_box_info)
                used_indices.add(i)
                used_indices.update(overlapping_indices)
            else:
                merged_boxes.append(box_info)
                used_indices.add(i)
        
        return merged_boxes
    
    def pixelate_region(self, image: np.ndarray, region: List[int], pixel_size: int = 10) -> np.ndarray:
        """
        Pixelate a specific region in the image.
        
        Args:
            image (np.ndarray): Input image
            region (List[int]): [x1, y1, x2, y2] coordinates of region to pixelate
            pixel_size (int): Size of pixels for pixelation
            
        Returns:
            Image with pixelated region
        """
        x1, y1, x2, y2 = map(int, region)
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        if x1 >= x2 or y1 >= y2:
            return image
        
        # Extract the region
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return image
        
        # Resize down and up to create pixelation effect
        small_roi = cv2.resize(roi, (max(1, roi.shape[1] // pixel_size), max(1, roi.shape[0] // pixel_size)))
        pixelated_roi = cv2.resize(small_roi, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply the pixelated region back to the image
        image[y1:y2, x1:x2] = pixelated_roi
        
        return image
    
    def blur_region(self, image: np.ndarray, region: List[int], blur_radius: int = 20) -> np.ndarray:
        """
        Blur a specific region in the image.
        
        Args:
            image (np.ndarray): Input image
            region (List[int]): [x1, y1, x2, y2] coordinates of region to blur
            blur_radius (int): Blur radius
            
        Returns:
            Image with blurred region
        """
        x1, y1, x2, y2 = map(int, region)
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        if x1 >= x2 or y1 >= y2:
            return image
        
        # Extract the region
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return image
        
        # Apply Gaussian blur
        blurred_roi = cv2.GaussianBlur(roi, (blur_radius, blur_radius), 0)
        
        # Apply the blurred region back to the image
        image[y1:y2, x1:x2] = blurred_roi
        
        return image
    
    def nudenet_full_image_stage(self, image_path):
        """
        First stage: NudeNet full image detection and blurring.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            Path to processed image
        """
        try:
            import cv2
            from PIL import Image
            import os
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return image_path
            
            height, width = img.shape[:2]
            
            # Convert to PIL Image for NudeNet
            pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Save temporary image for NudeNet
            temp_path = "temp_nudenet_full.jpg"
            pil_image.save(temp_path)
            
            # Run NudeNet detection
            nudenet_results = self.nudenet_detector.detect(temp_path)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Process NudeNet detections
            nudenet_detection_count = 0
            
            for result in nudenet_results:
                if result['score'] >= self.nudenet_confidence_threshold:
                    nudenet_detection_count += 1
                    box = result['box']  # [x, y, w, h] format
                    score = result['score']
                    class_name = result['class']
                    
                    # Convert [x, y, w, h] to [x1, y1, x2, y2] for processing
                    x, y, w, h = box
                    region = [x, y, x + w, y + h]
                    
                    if self.blur_method == 'pixelate':
                        img = self.pixelate_region(img, region, self.pixel_size)
                    else:  # blur
                        img = self.blur_region(img, region, self.pixel_size)
            
            # Save the processed image
            base_name = os.path.splitext(image_path)[0]
            nudenet_processed_path = f"{base_name}_nudenet_full_processed.jpg"
            cv2.imwrite(nudenet_processed_path, img)
            
            return nudenet_processed_path
            
        except Exception as e:
            print(f"Error in NudeNet full image stage: {str(e)}")
            return image_path
    
    def nudenet_sliding_window_stage(self, image_path):
        """
        Second stage: NudeNet sliding window detection and blurring.
        
        Args:
            image_path (str): Path to input image (already processed by NudeNet full image)
            
        Returns:
            Path to processed image
        """
        try:
            import cv2
            from PIL import Image
            import os
            import numpy as np
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return image_path
            
            height, width = img.shape[:2]
            
            # Create sliding windows
            windows = self.create_sliding_windows(width, height)
            
            # Process each window
            all_detections = []
            
            for i, window in enumerate(windows):
                window_x, window_y, window_w, window_h = window
                
                # Crop the window
                window_image = self.crop_window(img, window)
                
                # Convert to PIL Image for NudeNet
                pil_window = Image.fromarray(cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB))
                
                # Save temporary window image
                temp_window_path = f"temp_nudenet_window_{i}.jpg"
                pil_window.save(temp_window_path)
                
                # Run NudeNet detection on the window
                window_results = self.nudenet_detector.detect(temp_window_path)
                
                # Clean up temporary file
                if os.path.exists(temp_window_path):
                    os.remove(temp_window_path)
                
                # Process window detections and translate coordinates
                for result in window_results:
                    if result['score'] >= self.nudenet_confidence_threshold:
                        # Translate coordinates to original image space
                        box = result['box']  # [x, y, w, h] format
                        x, y, w, h = box
                        
                        # Translate to original image coordinates
                        translated_x = x + window_x
                        translated_y = y + window_y
                        translated_w = min(w, width - translated_x)
                        translated_h = min(h, height - translated_y)
                        
                        # Create detection info
                        detection_info = {
                            'box': [translated_x, translated_y, translated_x + translated_w, translated_y + translated_h],
                            'conf': result['score'],
                            'cls': result['class'],
                            'detection': result
                        }
                        all_detections.append(detection_info)
            
            # Merge overlapping detections
            merged_detections = self.merge_overlapping_detections(all_detections)
            
            # Apply blurring/pixelation to NudeNet detected regions
            nudenet_detection_count = 0
            
            for detection_info in merged_detections:
                nudenet_detection_count += 1
                box = detection_info['box']
                conf = detection_info['conf']
                cls = detection_info['cls']
                
                if self.blur_method == 'pixelate':
                    img = self.pixelate_region(img, box, self.pixel_size)
                else:  # blur
                    img = self.blur_region(img, box, self.pixel_size)
            
            # Save the processed image
            base_name = os.path.splitext(image_path)[0]
            nudenet_processed_path = f"{base_name}_nudenet_sliding_processed.jpg"
            cv2.imwrite(nudenet_processed_path, img)
            
            return nudenet_processed_path
            
        except Exception as e:
            print(f"Error in NudeNet sliding window stage: {str(e)}")
            return image_path
    
    def yolo_full_image_stage(self, image_path):
        """
        Third stage: YOLO full image detection and blurring.
        
        Args:
            image_path (str): Path to input image (already processed by NudeNet stages)
            
        Returns:
            Path to processed image
        """
        try:
            import cv2
            import os
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return image_path
            
            height, width = img.shape[:2]
            
            # Run detection
            results = self.yolo_model(image_path, conf=self.yolo_confidence_threshold)
            
            # Process YOLO detections
            yolo_detection_count = 0
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # Extract detection information
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        confidence = float(confs[i])
                        cls = int(classes[i])
                        
                        if self.blur_method == 'pixelate':
                            img = self.pixelate_region(img, [x1, y1, x2, y2], self.pixel_size)
                        else:  # blur
                            img = self.blur_region(img, [x1, y1, x2, y2], self.pixel_size)
                        
                        yolo_detection_count += 1
            
            # Save the processed image
            base_name = os.path.splitext(image_path)[0]
            yolo_processed_path = f"{base_name}_yolo_full_processed.jpg"
            cv2.imwrite(yolo_processed_path, img)
            
            return yolo_processed_path
            
        except Exception as e:
            print(f"Error in YOLO full image stage: {str(e)}")
            return image_path
    
    def yolo_sliding_window_stage(self, image_path):
        """
        Fourth stage: YOLO sliding window detection and blurring.
        
        Args:
            image_path (str): Path to input image (already processed by previous stages)
            
        Returns:
            Path to processed image
        """
        try:
            import cv2
            import os
            import numpy as np
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return image_path
            
            height, width = img.shape[:2]
            
            # Create sliding windows
            windows = self.create_sliding_windows(width, height)
            
            # Process each window
            all_detections = []
            
            for i, window in enumerate(windows):
                window_x, window_y, window_w, window_h = window
                
                # Crop the window
                window_image = self.crop_window(img, window)
                
                # Save temporary window image
                temp_window_path = f"temp_yolo_window_{i}.jpg"
                cv2.imwrite(temp_window_path, window_image)
                
                # Run YOLO detection on the window
                result = self.yolo_model(temp_window_path, conf=self.yolo_confidence_threshold)
                
                # Clean up temporary file
                if os.path.exists(temp_window_path):
                    os.remove(temp_window_path)
                
                # Process window detections and translate coordinates
                if result and len(result) > 0:
                    result = result[0]
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                        # Extract detection information
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        
                        for j, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
                            confidence = float(confs[j])
                            cls = int(classes[j])
                            
                            # Translate to original image coordinates
                            translated_x1 = x1 + window_x
                            translated_y1 = y1 + window_y
                            translated_x2 = min(x2 + window_x, width)
                            translated_y2 = min(y2 + window_y, height)
                            
                            # Create detection info
                            detection_info = {
                                'box': [translated_x1, translated_y1, translated_x2, translated_y2],
                                'conf': confidence,
                                'cls': cls
                            }
                            all_detections.append(detection_info)
            
            # Merge overlapping detections
            merged_detections = self.merge_overlapping_detections(all_detections)
            
            # Apply blurring/pixelation to YOLO detected regions
            yolo_detection_count = 0
            
            for detection_info in merged_detections:
                yolo_detection_count += 1
                box = detection_info['box']
                conf = detection_info['conf']
                cls = detection_info['cls']
                
                if self.blur_method == 'pixelate':
                    img = self.pixelate_region(img, box, self.pixel_size)
                else:  # blur
                    img = self.blur_region(img, box, self.pixel_size)
            
            # Save the processed image
            base_name = os.path.splitext(image_path)[0]
            yolo_processed_path = f"{base_name}_yolo_sliding_processed.jpg"
            cv2.imwrite(yolo_processed_path, img)
            
            return yolo_processed_path
            
        except Exception as e:
            print(f"Error in YOLO sliding window stage: {str(e)}")
            return image_path
    
    def process_image(self, input_path, output_path):
        """
        Process a single image with comprehensive detection pipeline.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to output image
            
        Returns:
            str: Path to processed image
        """
        # Print single message for new image being processed
        print(f"Processing new image: {os.path.basename(input_path)}")
        
        # Validate configuration
        if not self.nudenet_enabled and not self.yolo_enabled:
            print("Error: At least one detection method must be enabled (NudeNet or YOLO)")
            return None
        
        # Process the image through all stages
        current_image_path = input_path
        
        # Stage 1: NudeNet full image detection and blurring (if enabled)
        if self.nudenet_enabled:
            current_image_path = self.nudenet_full_image_stage(current_image_path)
        
        # Stage 2: NudeNet sliding window detection and blurring (if enabled)
        if self.nudenet_enabled:
            current_image_path = self.nudenet_sliding_window_stage(current_image_path)
        
        # Stage 3: YOLO full image detection and blurring (if enabled)
        if self.yolo_enabled:
            current_image_path = self.yolo_full_image_stage(current_image_path)
        
        # Stage 4: YOLO sliding window detection and blurring (if enabled)
        if self.yolo_enabled:
            current_image_path = self.yolo_sliding_window_stage(current_image_path)
        
        # Save final result
        try:
            import cv2
            import shutil
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Copy the final processed image to output path
            shutil.copy2(current_image_path, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error saving final result: {str(e)}")
            return None


def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Detection and Blurring Script with Two-Stage YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two-Stage Sliding Window Detector with NudeNet and YOLO
=====================================================

This script implements a comprehensive detection pipeline:
  Stage 1: NudeNet full image detection
  Stage 2: NudeNet sliding window detection
  Stage 3: YOLO full image detection
  Stage 4: YOLO sliding window detection

Usage examples:
  # Full pipeline (all stages enabled)
  python detect_slide.py input.jpg output.jpg
  # NudeNet only (stages 1-2)
  python detect_slide.py input.jpg output.jpg --use-yolo-two-stage=false
  # YOLO only (stages 3-4)
  python detect_slide.py input.jpg output.jpg --use-nudenet=false
  # NudeNet full image only (stage 1)
  python detect_slide.py input.jpg output.jpg --use-nudenet-two-stage=false --use-yolo-two-stage=false
  # Custom parameters
  python detect_slide.py input.jpg output.jpg --window-size 320 --stride 160 --nudenet-confidence 0.2 --yolo-confidence 0.3
        """
    )
    
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', nargs='?', help='Output image path (optional)')
    
    parser.add_argument('--nudenet-model', default='../models/640m.onnx',
                       help='Path to NudeNet model (default: ../models/640m.onnx)')
    
    parser.add_argument('--yolo-model', default='runs/detect/train15/weights/best.pt',
                       help='Path to YOLO model (default: runs/detect/train15/weights/best.pt)')
    
    parser.add_argument('--window-size', type=int, default=640,
                       help='Sliding window size in pixels (default: 640)')
    
    parser.add_argument('--stride', type=int, default=320,
                       help='Stride between sliding windows in pixels (default: 320)')
    
    parser.add_argument('--overlap-threshold', type=float, default=0.3,
                       help='IoU threshold for merging overlapping detections (default: 0.3)')
    
    parser.add_argument('--nudenet-confidence', type=float, default=0.1,
                       help='Minimum confidence score for NudeNet detections (default: 0.1)')
    
    parser.add_argument('--yolo-confidence', type=float, default=0.1,
                       help='Minimum confidence score for YOLO detections (default: 0.1)')
    
    parser.add_argument('--pixel-size', type=int, default=15,
                       help='Pixel size for pixelation (default: 15)')
    
    parser.add_argument('--blur-method', choices=['pixelate', 'blur'], default='pixelate',
                       help='Blurring method (default: pixelate)')
    
    parser.add_argument('--no-nudenet', action='store_true',
                       help='Disable NudeNet detection')
    
    parser.add_argument('--no-yolo', action='store_true',
                       help='Disable YOLO detection')
    
    parser.add_argument('--yolo-only', action='store_true',
                       help='Use only YOLO detection (disable NudeNet)')
    
    parser.add_argument('--nudenet-only', action='store_true',
                       help='Use only NudeNet detection (disable YOLO)')
    
    parser.add_argument('--use-nudenet', action='store_true', default=True,
                        help='Enable NudeNet detection (default: True)')
    
    parser.add_argument('--use-nudenet-two-stage', action='store_true', default=True,
                        help='Enable two-stage NudeNet detection (full image + sliding window) (default: True)')
    
    parser.add_argument('--use-yolo-two-stage', action='store_true', default=True,
                        help='Enable two-stage YOLO detection (full image + sliding window) (default: True)')
    
    args = parser.parse_args()
    
    # Determine which detection methods to use
    use_nudenet = not args.no_nudenet and not args.yolo_only
    use_nudenet_two_stage = use_nudenet and args.use_nudenet_two_stage
    use_yolo_two_stage = not args.no_yolo and not args.nudenet_only and args.use_yolo_two_stage
    
    # Initialize detector
    detector = TwoStageSlidingWindowDetector(
        nudenet_model_path=args.nudenet_model,
        yolo_model_path=args.yolo_model,
        window_size=args.window_size,
        stride=args.stride,
        overlap_threshold=args.overlap_threshold,
        nudenet_enabled=use_nudenet,
        yolo_enabled=use_yolo_two_stage,
        nudenet_confidence_threshold=args.nudenet_confidence,
        yolo_confidence_threshold=args.yolo_confidence,
        blur_method=args.blur_method,
        pixel_size=args.pixel_size
    )
    
    # Process the image
    output_path = detector.process_image(
        input_path=args.input,
        output_path=args.output
    )
    
    print(f"Processing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    # Example usage without command line arguments
    if len(sys.argv) == 1:
        # Use default settings for the example image with two-stage YOLO
        detector = TwoStageSlidingWindowDetector(
            nudenet_model_path="../models/640m.onnx",
            yolo_model_path="runs/detect/train15/weights/best.pt",
            window_size=160,
            stride=120,
            overlap_threshold=0.7
        )
        
        result = detector.process_image(
            input_path="../data/brazzers.jpg",
            output_path="./blurred_comprehensive.jpg"
        )
        
        print(f"Default comprehensive processing complete! Output saved to: {result}")
    else:
        main()