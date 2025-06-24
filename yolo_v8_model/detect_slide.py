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
    def __init__(self, nudenet_model_path="../models/640m.onnx", yolo_model_path="runs/detect/train8/weights/best.pt",
                 window_size=640, stride=320, overlap_threshold=0.3):
        """
        Initialize TwoStageSlidingWindowDetector with both NudeNet and YOLO models.
        
        Args:
            nudenet_model_path (str): Path to the NudeNet ONNX model
            yolo_model_path (str): Path to the YOLO model
            window_size (int): Size of the sliding window (width=height)
            stride (int): Stride between windows (should be <= window_size)
            overlap_threshold (float): Threshold for merging overlapping detections
        """
        print("Initializing TwoStageSlidingWindowDetector")
        print("  NudeNet model: {}".format(nudenet_model_path))
        print("  YOLO model: {}".format(yolo_model_path))
        
        # Initialize NudeNet detector
        self.nudenet_detector = NudeDetector(model_path=nudenet_model_path)
        
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Sliding window parameters
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        
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
    
    def nudenet_stage(self, image: np.ndarray, nudenet_confidence_threshold: float = 0.1,
                     pixel_size: int = 15, blur_method: str = 'pixelate') -> np.ndarray:
        """
        First stage: NudeNet detection and blurring.
        
        Args:
            image (np.ndarray): Input image
            nudenet_confidence_threshold (float): Minimum confidence for NudeNet detections
            pixel_size (int): Pixel size for pixelation
            blur_method (str): 'pixelate' or 'blur'
            
        Returns:
            Image with NudeNet detections blurred
        """
        print("=== Stage 1: NudeNet Detection ===")
        
        # Convert numpy array to PIL Image for NudeNet
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Save temporary image for NudeNet detection
        temp_path = "temp_nudenet.jpg"
        pil_image.save(temp_path)
        
        # Run NudeNet detection
        nudenet_results = self.nudenet_detector.detect(temp_path)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"NudeNet found {len(nudenet_results)} detections")
        
        # Process NudeNet detections
        processed_image = image.copy()
        nudenet_detection_count = 0
        
        for result in nudenet_results:
            if result['class'] in self.nudenet_parts and result['score'] >= nudenet_confidence_threshold:
                nudenet_detection_count += 1
                box = result['box']  # [x, y, w, h] format
                score = result['score']
                class_name = result['class']
                
                print(f"  NudeNet: {class_name} (confidence: {score:.3f}) at {box}")
                
                # Convert [x, y, w, h] to [x1, y1, x2, y2] for processing
                x, y, w, h = box
                region = [x, y, x + w, y + h]
                
                if blur_method == 'pixelate':
                    processed_image = self.pixelate_region(processed_image, region, pixel_size)
                else:  # blur
                    processed_image = self.blur_region(processed_image, region, pixel_size)
        
        print(f"NudeNet stage processed {nudenet_detection_count} detections")
        return processed_image
    
    def yolo_stage(self, image: np.ndarray, yolo_confidence_threshold: float = 0.1,
                  pixel_size: int = 15, blur_method: str = 'pixelate') -> np.ndarray:
        """
        Second stage: YOLO sliding window detection and blurring.
        
        Args:
            image (np.ndarray): Input image (already processed by NudeNet)
            yolo_confidence_threshold (float): Minimum confidence for YOLO detections
            pixel_size (int): Pixel size for pixelation
            blur_method (str): 'pixelate' or 'blur'
            
        Returns:
            Image with YOLO detections blurred
        """
        print("=== Stage 2: YOLO Sliding Window Detection ===")
        print(f"Window size: {self.window_size}x{self.window_size}")
        print(f"Stride: {self.stride}")
        print(f"Overlap threshold: {self.overlap_threshold}")
        print(f"Confidence threshold: {yolo_confidence_threshold}")
        
        height, width = image.shape[:2]
        print(f"Image dimensions: {width}x{height}")
        
        # Create sliding windows
        windows = self.create_sliding_windows(width, height)
        print(f"Created {len(windows)} sliding windows")
        
        # Process each window
        all_detections = []
        
        for i, window in enumerate(windows):
            window_x, window_y, window_w, window_h = window
            print(f"Processing window {i+1}/{len(windows)}: ({window_x}, {window_y}, {window_w}, {window_h})")
            
            # Crop the window
            window_image = self.crop_window(image, window)
            
            # Run YOLO detection on the window
            results = self.yolo_model(window_image, conf=yolo_confidence_threshold)
            
            if results and len(results) > 0:
                result = results[0]
                
                # Check if any detections were found
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # Translate coordinates to original image space
                    translated_result = self.translate_detection_coordinates(result, window)
                    all_detections.append(translated_result)
                    print(f"  Found {len(result.boxes)} YOLO detections in window")
                else:
                    print(f"  No YOLO detections in window")
            else:
                print(f"  No YOLO detections in window")
        
        # Merge overlapping detections
        print(f"Total YOLO detections before merging: {len(all_detections)}")
        merged_detections = self.merge_overlapping_detections(all_detections)
        print(f"Total YOLO detections after merging: {len(merged_detections)}")
        
        # Apply blurring/pixelation to YOLO detected regions
        processed_image = image.copy()
        yolo_detection_count = 0
        
        for detection_info in merged_detections:
            yolo_detection_count += 1
            box = detection_info['box']
            conf = detection_info['conf']
            cls = detection_info['cls']
            
            print(f"Processing YOLO detection: box={box}, confidence={conf:.3f}, class={cls}")
            
            if blur_method == 'pixelate':
                processed_image = self.pixelate_region(processed_image, box, pixel_size)
            else:  # blur
                processed_image = self.blur_region(processed_image, box, pixel_size)
        
        print(f"YOLO stage processed {yolo_detection_count} detections")
        return processed_image
    
    def process_image(self, input_path: str, output_path: str = None, 
                     pixel_size: int = 15, nudenet_confidence_threshold: float = 0.1,
                     yolo_confidence_threshold: float = 0.1, blur_method: str = 'pixelate') -> str:
        """
        Process an image using two-stage detection: NudeNet first, then YOLO sliding window.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to output image (optional)
            pixel_size (int): Pixel size for pixelation
            nudenet_confidence_threshold (float): Minimum confidence for NudeNet detections
            yolo_confidence_threshold (float): Minimum confidence for YOLO detections
            blur_method (str): 'pixelate' or 'blur'
            
        Returns:
            Path to processed image
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"blurred_two_stage_{base_name}.jpg"
        
        print(f"Processing image: {input_path}")
        print(f"Two-stage detection pipeline:")
        print(f"  1. NudeNet detection (confidence ≥ {nudenet_confidence_threshold})")
        print(f"  2. YOLO sliding window detection (confidence ≥ {yolo_confidence_threshold})")
        print(f"  Blur method: {blur_method}")
        print(f"  Pixel size: {pixel_size}")
        
        # Load the image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Stage 1: NudeNet detection and blurring
        image_after_nudenet = self.nudenet_stage(
            image, 
            nudenet_confidence_threshold, 
            pixel_size, 
            blur_method
        )
        
        # Stage 2: YOLO sliding window detection and blurring
        final_image = self.yolo_stage(
            image_after_nudenet, 
            yolo_confidence_threshold, 
            pixel_size, 
            blur_method
        )
        
        # Save the processed image
        cv2.imwrite(output_path, final_image)
        print(f"Two-stage processing complete! Output saved to: {output_path}")
        
        return output_path


def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Two-Stage Sliding Window Detection and Blurring Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jpg output.jpg
  %(prog)s input.jpg --window-size 512 --stride 256
  %(prog)s input.jpg --nudenet-confidence 0.2 --yolo-confidence 0.1
  %(prog)s input.jpg --blur-method blur --pixel-size 20
        """
    )
    
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', nargs='?', help='Output image path (optional)')
    
    parser.add_argument('--nudenet-model', default='../models/640m.onnx',
                       help='Path to NudeNet model (default: ../models/640m.onnx)')
    
    parser.add_argument('--yolo-model', default='runs/detect/train8/weights/best.pt',
                       help='Path to YOLO model (default: runs/detect/train8/weights/best.pt)')
    
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
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = TwoStageSlidingWindowDetector(
        nudenet_model_path=args.nudenet_model,
        yolo_model_path=args.yolo_model,
        window_size=args.window_size,
        stride=args.stride,
        overlap_threshold=args.overlap_threshold
    )
    
    # Process the image
    output_path = detector.process_image(
        input_path=args.input,
        output_path=args.output,
        pixel_size=args.pixel_size,
        nudenet_confidence_threshold=args.nudenet_confidence,
        yolo_confidence_threshold=args.yolo_confidence,
        blur_method=args.blur_method
    )
    
    print(f"Processing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    # Example usage without command line arguments
    if len(sys.argv) == 1:
        # Use default settings for the example image
        detector = TwoStageSlidingWindowDetector(
            nudenet_model_path="../models/640m.onnx",
            yolo_model_path="runs/detect/train8/weights/best.pt",
            window_size=640,
            stride=320,
            overlap_threshold=0.3
        )
        
        result = detector.process_image(
            input_path="../data/Teen-Porn-Video-55.jpg",
            output_path="blurred_two_stage.jpg",
            pixel_size=15,
            nudenet_confidence_threshold=0.1,
            yolo_confidence_threshold=0.1,
            blur_method='pixelate'
        )
        
        print(f"Default two-stage processing complete! Output saved to: {result}")
    else:
        main()