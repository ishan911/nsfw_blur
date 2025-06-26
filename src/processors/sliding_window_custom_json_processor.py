#!/usr/bin/env python3
"""
Sliding Window Custom JSON Image Processor Class

This module contains the SlidingWindowCustomJSONImageProcessor class that provides
sliding window image processing functionality for images specified in JSON format.
"""

import os
from datetime import datetime

from .custom_json_processor import CustomJSONImageProcessor
from ..blurrer import SlidingWindowBlurrer


class SlidingWindowCustomJSONImageProcessor(CustomJSONImageProcessor):
    def __init__(self, json_url=None, json_file=None, base_url=None, model_path='models/640m.onnx', 
                 database_path='data/processed_images.json', window_size=640, stride=320, overlap_threshold=0.3,
                 yolo_model_path=None):
        """Initialize sliding window custom JSON image processor."""
        # Initialize parent without creating the blurrer
        self.model_path = model_path
        self.database_path = database_path
        self.processed_images = self.load_database()
        self.json_url = json_url
        self.json_file = json_file
        self.base_url = base_url.rstrip('/') if base_url else None
        
        # Sliding window parameters
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        
        # YOLO model path
        self.yolo_model_path = yolo_model_path or "yolo_v8_model/runs/detect/train15/weights/best.pt"
        
        # Initialize the sliding window blurrer
        self.blurrer = SlidingWindowBlurrer(
            model_path=model_path, 
            parts=[
                'BUTTOCKS_EXPOSED',
                'BUTTOCKS_COVERED',
                'FEMALE_BREAST_EXPOSED',
                'FEMALE_GENITALIA_EXPOSED',
                'FEMALE_GENITALIA_COVERED',
                'FEMALE_BREAST_COVERED',
                'ANUS_EXPOSED',
                'ANUS_COVERED',
                'MALE_GENITALIA_EXPOSED',
            ],
            window_size=window_size,
            stride=stride,
            overlap_threshold=overlap_threshold
        )
    
    def is_already_processed(self, input_path, output_path, pixel_size=10, confidence_threshold=0.1):
        """Check if an image has already been processed with the same settings."""
        if input_path not in self.processed_images:
            return False
        
        record = self.processed_images[input_path]
        
        # Check if output file still exists
        if not os.path.exists(record['output_path']):
            return False
        
        # Check if file hash has changed (file was modified)
        current_hash = self.get_file_hash(input_path)
        if current_hash != record['file_hash']:
            return False
        
        # Check if processing settings are the same (including sliding window params)
        if (record['pixel_size'] != pixel_size or 
            record['output_path'] != output_path or
            record['model_path'] != self.model_path or
            record.get('window_size') != self.window_size or
            record.get('stride') != self.stride or
            record.get('overlap_threshold') != self.overlap_threshold or
            record.get('confidence_threshold') != confidence_threshold):
            return False
        
        return True
    
    def record_processed_image(self, input_path, output_path, pixel_size=10, confidence_threshold=0.1):
        """Record a processed image in the database with sliding window parameters."""
        file_hash = self.get_file_hash(input_path)
        
        self.processed_images[input_path] = {
            'output_path': output_path,
            'file_hash': file_hash,
            'pixel_size': pixel_size,
            'model_path': self.model_path,
            'window_size': self.window_size,
            'stride': self.stride,
            'overlap_threshold': self.overlap_threshold,
            'confidence_threshold': confidence_threshold,
            'processed_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(input_path) if os.path.exists(input_path) else 0
        }
        
        self.save_database()
    
    def process_single_image(self, input_path, output_path, pixel_size=10, confidence_threshold=0.1, force=False, image_type=None, 
                           use_yolo_detection=False, yolo_model_path=None, yolo_confidence_threshold=0.1,
                           use_nudenet_two_stage=True, use_yolo_two_stage=True, blur_method='pixelate'):
        """Process a single image with comprehensive four-stage detection pipeline and WordPress sizing."""
        # Check if already processed
        if not force and self.is_already_processed(input_path, output_path, pixel_size, confidence_threshold):
            return self.processed_images[input_path]['output_path']
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Print single message for new image being processed
            print(f"Processing new image: {os.path.basename(input_path)}")
            
            # Validate configuration
            if not use_nudenet_two_stage and not use_yolo_two_stage:
                print("Error: At least one detection method must be enabled (NudeNet or YOLO)")
                return None
            
            # Process the image through all stages
            current_image_path = input_path
            
            # Stage 1: NudeNet full image detection and blurring (if enabled)
            if use_nudenet_two_stage:
                current_image_path = self.nudenet_full_image_stage(
                    current_image_path, 
                    confidence_threshold, 
                    pixel_size, 
                    blur_method
                )
            
            # Stage 2: NudeNet sliding window detection and blurring (if enabled)
            if use_nudenet_two_stage:
                current_image_path = self.nudenet_sliding_window_stage(
                    current_image_path, 
                    confidence_threshold, 
                    pixel_size, 
                    blur_method
                )
            
            # Stage 3: YOLO full image detection and blurring (if enabled)
            if use_yolo_two_stage:
                current_image_path = self.yolo_full_image_stage(
                    current_image_path, 
                    yolo_confidence_threshold, 
                    pixel_size, 
                    blur_method
                )
            
            # Stage 4: YOLO sliding window detection and blurring (if enabled)
            if use_yolo_two_stage:
                current_image_path = self.yolo_sliding_window_stage(
                    current_image_path, 
                    yolo_confidence_threshold, 
                    pixel_size, 
                    blur_method
                )
            
            # Now use the original sliding window blurrer for WordPress sizing and resizing
            # This ensures we get the proper WordPress directory structure and resized images
            result = self.blurrer.process_image(
                input_path=current_image_path,  # Use the processed image from four-stage detection
                output_path=output_path,
                pixel_size=pixel_size,
                confidence_threshold=confidence_threshold,
                create_wordpress_sizes=True,
                image_type=image_type,
                use_scaled_detections=True  # Use scaled detections for better WordPress sizing
            )
            
            # Record the processed image
            self.record_processed_image(input_path, output_path, pixel_size, confidence_threshold)
            
            return output_path
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def run_yolo_detection(self, input_path, yolo_model_path=None, confidence_threshold=0.1, pixel_size=15):
        """
        Run YOLO detection on an image using the custom trained model and apply pixelation.
        
        Args:
            input_path (str): Path to input image
            yolo_model_path (str): Path to YOLO model (defaults to custom trained model)
            confidence_threshold (float): Minimum confidence for detections
            pixel_size (int): Pixel size for pixelation
            
        Returns:
            Tuple of (processed_image_path, detection_count) or (None, 0) if error
        """
        try:
            from ultralytics import YOLO
            import cv2
            import numpy as np
            import os
            
            # Use default custom model path if not specified
            if yolo_model_path is None:
                yolo_model_path = "/Users/ishanjayman/MrGeek/Projects/Daniel/AI/BlurImages/blurapi/yolo_v8_model/runs/detect/train10/weights/best.pt"
            
            model = YOLO(yolo_model_path)
            
            # Load the image
            img = cv2.imread(input_path)
            if img is None:
                print(f"Could not load image: {input_path}")
                return None, 0
            
            # Run detection
            results = model(input_path, conf=confidence_threshold)
            
            detection_count = 0
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # Extract detection information and apply pixelation
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        confidence = float(confs[i])
                        class_id = int(classes[i])
                        
                        # Apply pixelation to the detected region
                        roi = img[y1:y2, x1:x2]
                        if roi.size > 0:
                            # Resize down and up to create pixelation effect
                            small_roi = cv2.resize(roi, (max(1, roi.shape[1] // pixel_size), max(1, roi.shape[0] // pixel_size)))
                            pixelated_roi = cv2.resize(small_roi, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                            # Apply the pixelated region back to the image
                            img[y1:y2, x1:x2] = pixelated_roi
                            detection_count += 1
                    
                    # Save the YOLO-processed image to a temporary file
                    base_name = os.path.splitext(input_path)[0]
                    yolo_processed_path = f"{base_name}_yolo_processed.jpg"
                    cv2.imwrite(yolo_processed_path, img)
                    
                    return yolo_processed_path, detection_count
                else:
                    return input_path, 0  # Return original image path if no detections
            else:
                return input_path, 0  # Return original image path if no results
                
        except ImportError:
            return input_path, 0
        except Exception as e:
            print(f"Error in YOLO detection: {str(e)}")
            return input_path, 0
    
    def process_custom_json_images(self, output_dir="data/custom_processed", pixel_size=10, 
                                 confidence_threshold=0.1, force=False, download_only=False,
                                 use_yolo_detection=False, yolo_model_path=None, yolo_confidence_threshold=0.1,
                                 use_nudenet_two_stage=True, use_yolo_two_stage=True, blur_method='pixelate'):
        """Process images from custom JSON format with comprehensive four-stage detection pipeline."""
        json_data = self.fetch_json_data()
        
        if not json_data:
            print("No data found or error loading JSON.")
            return
        
        total_images = 0
        downloaded_count = 0
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for item in json_data:
            slug = item.get('slug', 'unknown')
            
            # Extract all image URLs from this item using extract_image_urls
            # This ensures base_url is properly joined for relative URLs
            image_urls = self.extract_image_urls(item)
            
            for image_info in image_urls:
                total_images += 1
                image_url = image_info['url']
                image_type = image_info['type']
                
                try:
                    # Download the image
                    local_path = self.download_image(image_url, output_dir, slug, image_type)
                    if local_path:
                        downloaded_count += 1
                        
                        # Process the image if not download-only
                        if not download_only:
                            output_path = self.get_output_path(local_path, output_dir, slug, image_type)
                            result = self.process_single_image(
                                local_path, output_path, pixel_size, confidence_threshold, force,
                                image_type, use_yolo_detection, yolo_model_path, yolo_confidence_threshold,
                                use_nudenet_two_stage, use_yolo_two_stage, blur_method
                            )
                            if result:
                                processed_count += 1
                            else:
                                error_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    print(f"    Error processing {image_type} for {slug}: {str(e)}")
                    error_count += 1
        
        print(f"\nFour-stage custom JSON processing complete:")
        print(f"  Total items: {len(json_data)}")
        print(f"  Total images found: {total_images}")
        print(f"  Downloaded: {downloaded_count} images")
        print(f"  Processed: {processed_count} images")
        print(f"  Skipped: {skipped_count} images")
        print(f"  Errors: {error_count} images")
        print(f"  Window size: {self.window_size}x{self.window_size}")
        print(f"  Stride: {self.stride} ({(self.window_size - self.stride) / self.stride * 100:.1f}% overlap)")
        
        # Display detection summary
        if use_nudenet_two_stage:
            print(f"  NudeNet two-stage: Enabled")
            print(f"  NudeNet confidence: {confidence_threshold}")
        if use_yolo_two_stage:
            print(f"  YOLO two-stage: Enabled")
            print(f"  YOLO model: {yolo_model_path or self.yolo_model_path}")
            print(f"  YOLO confidence: {yolo_confidence_threshold}")
        print(f"  Blur method: {blur_method}")
        print(f"  Pixel size: {pixel_size}")
        print(f"  WordPress sizing: Enabled (screenshot_full_url → 170x145, 250x212 | review_full_image → 590x504)")
    
    def nudenet_full_image_stage(self, image_path, nudenet_confidence_threshold=0.1, pixel_size=15, blur_method='pixelate'):
        """
        First stage: NudeNet full image detection and blurring.
        
        Args:
            image_path (str): Path to input image
            nudenet_confidence_threshold (float): Minimum confidence for NudeNet detections
            pixel_size (int): Pixel size for pixelation
            blur_method (str): 'pixelate' or 'blur'
            
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
            nudenet_results = self.blurrer.detector.detect(temp_path)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Process NudeNet detections
            nudenet_detection_count = 0
            
            for result in nudenet_results:
                if result['class'] in self.blurrer.parts and result['score'] >= nudenet_confidence_threshold:
                    nudenet_detection_count += 1
                    box = result['box']  # [x, y, w, h] format
                    score = result['score']
                    class_name = result['class']
                    
                    # Convert [x, y, w, h] to [x1, y1, x2, y2] for processing
                    x, y, w, h = box
                    region = [x, y, x + w, y + h]
                    
                    if blur_method == 'pixelate':
                        img = self.pixelate_region(img, region, pixel_size)
                    else:  # blur
                        img = self.blur_region(img, region, pixel_size)
            
            # Save the processed image
            base_name = os.path.splitext(image_path)[0]
            nudenet_processed_path = f"{base_name}_nudenet_full_processed.jpg"
            cv2.imwrite(nudenet_processed_path, img)
            
            return nudenet_processed_path
            
        except Exception as e:
            print(f"Error in NudeNet full image stage: {str(e)}")
            return image_path
    
    def nudenet_sliding_window_stage(self, image_path, nudenet_confidence_threshold=0.1, pixel_size=15, blur_method='pixelate'):
        """
        Second NudeNet stage: Sliding window detection and blurring.
        
        Args:
            image_path (str): Path to input image (already processed by NudeNet full image)
            nudenet_confidence_threshold (float): Minimum confidence for NudeNet detections
            pixel_size (int): Pixel size for pixelation
            blur_method (str): 'pixelate' or 'blur'
            
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
                window_results = self.blurrer.detector.detect(temp_window_path)
                
                # Clean up temporary file
                if os.path.exists(temp_window_path):
                    os.remove(temp_window_path)
                
                # Process window detections and translate coordinates
                for result in window_results:
                    if result['class'] in self.blurrer.parts and result['score'] >= nudenet_confidence_threshold:
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
                
                if blur_method == 'pixelate':
                    img = self.pixelate_region(img, box, pixel_size)
                else:  # blur
                    img = self.blur_region(img, box, pixel_size)
            
            # Save the processed image
            base_name = os.path.splitext(image_path)[0]
            nudenet_processed_path = f"{base_name}_nudenet_sliding_processed.jpg"
            cv2.imwrite(nudenet_processed_path, img)
            
            return nudenet_processed_path
            
        except Exception as e:
            print(f"Error in NudeNet sliding window stage: {str(e)}")
            return image_path 
    
    def yolo_full_image_stage(self, image_path, yolo_confidence_threshold=0.1, pixel_size=15, blur_method='pixelate'):
        """
        Third stage: YOLO full image detection and blurring.
        
        Args:
            image_path (str): Path to input image (already processed by NudeNet stages)
            yolo_confidence_threshold (float): Minimum confidence for YOLO detections
            pixel_size (int): Pixel size for pixelation
            blur_method (str): 'pixelate' or 'blur'
            
        Returns:
            Path to processed image
        """
        try:
            from ultralytics import YOLO
            import cv2
            import os
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return image_path
            
            height, width = img.shape[:2]
            
            # Load YOLO model
            model = YOLO(self.yolo_model_path)
            
            # Run detection
            results = model(image_path, conf=yolo_confidence_threshold)
            
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
                        
                        if blur_method == 'pixelate':
                            img = self.pixelate_region(img, [x1, y1, x2, y2], pixel_size)
                        else:  # blur
                            img = self.blur_region(img, [x1, y1, x2, y2], pixel_size)
                        
                        yolo_detection_count += 1
            
            # Save the processed image
            base_name = os.path.splitext(image_path)[0]
            yolo_processed_path = f"{base_name}_yolo_full_processed.jpg"
            cv2.imwrite(yolo_processed_path, img)
            
            return yolo_processed_path
            
        except ImportError:
            print("Warning: ultralytics not available. YOLO detection skipped.")
            return image_path
        except Exception as e:
            print(f"Error in YOLO full image stage: {str(e)}")
            return image_path
    
    def yolo_sliding_window_stage(self, image_path, yolo_confidence_threshold=0.1, pixel_size=15, blur_method='pixelate'):
        """
        Fourth stage: YOLO sliding window detection and blurring.
        
        Args:
            image_path (str): Path to input image (already processed by previous stages)
            yolo_confidence_threshold (float): Minimum confidence for YOLO detections
            pixel_size (int): Pixel size for pixelation
            blur_method (str): 'pixelate' or 'blur'
            
        Returns:
            Path to processed image
        """
        try:
            from ultralytics import YOLO
            import cv2
            import os
            import numpy as np
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return image_path
            
            height, width = img.shape[:2]
            
            # Load YOLO model
            model = YOLO(self.yolo_model_path)
            
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
                window_results = model(temp_window_path, conf=yolo_confidence_threshold)
                
                # Clean up temporary file
                if os.path.exists(temp_window_path):
                    os.remove(temp_window_path)
                
                # Process window detections and translate coordinates
                if window_results and len(window_results) > 0:
                    result = window_results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        
                        for j, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box)
                            confidence = float(confs[j])
                            class_id = int(classes[j])
                            
                            # Translate to original image coordinates
                            translated_x1 = x1 + window_x
                            translated_y1 = y1 + window_y
                            translated_x2 = x2 + window_x
                            translated_y2 = y2 + window_y
                            
                            # Create detection info
                            detection_info = {
                                'box': [translated_x1, translated_y1, translated_x2, translated_y2],
                                'conf': confidence,
                                'cls': f'class_{class_id}',
                                'detection': result
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
                
                if blur_method == 'pixelate':
                    img = self.pixelate_region(img, box, pixel_size)
                else:  # blur
                    img = self.blur_region(img, box, pixel_size)
            
            # Save the processed image
            base_name = os.path.splitext(image_path)[0]
            yolo_processed_path = f"{base_name}_yolo_sliding_processed.jpg"
            cv2.imwrite(yolo_processed_path, img)
            
            return yolo_processed_path
            
        except ImportError:
            print("Warning: ultralytics not available. YOLO detection skipped.")
            return image_path
        except Exception as e:
            print(f"Error in YOLO sliding window stage: {str(e)}")
            return image_path 
    
    def pixelate_region(self, image, region, pixel_size=10):
        """
        Apply pixelation to a specific region of the image.
        
        Args:
            image: OpenCV image array
            region: [x1, y1, x2, y2] coordinates
            pixel_size: Size of pixels for pixelation
            
        Returns:
            Modified image with pixelated region
        """
        import cv2
        import numpy as np
        
        x1, y1, x2, y2 = region
        
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
    
    def blur_region(self, image, region, blur_radius=20):
        """
        Apply blur to a specific region of the image.
        
        Args:
            image: OpenCV image array
            region: [x1, y1, x2, y2] coordinates
            blur_radius: Blur radius (must be odd)
            
        Returns:
            Modified image with blurred region
        """
        import cv2
        import numpy as np
        
        x1, y1, x2, y2 = region
        
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
    
    def create_sliding_windows(self, image_width, image_height):
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
    
    def crop_window(self, image, window):
        """
        Crop a window from the image.
        
        Args:
            image: OpenCV image array
            window: (x, y, width, height) of the window
            
        Returns:
            Cropped image window
        """
        x, y, w, h = window
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2] of first box
            box2: [x1, y1, x2, y2] of second box
            
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
    
    def merge_overlapping_detections(self, detections):
        """
        Merge overlapping detections based on IoU threshold.
        
        Args:
            detections: List of detection results
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
        merged_detections = []
        used_indices = set()
        
        for i, detection_info in enumerate(detections):
            if i in used_indices:
                continue
            
            current_box = detection_info['box']
            current_conf = detection_info['conf']
            current_cls = detection_info['cls']
            
            # Find overlapping boxes
            overlapping_indices = []
            for j, other_detection_info in enumerate(detections):
                if j <= i or j in used_indices:
                    continue
                
                other_box = other_detection_info['box']
                iou = self.calculate_iou(current_box, other_box)
                
                if iou > self.overlap_threshold:
                    overlapping_indices.append(j)
            
            # Merge overlapping boxes
            if overlapping_indices:
                # Use the box with highest confidence
                best_detection_info = detection_info
                best_conf = current_conf
                
                for idx in overlapping_indices:
                    other_detection_info = detections[idx]
                    if other_detection_info['conf'] > best_conf:
                        best_detection_info = other_detection_info
                        best_conf = other_detection_info['conf']
                
                merged_detections.append(best_detection_info)
                used_indices.add(i)
                used_indices.update(overlapping_indices)
            else:
                merged_detections.append(detection_info)
                used_indices.add(i)
        
        return merged_detections
    
    def download_image(self, image_url, output_dir, slug, image_type):
        """Download an image from URL to local path with proper directory structure."""
        try:
            import requests
            import os
            from urllib.parse import urlparse
            
            # Create the download directory structure
            download_dir = os.path.join(output_dir, "downloads", slug)
            os.makedirs(download_dir, exist_ok=True)
            
            # Generate filename based on image type
            parsed_url = urlparse(image_url)
            original_filename = os.path.basename(parsed_url.path)
            
            if not original_filename or '.' not in original_filename:
                # Fallback filename
                original_filename = f"{image_type}.jpg"
            
            # Create local path
            local_path = os.path.join(download_dir, original_filename)
            
            # Download the image
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return local_path
            
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
            return None
    
    def get_output_path(self, local_path, output_dir, slug, image_type):
        """Generate output path for processed image with WordPress directory structure."""
        import os
        from urllib.parse import urlparse
        
        # Create WordPress-style directory structure
        if image_type == 'screenshot_full_url':
            # Create screenshots directory
            wp_dir = os.path.join(output_dir, "wp-content", "uploads", "screenshots")
        else:  # review_full_image
            # Create regular uploads directory
            wp_dir = os.path.join(output_dir, "wp-content", "uploads")
        
        os.makedirs(wp_dir, exist_ok=True)
        
        # Get original filename
        original_filename = os.path.basename(local_path)
        
        # Create output path
        output_path = os.path.join(wp_dir, original_filename)
        
        return output_path 