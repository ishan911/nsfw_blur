import os
import numpy as np
from PIL import Image, ImageFilter
from nudenet import NudeDetector
from typing import List, Tuple, Dict, Any
import cv2

class SlidingWindowBlurrer:
    def __init__(self, model_path="models/640m.onnx", parts=None, window_size=640, stride=320, overlap_threshold=0.3):
        """
        Initialize SlidingWindowBlurrer with sliding window parameters.
        
        Args:
            model_path (str): Path to the ONNX model
            parts (list): List of body parts to detect and blur
            window_size (int): Size of the sliding window (width=height)
            stride (int): Stride between windows (should be <= window_size)
            overlap_threshold (float): Threshold for merging overlapping detections
        """
        print("Initializing SlidingWindowBlurrer")
        self.detector = NudeDetector(model_path=model_path)
        self.parts = parts or []
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        
        # WordPress image sizes
        self.wordpress_sizes = {
            'blog-tn': (170, 145, False),      # 510x315, cropped
            'category-thumb': (250, 212, True),  # 250x212, cropped
            'swiper-desktop': (590, 504, False)  # 590x504, not cropped
        }
        
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
    
    def crop_window(self, image: Image.Image, window: Tuple[int, int, int, int]) -> Image.Image:
        """
        Crop a window from the image.
        
        Args:
            image (Image.Image): Input image
            window (tuple): (x, y, width, height) of the window
            
        Returns:
            Cropped image window
        """
        x, y, w, h = window
        cropped = image.crop((x, y, x + w, y + h))
        
        # Convert to RGB if needed to ensure compatibility
        if cropped.mode in ['P', 'RGBA', 'LA', 'PA']:
            cropped = cropped.convert('RGB')
        
        return cropped
    
    def save_temp_window(self, window_image: Image.Image, temp_path: str) -> None:
        """
        Save a temporary window image with proper format handling.
        
        Args:
            window_image (Image.Image): Window image to save
            temp_path (str): Path to save the temporary file
        """
        # Ensure image is in RGB mode for compatibility
        if window_image.mode != 'RGB':
            window_image = window_image.convert('RGB')
        
        # Use PNG format to avoid JPEG mode issues
        if temp_path.endswith('.jpg'):
            temp_path = temp_path.replace('.jpg', '.png')
        
        window_image.save(temp_path, 'PNG')
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int], crop: bool = False) -> Image.Image:
        """
        Resize image to specified dimensions with optional cropping.
        
        Args:
            image (Image.Image): Input image
            size (tuple): (width, height) target size
            crop (bool): Whether to crop to exact size or maintain aspect ratio
            
        Returns:
            Resized image
        """
        target_width, target_height = size
        
        if crop:
            # Crop to exact size (WordPress style)
            # Calculate aspect ratios
            img_ratio = image.width / image.height
            target_ratio = target_width / target_height
            
            if img_ratio > target_ratio:
                # Image is wider, crop width
                new_width = int(image.height * target_ratio)
                left = (image.width - new_width) // 2
                image = image.crop((left, 0, left + new_width, image.height))
            else:
                # Image is taller, crop height
                new_height = int(image.width / target_ratio)
                top = (image.height - new_height) // 2
                image = image.crop((0, top, image.width, top + new_height))
        
        # Resize to target size
        resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return resized
    
    def create_wordpress_sizes(self, processed_image: Image.Image, base_filename: str, output_dir: str, image_type: str = None) -> List[str]:
        """
        Create WordPress-sized images from the processed image based on image type.
        
        Args:
            processed_image (Image.Image): The processed image
            base_filename (str): Base filename without extension
            output_dir (str): Output directory
            image_type (str): Type of image ('review_full_image', 'screenshot_full_url', etc.)
            
        Returns:
            List of created file paths
        """
        created_files = []
        
        # Determine which sizes to create based on image type
        if image_type == 'review_full_image':
            # Only create swiper-desktop size (590x504)
            sizes_to_create = ['swiper-desktop']
        elif image_type == 'screenshot_full_url':
            # Only create blog-tn and category-thumb sizes (170x145, 250x212)
            sizes_to_create = ['blog-tn', 'category-thumb']
        else:
            # Default: create all sizes
            sizes_to_create = list(self.wordpress_sizes.keys())
        
        for size_name in sizes_to_create:
            width, height, crop = self.wordpress_sizes[size_name]
            
            # Create resized image
            resized_image = self.resize_image(processed_image, (width, height), crop)
            
            # Generate filename
            if size_name == 'blog-tn':
                filename = f"{base_filename}-170x145.jpg"
            elif size_name == 'category-thumb':
                filename = f"{base_filename}-250x212.jpg"
            elif size_name == 'swiper-desktop':
                filename = f"{base_filename}-590x504.jpg"
            else:
                filename = f"{base_filename}-{width}x{height}.jpg"
            
            # Save resized image
            output_path = os.path.join(output_dir, filename)
            resized_image.save(output_path, 'JPEG', quality=85)
            created_files.append(output_path)
            print(f"  Created {size_name} size: {filename}")
        
        return created_files
    
    def translate_detection_coordinates(self, detection: Dict[str, Any], window: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Translate detection coordinates from window space to original image space.
        
        Args:
            detection (dict): Detection result from the model
            window (tuple): (x, y, width, height) of the window
            
        Returns:
            Detection with translated coordinates
        """
        window_x, window_y, window_w, window_h = window
        x, y, w, h = detection['box']
        
        # Translate coordinates to original image space
        translated_detection = detection.copy()
        translated_detection['box'] = [
            x + window_x,
            y + window_y,
            w,
            h
        ]
        
        return translated_detection
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1 (list): [x, y, width, height] of first box
            box2 (list): [x, y, width, height] of second box
            
        Returns:
            IoU value between 0 and 1
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
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def merge_overlapping_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping detections to avoid duplicates.
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
            
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            current_group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                # Check if detections are of the same class and overlap
                if (det1['class'] == det2['class'] and 
                    self.calculate_iou(det1['box'], det2['box']) > self.overlap_threshold):
                    current_group.append(det2)
                    used.add(j)
            
            # Merge the group
            if len(current_group) == 1:
                merged.append(current_group[0])
            else:
                # Take the detection with highest confidence
                best_detection = max(current_group, key=lambda x: x['score'])
                merged.append(best_detection)
        
        return merged
    
    def blur_region(self, image: Image.Image, region: List[int], blur_radius: int = 20) -> Image.Image:
        """Blur a specific region in the image."""
        x, y, w, h = region
        region_img = image.crop((x, y, x + w, y + h))
        blurred_region = region_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        image.paste(blurred_region, (x, y))
        return image

    def pixelate_region(self, image: Image.Image, region: List[int], pixel_size: int = 10) -> Image.Image:
        """Create a pixelated effect in a specific region of the image."""
        x, y, w, h = region
        
        # Crop the region
        region_img = image.crop((x, y, x + w, y + h))
        
        # Calculate new size (smaller size = more pixelated)
        small_w = max(w // pixel_size, 1)
        small_h = max(h // pixel_size, 1)
        
        # Resize down
        small_img = region_img.resize((small_w, small_h), Image.Resampling.NEAREST)
        
        # Resize back up
        pixelated_region = small_img.resize((w, h), Image.Resampling.NEAREST)
        
        # Paste the pixelated region back
        image.paste(pixelated_region, (x, y))
        return image
    
    def process_image(self, input_path: str, output_path: str = None, pixel_size: int = 15, 
                     confidence_threshold: float = 0.1, create_wordpress_sizes: bool = True, image_type: str = None) -> Image.Image:
        """
        Process an image using sliding window approach for better detection.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image
            pixel_size (int): Size of pixels for pixelation effect
            confidence_threshold (float): Minimum confidence score for detections
            create_wordpress_sizes (bool): Whether to create WordPress-sized images
            image_type (str): Type of image for WordPress sizing ('review_full_image', 'screenshot_full_url', etc.)
            
        Returns:
            Processed image
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
            
        # Load image
        image = Image.open(input_path)
        image_width, image_height = image.size
        
        print(f"Processing image: {image_width}x{image_height}")
        
        # Create sliding windows
        windows = self.create_sliding_windows(image_width, image_height)
        print(f"Created {len(windows)} sliding windows")
        
        all_detections = []
        
        # Process each window
        for i, window in enumerate(windows):
            print(f"Processing window {i+1}/{len(windows)}: {window}")
            
            # Crop window
            window_image = self.crop_window(image, window)
            
            # Save temporary window image for detection
            temp_path = f"temp_window_{i}.png"
            
            try:
                # Save window with proper format handling
                self.save_temp_window(window_image, temp_path)
                
                # Detect in window
                window_detections = self.detector.detect(temp_path)
                
                # Translate coordinates and filter by parts
                for detection in window_detections:
                    if not self.parts or detection['class'] in self.parts:
                        if detection['score'] >= confidence_threshold:
                            translated_detection = self.translate_detection_coordinates(detection, window)
                            all_detections.append(translated_detection)
                            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Merge overlapping detections
        print(f"Found {len(all_detections)} detections before merging")
        merged_detections = self.merge_overlapping_detections(all_detections)
        print(f"After merging: {len(merged_detections)} detections")
        
        # Apply blurring/pixelation
        for detection in merged_detections:
            print(f"Blurring {detection['class']} with confidence {detection['score']:.3f}")
            image = self.pixelate_region(image, detection['box'], pixel_size)
        
        # Save main processed image
        if output_path:
            # Remove 'processed_' prefix if present
            if os.path.basename(output_path).startswith('processed_'):
                output_path = output_path.replace('processed_', '', 1)
            
            # Modify output path to include WordPress uploads structure in root
            filename = os.path.basename(output_path)
            
            if image_type == 'review_full_image':
                # Save in root wp-content/uploads/screenshots
                wp_upload_dir = os.path.join('wp-content', 'uploads', 'screenshots')
                output_path = os.path.join(wp_upload_dir, filename)
            else:
                # Save in root wp-content/uploads
                wp_upload_dir = os.path.join('wp-content', 'uploads')
                output_path = os.path.join(wp_upload_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            print(f"Saved processed image to: {output_path}")
            
            # Create WordPress sizes if requested
            if create_wordpress_sizes:
                base_filename = os.path.splitext(os.path.basename(output_path))[0]
                output_dir = os.path.dirname(output_path)
                created_files = self.create_wordpress_sizes(image, base_filename, output_dir, image_type)
                print(f"Created {len(created_files)} WordPress-sized images")
        
        return image
    
    def process_image_with_visualization(self, input_path: str, output_path: str = None, 
                                       pixel_size: int = 10, confidence_threshold: float = 0.1,
                                       show_windows: bool = False, create_wordpress_sizes: bool = True, image_type: str = None) -> Image.Image:
        """
        Process image with optional visualization of sliding windows.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image
            pixel_size (int): Size of pixels for pixelation effect
            confidence_threshold (float): Minimum confidence score for detections
            show_windows (bool): Whether to visualize sliding windows on output
            create_wordpress_sizes (bool): Whether to create WordPress-sized images
            image_type (str): Type of image for WordPress sizing ('review_full_image', 'screenshot_full_url', etc.)
            
        Returns:
            Processed image
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
            
        # Load image
        image = Image.open(input_path)
        image_width, image_height = image.size
        
        # Create sliding windows
        windows = self.create_sliding_windows(image_width, image_height)
        
        all_detections = []
        
        # Process each window
        for i, window in enumerate(windows):
            window_image = self.crop_window(image, window)
            
            # Save temporary window image for detection
            temp_path = f"temp_window_{i}.png"
            
            try:
                # Save window with proper format handling
                self.save_temp_window(window_image, temp_path)
                
                window_detections = self.detector.detect(temp_path)
                
                for detection in window_detections:
                    if not self.parts or detection['class'] in self.parts:
                        if detection['score'] >= confidence_threshold:
                            translated_detection = self.translate_detection_coordinates(detection, window)
                            all_detections.append(translated_detection)
                            
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Merge overlapping detections
        merged_detections = self.merge_overlapping_detections(all_detections)
        
        # Apply blurring/pixelation
        for detection in merged_detections:
            image = self.pixelate_region(image, detection['box'], pixel_size)
        
        # Optionally draw window boundaries for visualization
        if show_windows:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            
            for window in windows:
                x, y, w, h = window
                draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
        
        # Save output
        if output_path:
            # Remove 'processed_' prefix if present
            if os.path.basename(output_path).startswith('processed_'):
                output_path = output_path.replace('processed_', '', 1)
            
            # Modify output path to include WordPress uploads structure in root
            filename = os.path.basename(output_path)
            
            if image_type == 'review_full_image':
                # Save in root wp-content/uploads/screenshots
                wp_upload_dir = os.path.join('wp-content', 'uploads', 'screenshots')
                output_path = os.path.join(wp_upload_dir, filename)
            else:
                # Save in root wp-content/uploads
                wp_upload_dir = os.path.join('wp-content', 'uploads')
                output_path = os.path.join(wp_upload_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            
            # Create WordPress sizes if requested
            if create_wordpress_sizes:
                base_filename = os.path.splitext(os.path.basename(output_path))[0]
                output_dir = os.path.dirname(output_path)
                created_files = self.create_wordpress_sizes(image, base_filename, output_dir, image_type)
                print(f"Created {len(created_files)} WordPress-sized images")
        
        return image 