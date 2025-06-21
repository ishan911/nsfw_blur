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
    
    def calculate_optimal_pixel_size(self, image_width: int, image_height: int, base_pixel_size: int = 10) -> int:
        """
        Calculate optimal pixel size for an image to ensure pixelation remains visible.
        
        Args:
            image_width (int): Width of the image
            image_height (int): Height of the image
            base_pixel_size (int): Base pixel size for reference image (e.g., 1920x1080)
            
        Returns:
            Optimal pixel size for the given image dimensions
        """
        # Reference dimensions (typical full-size image)
        reference_width = 1920
        reference_height = 1080
        
        # Calculate scale factor based on the smaller dimension
        scale_factor = min(image_width / reference_width, image_height / reference_height)
        
        # Calculate optimal pixel size
        optimal_size = max(1, int(base_pixel_size * scale_factor))
        
        # Ensure minimum visibility (at least 2x2 pixels)
        min_size = max(2, min(image_width, image_height) // 50)
        optimal_size = max(optimal_size, min_size)
        
        return optimal_size

    def create_wordpress_sizes_with_scaled_detections(self, original_image: Image.Image, detections: List[Dict[str, Any]], 
                                                    base_filename: str, output_dir: str, image_type: str = None, 
                                                    base_pixel_size: int = 10) -> List[str]:
        """
        Create WordPress-sized images by scaling detection coordinates and applying pixelation.
        This ensures consistent pixel size across all image sizes.
        
        Args:
            original_image (Image.Image): The original unprocessed image
            detections (List[Dict]): List of detection results
            base_filename (str): Base filename without extension
            output_dir (str): Output directory
            image_type (str): Type of image ('review_full_image', 'screenshot_full_url', etc.)
            base_pixel_size (int): Pixel size to use for all image sizes
            
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
            
            # First, resize the original image
            resized_image = self.resize_image(original_image, (width, height), crop)
            
            # Scale detection coordinates to the resized image
            scaled_detections = []
            for detection in detections:
                if detection['score'] > 0.1:  # Apply confidence threshold
                    scaled_detection = self.scale_detection_to_size(detection, original_image.size, (width, height))
                    if scaled_detection:  # Only add if detection is valid after scaling
                        scaled_detections.append(scaled_detection)
            
            # Apply pixelation to the resized image using scaled detections
            for detection in scaled_detections:
                resized_image = self.pixelate_region(resized_image, detection['box'], base_pixel_size)
            
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
            print(f"  Created {size_name} size: {filename} (pixel_size: {base_pixel_size}, detections: {len(scaled_detections)})")
        
        return created_files

    def scale_detection_to_size(self, detection: Dict[str, Any], original_size: Tuple[int, int], 
                               target_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Scale detection coordinates from original image size to target size.
        
        Args:
            detection (dict): Detection result from the model
            original_size (tuple): (width, height) of original image
            target_size (tuple): (width, height) of target image
            
        Returns:
            Detection with scaled coordinates, or None if detection is too small
        """
        original_width, original_height = original_size
        target_width, target_height = target_size
        
        # Calculate scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Scale the detection box
        x, y, w, h = detection['box']
        
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        scaled_w = int(w * scale_x)
        scaled_h = int(h * scale_y)
        
        # Ensure coordinates are within bounds
        scaled_x = max(0, min(scaled_x, target_width - 1))
        scaled_y = max(0, min(scaled_y, target_height - 1))
        scaled_w = max(1, min(scaled_w, target_width - scaled_x))
        scaled_h = max(1, min(scaled_h, target_height - scaled_y))
        
        # Only return detection if it's large enough to be meaningful
        min_size = 5  # Minimum detection size in pixels
        if scaled_w >= min_size and scaled_h >= min_size:
            scaled_detection = detection.copy()
            scaled_detection['box'] = [scaled_x, scaled_y, scaled_w, scaled_h]
            return scaled_detection
        else:
            return None

    def create_wordpress_sizes_adaptive_pixel(self, processed_image: Image.Image, base_filename: str, 
                                            output_dir: str, image_type: str = None, 
                                            base_pixel_size: int = 10) -> List[str]:
        """
        DEPRECATED: Use create_wordpress_sizes_with_scaled_detections instead.
        This method is kept for backward compatibility.
        """
        print("Warning: create_wordpress_sizes_adaptive_pixel is deprecated. Use create_wordpress_sizes_with_scaled_detections.")
        return self.create_wordpress_sizes(processed_image, base_filename, output_dir, image_type)

    def apply_subtle_enhancement(self, image: Image.Image, base_pixel_size: int) -> Image.Image:
        """
        DEPRECATED: This method is no longer used.
        """
        return image

    def create_wordpress_sizes_with_pixelation(self, original_image: Image.Image, detections: List[Dict[str, Any]], 
                                             base_filename: str, output_dir: str, image_type: str = None, 
                                             base_pixel_size: int = 10) -> List[str]:
        """
        Create WordPress-sized images with scale-aware pixelation.
        Applies pixelation after resizing to maintain visibility.
        
        Args:
            original_image (Image.Image): The original unprocessed image
            detections (List[Dict]): List of detection results
            base_filename (str): Base filename without extension
            output_dir (str): Output directory
            image_type (str): Type of image ('review_full_image', 'screenshot_full_url', etc.)
            base_pixel_size (int): Base pixel size for the original image
            
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
            
            # First, resize the original image
            resized_image = self.resize_image(original_image, (width, height), crop)
            
            # Calculate scale factor for this size
            scale_factor = min(width / original_image.width, height / original_image.height)
            
            # Scale the pixel size for this image size
            scaled_pixel_size = self.calculate_optimal_pixel_size(width, height, base_pixel_size)
            
            # Scale detection coordinates to the resized image
            scaled_detections = []
            for detection in detections:
                scaled_detection = detection.copy()
                x, y, w, h = detection['box']
                
                # Scale coordinates to the resized image
                scaled_x = int(x * scale_factor)
                scaled_y = int(y * scale_factor)
                scaled_w = int(w * scale_factor)
                scaled_h = int(h * scale_factor)
                
                # Ensure coordinates are within bounds
                scaled_x = max(0, min(scaled_x, width - 1))
                scaled_y = max(0, min(scaled_y, height - 1))
                scaled_w = max(1, min(scaled_w, width - scaled_x))
                scaled_h = max(1, min(scaled_h, height - scaled_y))
                
                scaled_detection['box'] = [scaled_x, scaled_y, scaled_w, scaled_h]
                scaled_detections.append(scaled_detection)
            
            # Apply pixelation to the resized image
            for detection in scaled_detections:
                if detection['score'] > 0.1:  # Apply confidence threshold
                    resized_image = self.pixelate_region(resized_image, detection['box'], scaled_pixel_size)
            
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
            print(f"  Created {size_name} size: {filename} (pixel_size: {scaled_pixel_size})")
        
        return created_files
    
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
        
        # Add 10px extra to all 4 sides
        extra_padding = 3
        x_expanded = max(0, x - extra_padding)
        y_expanded = max(0, y - extra_padding)
        w_expanded = min(w + 2 * extra_padding, image.width - x_expanded)
        h_expanded = min(h + 2 * extra_padding, image.height - y_expanded)
        
        # Crop the expanded region
        region_img = image.crop((x_expanded, y_expanded, x_expanded + w_expanded, y_expanded + h_expanded))
        
        # Calculate new size (smaller size = more pixelated)
        small_w = max(w_expanded // pixel_size, 1)
        small_h = max(h_expanded // pixel_size, 1)
        
        # Resize down
        small_img = region_img.resize((small_w, small_h), Image.Resampling.NEAREST)
        
        # Resize back up
        pixelated_region = small_img.resize((w_expanded, h_expanded), Image.Resampling.NEAREST)
        
        # Paste the pixelated region back
        image.paste(pixelated_region, (x_expanded, y_expanded))
        return image
    
    def process_image(self, input_path: str, output_path: str = None, pixel_size: int = 15, 
                     confidence_threshold: float = 0.1, create_wordpress_sizes: bool = True, 
                     image_type: str = None, use_scaled_detections: bool = True) -> Image.Image:
        """
        Process an image using sliding window approach for better detection.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image
            pixel_size (int): Size of pixels for pixelation effect
            confidence_threshold (float): Minimum confidence score for detections
            create_wordpress_sizes (bool): Whether to create WordPress-sized images
            image_type (str): Type of image for WordPress sizing ('review_full_image', 'screenshot_full_url', etc.)
            use_scaled_detections (bool): Whether to use scaled detections for WordPress sizes
            
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
        merged_detections = self.merge_overlapping_detections(all_detections)
        
        # Apply blurring/pixelation
        for detection in merged_detections:
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
                
                if use_scaled_detections:
                    # Load original image for scaled detection approach
                    original_image = Image.open(input_path)
                    
                    # Use scaled detection approach
                    created_files = self.create_wordpress_sizes_with_scaled_detections(
                        original_image, merged_detections, base_filename, output_dir, image_type, pixel_size
                    )
                    print(f"Created {len(created_files)} WordPress-sized images with scaled detections")
                else:
                    # Use original approach (pixelate first, then resize)
                    created_files = self.create_wordpress_sizes(image, base_filename, output_dir, image_type)
                    print(f"Created {len(created_files)} WordPress-sized images with original approach")
        
        return image
    
    def process_image_with_visualization(self, input_path: str, output_path: str = None, 
                                       pixel_size: int = 10, confidence_threshold: float = 0.1,
                                       show_windows: bool = False, create_wordpress_sizes: bool = True, 
                                       image_type: str = None, use_scaled_detections: bool = True) -> Image.Image:
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
            use_scaled_detections (bool): Whether to use scaled detections for WordPress sizes
            
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
                
                if use_scaled_detections:
                    # Load original image for scaled detection approach
                    original_image = Image.open(input_path)
                    
                    # Use scaled detection approach
                    created_files = self.create_wordpress_sizes_with_scaled_detections(
                        original_image, merged_detections, base_filename, output_dir, image_type, pixel_size
                    )
                    print(f"Created {len(created_files)} WordPress-sized images with scaled detections")
                else:
                    # Use original approach (pixelate first, then resize)
                    created_files = self.create_wordpress_sizes(image, base_filename, output_dir, image_type)
                    print(f"Created {len(created_files)} WordPress-sized images with original approach")
        
        return image 