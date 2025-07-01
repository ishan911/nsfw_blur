#!/usr/bin/env python3
"""
Enhanced Image Processing Script with NudeNet and YOLO Detection
This script combines NudeNet detection with YOLO detection for comprehensive content filtering.
Includes WordPress sizing and folder structure support with database tracking.
"""

import os
import sys
import argparse
import json
import requests
import hashlib
from datetime import datetime
from pathlib import Path
import cv2
from PIL import Image
from nudenet_detector import NudeNetDetector
from ultralytics import YOLO
import numpy as np

# Configuration
WORDPRESS_SIZES = {
    'blog-tn': (170, 145, False),      # 510x315, cropped
    'category-thumb': (250, 212, True),  # 250x212, cropped
    'swiper-desktop': (590, 504, False)  # 590x504, not cropped
}

# YOLO configuration - set to False to disable YOLO detection if there are compatibility issues
ENABLE_YOLO_DETECTION = True

class DatabaseTracker:
    """Simple database tracker for processed images."""
    
    def __init__(self, database_path='data/processed_images.json'):
        self.database_path = database_path
        self.processed_images = self.load_database()
    
    def load_database(self):
        """Load the processed images database."""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r') as f:
                    return json.load(f)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
                return {}
        except Exception as e:
            print(f"Warning: Could not load database: {e}")
            return {}
    
    def save_database(self):
        """Save the processed images database."""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.processed_images, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save database: {e}")
    
    def get_file_hash(self, file_path):
        """Generate a hash for the file to detect changes."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Warning: Could not generate hash for {file_path}: {e}")
            return None
    
    def is_already_processed(self, input_path, output_path, pixel_size=15, confidence_threshold=0.05):
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
        
        # Check if processing settings are the same
        if (record['pixel_size'] != pixel_size or 
            record['output_path'] != output_path or
            record.get('confidence_threshold') != confidence_threshold):
            return False
        
        return True
    
    def record_processed_image(self, input_path, output_path, pixel_size=15, confidence_threshold=0.05, 
                             nudenet_detections=0, yolo_detections=0, wordpress_files=None, image_type=None):
        """Record a processed image in the database."""
        file_hash = self.get_file_hash(input_path)
        
        self.processed_images[input_path] = {
            'output_path': output_path,
            'file_hash': file_hash,
            'pixel_size': pixel_size,
            'confidence_threshold': confidence_threshold,
            'nudenet_detections': nudenet_detections,
            'yolo_detections': yolo_detections,
            'wordpress_files': wordpress_files or [],
            'image_type': image_type,
            'processed_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(input_path) if os.path.exists(input_path) else 0
        }
        
        self.save_database()
        print(f"  📝 Recorded in database: {os.path.basename(input_path)}")
    
    def get_processing_stats(self):
        """Get statistics about processed images."""
        total_images = len(self.processed_images)
        total_size = sum(record.get('file_size', 0) for record in self.processed_images.values())
        total_nudenet = sum(record.get('nudenet_detections', 0) for record in self.processed_images.values())
        total_yolo = sum(record.get('yolo_detections', 0) for record in self.processed_images.values())
        
        print(f"\n📊 Database Statistics:")
        print(f"  Total images processed: {total_images}")
        print(f"  Total size processed: {total_size / (1024*1024):.2f} MB")
        print(f"  Total NudeNet detections: {total_nudenet}")
        print(f"  Total YOLO detections: {total_yolo}")
        
        if total_images > 0:
            # Show recent processing
            recent = sorted(
                self.processed_images.items(),
                key=lambda x: x[1]['processed_at'],
                reverse=True
            )[:5]
            
            print(f"  Recent processing:")
            for input_path, record in recent:
                filename = os.path.basename(input_path)
                print(f"    {filename} -> {os.path.basename(record['output_path'])} ({record['processed_at'][:19]})")

# Initialize database tracker
db_tracker = DatabaseTracker()

def download_image(url, download_dir="downloads"):
    """
    Download an image from URL to local directory.
    
    Args:
        url (str): Image URL to download
        download_dir (str): Directory to save downloaded images
        
    Returns:
        str: Path to downloaded image file, or None if failed
    """
    try:
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        # Extract filename from URL
        filename = url.split('/')[-1]
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            filename += '.jpg'  # Default extension
        
        filepath = os.path.join(download_dir, filename)
        
        # Download the image
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {filename}")
        return filepath
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def resize_image(image, target_size, crop=False):
    """
    Resize image to target size with optional cropping.
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Target (width, height)
        crop (bool): Whether to crop to exact size
        
    Returns:
        PIL.Image: Resized image
    """
    target_width, target_height = target_size
    
    if crop:
        # For cropping, first resize to cover the target area, then crop
        # Calculate aspect ratios
        target_ratio = target_width / target_height
        image_ratio = image.width / image.height
        
        if image_ratio > target_ratio:
            # Image is wider than target, resize by height first
            new_height = target_height
            new_width = int(image.width * (target_height / image.height))
        else:
            # Image is taller than target, resize by width first
            new_width = target_width
            new_height = int(image.height * (target_width / image.width))
        
        # Resize image to cover target area
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop to exact target size
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return resized.crop((left, top, right, bottom))
    else:
        # For non-cropping, resize maintaining aspect ratio
        # Calculate new size that fits within target dimensions
        image_ratio = image.width / image.height
        target_ratio = target_width / target_height
        
        if image_ratio > target_ratio:
            # Image is wider, fit by width
            new_width = target_width
            new_height = int(target_width / image_ratio)
        else:
            # Image is taller, fit by height
            new_height = target_height
            new_width = int(target_height * image_ratio)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target background (white or transparent)
        if resized.mode in ('RGBA', 'LA'):
            # For images with transparency, use transparent background
            new_image = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
        else:
            # For opaque images, use white background
            new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        
        # Paste resized image in center
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_image.paste(resized, (paste_x, paste_y))
        
        return new_image

def create_wordpress_versions(original_image_path, processed_image_path, base_filename, image_type=None):
    """
    Create WordPress-sized images from the processed image based on image type.
    
    Args:
        original_image_path (str): Path to original image
        processed_image_path (str): Path to processed image
        base_filename (str): Base filename without extension
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
    elif image_type == 'category_thumb':
        # For category thumbnails, create category-thumb size (250x212)
        sizes_to_create = ['category-thumb']
    else:
        # Default: create all sizes
        sizes_to_create = list(WORDPRESS_SIZES.keys())
    
    # Detect original image format
    original_format = 'JPEG'  # Default
    if original_image_path.lower().endswith('.png'):
        original_format = 'PNG'
    elif original_image_path.lower().endswith(('.jpg', '.jpeg')):
        original_format = 'JPEG'
    
    # Determine file extension and save parameters
    if original_format.upper() == 'PNG':
        file_extension = '.png'
        save_format = 'PNG'
        save_kwargs = {}
    else:
        file_extension = '.jpg'
        save_format = 'JPEG'
        save_kwargs = {'quality': 85}
    
    # Load the processed image
    processed_image = Image.open(processed_image_path)
    
    for size_name in sizes_to_create:
        width, height, crop = WORDPRESS_SIZES[size_name]
        
        # Create resized image
        resized_image = resize_image(processed_image, (width, height), crop)
        
        # Generate filename with correct extension
        if size_name == 'blog-tn':
            filename = f"{base_filename}-170x145{file_extension}"
        elif size_name == 'category-thumb':
            filename = f"{base_filename}-250x212{file_extension}"
        elif size_name == 'swiper-desktop':
            filename = f"{base_filename}-590x504{file_extension}"
        else:
            filename = f"{base_filename}-{width}x{height}{file_extension}"
        
        # Determine output directory based on image type
        if image_type == 'review_full_image':
            # Save in wp-content/uploads/screenshots
            wp_upload_dir = os.path.join('wp-content', 'uploads', 'screenshots')
        elif image_type == 'category_thumb':
            # Save in wp-content/uploads/cthumbnails
            wp_upload_dir = os.path.join('wp-content', 'uploads', 'cthumbnails')
        else:
            # Save in wp-content/uploads
            wp_upload_dir = os.path.join('wp-content', 'uploads')
        
        # Create output directory
        os.makedirs(wp_upload_dir, exist_ok=True)
        
        # Save resized image with correct format
        output_path = os.path.join(wp_upload_dir, filename)
        resized_image.save(output_path, save_format, **save_kwargs)
        created_files.append(output_path)
        print(f"  Created {size_name} size: {filename}")
    
    return created_files

def create_wordpress_sizes_with_pixelation(original_image_path, detections, base_filename, image_type=None, pixel_size=15):
    """
    Create WordPress-sized images with consistent pixelation applied after resizing.
    
    Args:
        original_image_path (str): Path to original image
        detections (list): List of detection dictionaries from NudeNet
        base_filename (str): Base filename without extension
        image_type (str): Type of image ('review_full_image', 'screenshot_full_url', etc.)
        pixel_size (int): Pixel size to maintain across all image sizes
        
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
    elif image_type == 'category_thumb':
        # For category thumbnails, create category-thumb size (250x212)
        sizes_to_create = ['category-thumb']
    else:
        # Default: create all sizes
        sizes_to_create = list(WORDPRESS_SIZES.keys())
    
    # Detect original image format
    original_format = 'JPEG'  # Default
    if original_image_path.lower().endswith('.png'):
        original_format = 'PNG'
    elif original_image_path.lower().endswith(('.jpg', '.jpeg')):
        original_format = 'JPEG'
    
    # Determine file extension and save parameters
    if original_format.upper() == 'PNG':
        file_extension = '.png'
        save_format = 'PNG'
        save_kwargs = {}
    else:
        file_extension = '.jpg'
        save_format = 'JPEG'
        save_kwargs = {'quality': 85}
    
    # Load the original image
    original_image = Image.open(original_image_path)
    original_width, original_height = original_image.size
    
    for size_name in sizes_to_create:
        width, height, crop = WORDPRESS_SIZES[size_name]
        
        # First, resize the original image
        resized_image = resize_image(original_image, (width, height), crop)
        
        # Calculate scale factors for detection coordinates
        if crop:
            # For cropped images, calculate the scale factor based on the resize operation
            scale_x = width / original_width
            scale_y = height / original_height
            scale_factor = max(scale_x, scale_y)  # Use the larger scale factor for cropping
        else:
            # For non-cropped images, calculate scale factor
            scale_x = width / original_width
            scale_y = height / original_height
            scale_factor = min(scale_x, scale_y)  # Use the smaller scale factor to fit within bounds
        
        # Scale detection coordinates to the resized image
        scaled_detections = []
        for detection in detections:
            if detection['score'] > 0.1:  # Apply confidence threshold
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
                
                scaled_detections.append({
                    'box': [scaled_x, scaled_y, scaled_w, scaled_h],
                    'score': detection['score'],
                    'class': detection['class']
                })
        
        # Apply pixelation to the resized image using the same pixel_size
        if scaled_detections:
            # Convert PIL image to OpenCV format for pixelation
            resized_cv = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
            
            # Apply pixelation to each detection
            for detection in scaled_detections:
                x, y, w, h = detection['box']
                
                # Add 5px padding (scaled)
                padding = max(1, int(5 * scale_factor))
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                
                # Pixelate the region
                resized_cv = pixelate_region_cv(resized_cv, x1, y1, x2, y2, pixel_size)
            
            # Convert back to PIL
            resized_image = Image.fromarray(cv2.cvtColor(resized_cv, cv2.COLOR_BGR2RGB))
        
        # Generate filename with correct extension
        if size_name == 'blog-tn':
            filename = f"{base_filename}-170x145{file_extension}"
        elif size_name == 'category-thumb':
            filename = f"{base_filename}-250x212{file_extension}"
        elif size_name == 'swiper-desktop':
            filename = f"{base_filename}-590x504{file_extension}"
        else:
            filename = f"{base_filename}-{width}x{height}{file_extension}"
        
        # Determine output directory based on image type
        if image_type == 'review_full_image':
            # Save in wp-content/uploads/screenshots
            wp_upload_dir = os.path.join('wp-content', 'uploads', 'screenshots')
        elif image_type == 'category_thumb':
            # Save in wp-content/uploads/cthumbnails
            wp_upload_dir = os.path.join('wp-content', 'uploads', 'cthumbnails')
        else:
            # Save in wp-content/uploads
            wp_upload_dir = os.path.join('wp-content', 'uploads')
        
        # Create output directory
        os.makedirs(wp_upload_dir, exist_ok=True)
        
        # Save resized image with correct format
        output_path = os.path.join(wp_upload_dir, filename)
        resized_image.save(output_path, save_format, **save_kwargs)
        created_files.append(output_path)
        print(f"  Created {size_name} size: {filename} (pixel_size: {pixel_size}, detections: {len(scaled_detections)})")
    
    return created_files

def pixelate_region_cv(img, x1, y1, x2, y2, pixel_size):
    """
    Pixelate a region of an OpenCV image with fixed pixel size.
    
    Args:
        img (np.ndarray): Input OpenCV image
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

def process_single_image(input_path, output_path, nudenet_detector, yolo_model, image_type=None, force=False, draw_rectangles=False, draw_labels=False):
    """
    Process a single image with both NudeNet and YOLO detection.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to output image
        nudenet_detector (NudeNetDetector): NudeNet detector instance
        yolo_model (YOLO): YOLO model instance
        image_type (str): Type of image for WordPress sizing
        force (bool): Force reprocessing even if already processed
        draw_rectangles (bool): Whether to draw rectangle borders for debugging
        draw_labels (bool): Whether to draw labels on rectangles for debugging
        
    Returns:
        dict: Processing results
    """
    try:
        print(f"Processing: {input_path}")
        
        # Check if already processed (unless force is True)
        if not force and db_tracker.is_already_processed(input_path, output_path):
            record = db_tracker.processed_images[input_path]
            print(f"  ⏭️  Already processed: {os.path.basename(input_path)}")
            print(f"    Output: {os.path.basename(record['output_path'])}")
            print(f"    Processed: {record['processed_at'][:19]}")
            print(f"    NudeNet detections: {record['nudenet_detections']}")
            print(f"    YOLO detections: {record['yolo_detections']}")
            return {
                'success': True,
                'nudenet_detections': record['nudenet_detections'],
                'yolo_detections': record['yolo_detections'],
                'total_detections': record['nudenet_detections'] + record['yolo_detections'],
                'wordpress_files': record.get('wordpress_files', []),
                'message': f"Already processed - NudeNet: {record['nudenet_detections']}, YOLO: {record['yolo_detections']}"
            }
        
        # Step 1: NudeNet detection and pixelation
        print("  Running NudeNet detection...")
        nudenet_result = nudenet_detector.process_image(
            input_path=input_path,
            output_path=output_path,
            use_sliding_window=True,
            draw_rectangles=draw_rectangles,
            draw_labels=draw_labels
        )
        
        if not nudenet_result['success']:
            print(f"  NudeNet processing failed: {nudenet_result['message']}")
            return nudenet_result
        
        print(f"  NudeNet detections: {nudenet_result['detection_count']}")
        
        # Step 2: YOLO detection and blurring
        print("  Running YOLO detection...")
        yolo_detections = []
        
        if not ENABLE_YOLO_DETECTION:
            print("    YOLO detection disabled in configuration")
        else:
            try:
                # Run YOLO detection with better error handling
                yolo_results = yolo_model(output_path, verbose=False)
                
                # Handle YOLO results properly with error checking
                if isinstance(yolo_results, list) and len(yolo_results) > 0:
                    result = yolo_results[0]  # Get first result
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            print(f"    Found {len(boxes)} YOLO detections")
                            
                            # Convert boxes to detection format
                            for box in boxes:
                                try:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    confidence = box.conf[0].cpu().numpy()
                                    class_id = int(box.cls[0].cpu().numpy())
                                    
                                    yolo_detections.append({
                                        'box': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                        'score': float(confidence),
                                        'class': f'yolo_class_{class_id}'
                                    })
                                except Exception as box_error:
                                    print(f"    Error processing YOLO box: {box_error}")
                                    continue
                            
                            # Apply additional YOLO blurring if needed
                            if yolo_detections:
                                img = cv2.imread(output_path)
                                for detection in yolo_detections:
                                    try:
                                        x1, y1, w, h = detection['box']
                                        x2, y2 = x1 + w, y1 + h
                                        roi = img[y1:y2, x1:x2]
                                        roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
                                        img[y1:y2, x1:x2] = roi_blur
                                    except Exception as blur_error:
                                        print(f"    Error applying blur to detection: {blur_error}")
                                        continue
                                
                                cv2.imwrite(output_path, img)
                                print(f"    Applied YOLO blurring to {len(yolo_detections)} regions")
                        else:
                            print("    No YOLO detections found")
                    else:
                        print("    No YOLO detections found")
                else:
                    print("    No YOLO detections found")
                
            except Exception as e:
                print(f"    Error in YOLO detection: {e}")
                print("    Skipping YOLO detection due to error")
                yolo_detections = []
        
        # Step 3: Create WordPress versions
        print("  Creating WordPress versions...")
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        created_files = create_wordpress_versions(
            input_path, 
            output_path, 
            base_filename, 
            image_type
        )
        print(f"  Created {len(created_files)} WordPress-sized images")
        
        # Step 4: Record in database
        db_tracker.record_processed_image(
            input_path=input_path,
            output_path=output_path,
            pixel_size=nudenet_detector.pixel_size,
            confidence_threshold=nudenet_detector.confidence_threshold,
            nudenet_detections=nudenet_result['detection_count'],
            yolo_detections=len(yolo_detections),
            wordpress_files=created_files,
            image_type=image_type
        )
        
        return {
            'success': True,
            'nudenet_detections': nudenet_result['detection_count'],
            'yolo_detections': len(yolo_detections),
            'total_detections': nudenet_result['detection_count'] + len(yolo_detections),
            'wordpress_files': created_files,
            'message': f"Processed successfully - NudeNet: {nudenet_result['detection_count']}, YOLO: {len(yolo_detections)}"
        }
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }

def process_single_image_enhanced(input_path, output_path, nudenet_detector, yolo_model, image_type=None, force=False, draw_rectangles=False, draw_labels=False):
    """
    Process a single image with enhanced detection methods.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save processed image
        nudenet_detector: NudeNet detector instance
        yolo_model: YOLO model instance
        image_type (str): Type of image for WordPress sizing
        force (bool): Force reprocessing
        draw_rectangles (bool): Whether to draw rectangle borders for debugging
        draw_labels (bool): Whether to draw labels on rectangles for debugging
        
    Returns:
        dict: Processing result
    """
    try:
        # Check if output already exists
        if os.path.exists(output_path) and not force:
            return {
                'success': True,
                'message': 'Already exists',
                'nudenet_detections': 0,
                'yolo_detections': 0,
                'wordpress_files': []
            }
        
        # Use the enhanced NudeNetDetector approach
        print(f"  Running enhanced detection with full image first...")
        nudenet_result = nudenet_detector.process_image(
            input_path=input_path,
            output_path=output_path,
            use_sliding_window=True,  # This will use detect_with_full_image_first
            draw_rectangles=draw_rectangles,
            draw_labels=draw_labels
        )
        
        if not nudenet_result['success']:
            print(f"  Enhanced processing failed: {nudenet_result['message']}")
            return nudenet_result
        
        print(f"  Enhanced detections: {nudenet_result['detection_count']}")
        
        # Run YOLO detection on the processed image
        print(f"  Running YOLO detection...")
        yolo_detections = []
        
        if not ENABLE_YOLO_DETECTION:
            print("    YOLO detection disabled in configuration")
        else:
            try:
                # Run YOLO detection with better error handling
                yolo_results = yolo_model(output_path, verbose=False)
                
                # Handle YOLO results properly with error checking
                if isinstance(yolo_results, list) and len(yolo_results) > 0:
                    result = yolo_results[0]  # Get first result
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            print(f"    Found {len(boxes)} YOLO detections")
                            
                            # Convert boxes to detection format
                            for box in boxes:
                                try:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    confidence = box.conf[0].cpu().numpy()
                                    class_id = int(box.cls[0].cpu().numpy())
                                    
                                    yolo_detections.append({
                                        'box': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                        'score': float(confidence),
                                        'class': f'yolo_class_{class_id}'
                                    })
                                except Exception as box_error:
                                    print(f"    Error processing YOLO box: {box_error}")
                                    continue
                            
                            # Apply additional YOLO blurring if needed
                            if yolo_detections:
                                img = cv2.imread(output_path)
                                for detection in yolo_detections:
                                    try:
                                        x1, y1, w, h = detection['box']
                                        x2, y2 = x1 + w, y1 + h
                                        roi = img[y1:y2, x1:x2]
                                        roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
                                        img[y1:y2, x1:x2] = roi_blur
                                    except Exception as blur_error:
                                        print(f"    Error applying blur to detection: {blur_error}")
                                        continue
                                
                                cv2.imwrite(output_path, img)
                                print(f"    Applied YOLO blurring to {len(yolo_detections)} regions")
                        else:
                            print("    No YOLO detections found")
                    else:
                        print("    No YOLO detections found")
                else:
                    print("    No YOLO detections found")
                
            except Exception as e:
                print(f"    Error in YOLO detection: {e}")
                print("    Skipping YOLO detection due to error")
                yolo_detections = []
        
        # Create WordPress versions if image_type is specified
        wordpress_files = []
        if image_type and image_type != 'category_thumb':
            base_filename = os.path.splitext(os.path.basename(output_path))[0]
            wordpress_files = create_wordpress_versions(
                input_path, 
                output_path, 
                base_filename, 
                image_type
            )
            print(f"  Created {len(wordpress_files)} WordPress-sized images")
        
        # Record in database
        db_tracker.record_processed_image(
            input_path=input_path,
            output_path=output_path,
            pixel_size=nudenet_detector.pixel_size,
            confidence_threshold=nudenet_detector.confidence_threshold,
            nudenet_detections=nudenet_result['detection_count'],
            yolo_detections=len(yolo_detections),
            wordpress_files=wordpress_files,
            image_type=image_type
        )
        
        return {
            'success': True,
            'nudenet_detections': nudenet_result['detection_count'],
            'yolo_detections': len(yolo_detections),
            'total_detections': nudenet_result['detection_count'] + len(yolo_detections),
            'wordpress_files': wordpress_files,
            'message': f"Enhanced processing completed - NudeNet: {nudenet_result['detection_count']}, YOLO: {len(yolo_detections)}"
        }
        
    except Exception as e:
        print(f"Error in enhanced processing: {e}")
        return {
            'success': False,
            'message': f"Error: {str(e)}",
            'nudenet_detections': 0,
            'yolo_detections': 0,
            'wordpress_files': []
        }

def sliding_json(json_url, output_dir="processed_images", base_url=None, force=False, download_only=False, draw_rectangles=False, draw_labels=False):
    """
    Process images from a JSON URL using sliding window detection.
    
    Args:
        json_url (str): URL to JSON file containing image data
        output_dir (str): Directory to save processed images
        base_url (str): Base URL for converting relative paths to absolute URLs
        force (bool): Force reprocessing even if output already exists
        download_only (bool): Only download images, do not process them
        draw_rectangles (bool): Whether to draw rectangle borders for debugging
        draw_labels (bool): Whether to draw labels on rectangles for debugging
        
    Returns:
        dict: Processing summary
    """
    try:
        print(f"=== Sliding JSON Processing ===")
        print(f"JSON URL: {json_url}")
        print(f"Output directory: {output_dir}")
        print(f"Base URL: {base_url}")
        print(f"Force reprocessing: {force}")
        print(f"Download only: {download_only}")
        print(f"Draw rectangles: {draw_rectangles}")
        print(f"Draw labels: {draw_labels}")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create backup directory for original images
        backup_dir = os.path.join(output_dir, "backup", "sliding")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Download JSON data
        print("Downloading JSON data...")
        response = requests.get(json_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"JSON data loaded successfully")
        
        # Initialize detectors
        if not download_only:
            print("Initializing detectors...")
            nudenet_detector = NudeNetDetector(
                confidence_threshold=0.05,
                pixel_size=15,
                padding=10
            )
            
            # Initialize YOLO model with error handling
            yolo_model = None
            if ENABLE_YOLO_DETECTION:
                try:
                    yolo_model = YOLO("yolo_v8_model/runs/detect/train15/weights/best.pt")
                    print("YOLO model initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize YOLO model: {e}")
                    print("Continuing without YOLO detection...")
                    yolo_model = None
            else:
                print("YOLO detection disabled in configuration")
            
            print("Detectors initialized successfully")
        
        # Process images
        processed_count = 0
        skipped_count = 0
        error_count = 0
        download_count = 0
        
        # Extract image URLs from JSON data
        image_urls = []
        
        # Handle the specific JSON structure format
        if isinstance(data, list):
            # Direct list structure: [{"slug": "...", "screenshot_full_url": "...", "review_full_image": "..."}, ...]
            for item in data:
                if isinstance(item, dict):
                    # Process screenshot_full_url
                    if 'screenshot_full_url' in item:
                        url = item['screenshot_full_url']
                        if url:
                            # Handle relative URLs with base_url
                            if base_url and not url.startswith(('http://', 'https://')):
                                url = base_url.rstrip('/') + '/' + url.lstrip('/')
                            image_urls.append({
                                'url': url,
                                'type': 'screenshot_full_url',
                                'slug': item.get('slug', 'unknown')
                            })
                    
                    # Process review_full_image
                    if 'review_full_image' in item:
                        url = item['review_full_image']
                        if url:
                            # Handle relative URLs with base_url
                            if base_url and not url.startswith(('http://', 'https://')):
                                url = base_url.rstrip('/') + '/' + url.lstrip('/')
                            image_urls.append({
                                'url': url,
                                'type': 'review_full_image',
                                'slug': item.get('slug', 'unknown')
                            })
        
        elif isinstance(data, dict):
            # Check for the specific structure used in the previous implementation
            if 'data' in data and isinstance(data['data'], list):
                # Structure: {"data": [{"screenshot_full_url": "...", "review_full_image": "...", "slug": "..."}, ...]}
                for item in data['data']:
                    if isinstance(item, dict):
                        # Process screenshot_full_url
                        if 'screenshot_full_url' in item:
                            url = item['screenshot_full_url']
                            if url:
                                if base_url and not url.startswith(('http://', 'https://')):
                                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                image_urls.append({
                                    'url': url,
                                    'type': 'screenshot_full_url',
                                    'slug': item.get('slug', 'unknown')
                                })
                        
                        # Process review_full_image
                        if 'review_full_image' in item:
                            url = item['review_full_image']
                            if url:
                                if base_url and not url.startswith(('http://', 'https://')):
                                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                image_urls.append({
                                    'url': url,
                                    'type': 'review_full_image',
                                    'slug': item.get('slug', 'unknown')
                                })
            
            elif 'images' in data and isinstance(data['images'], list):
                # Structure: {"images": [{"screenshot_full_url": "...", "review_full_image": "...", "slug": "..."}, ...]}
                for item in data['images']:
                    if isinstance(item, dict):
                        # Process screenshot_full_url
                        if 'screenshot_full_url' in item:
                            url = item['screenshot_full_url']
                            if url:
                                if base_url and not url.startswith(('http://', 'https://')):
                                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                image_urls.append({
                                    'url': url,
                                    'type': 'screenshot_full_url',
                                    'slug': item.get('slug', 'unknown')
                                })
                        
                        # Process review_full_image
                        if 'review_full_image' in item:
                            url = item['review_full_image']
                            if url:
                                if base_url and not url.startswith(('http://', 'https://')):
                                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                image_urls.append({
                                    'url': url,
                                    'type': 'review_full_image',
                                    'slug': item.get('slug', 'unknown')
                                })
            
            elif 'screenshot_full_url' in data:
                # Single image structure: {"screenshot_full_url": "...", "review_full_image": "...", "slug": "..."}
                url = data['screenshot_full_url']
                if base_url and not url.startswith(('http://', 'https://')):
                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                image_urls.append({
                    'url': url,
                    'type': 'screenshot_full_url',
                    'slug': data.get('slug', 'unknown')
                })
            
            elif 'review_full_image' in data:
                # Single image structure: {"review_full_image": "...", "slug": "..."}
                url = data['review_full_image']
                if base_url and not url.startswith(('http://', 'https://')):
                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                image_urls.append({
                    'url': url,
                    'type': 'review_full_image',
                    'slug': data.get('slug', 'unknown')
                })
        
        print(f"Found {len(image_urls)} image URLs to process")
        
        for i, image_data in enumerate(image_urls, 1):
            try:
                print(f"\n[{i}/{len(image_urls)}] Processing: {image_data['url']}")
                
                # Download image
                downloaded_path = download_image(image_data['url'])
                if not downloaded_path:
                    error_count += 1
                    continue
                
                download_count += 1
                
                # Create backup of original image
                backup_filename = os.path.basename(downloaded_path)
                backup_path = os.path.join(backup_dir, backup_filename)
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(downloaded_path, backup_path)
                    print(f"  📁 Backed up to: {backup_path}")
                
                if download_only:
                    continue
                
                # Determine image type based on URL or JSON structure
                image_type = image_data['type']
                
                print(f"  Detected image type: {image_type}")
                
                # Determine output path with WordPress structure
                filename = os.path.basename(downloaded_path)
                
                if image_type == 'review_full_image':
                    # Save in wp-content/uploads/screenshots
                    wp_upload_dir = os.path.join('wp-content', 'uploads', 'screenshots')
                    output_path = os.path.join(wp_upload_dir, filename)
                    print(f" Screenshots Output directory: {wp_upload_dir}")
                else:
                    # Save in wp-content/uploads
                    wp_upload_dir = os.path.join('wp-content', 'uploads')
                    output_path = os.path.join(wp_upload_dir, filename)
                
                print(f"  Output directory: {wp_upload_dir}")
                print(f"  Output path: {output_path}")
                
                # Create output directory structure
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Also create the screenshots folder for review_full_image type
                if image_type == 'review_full_image':
                    screenshots_dir = os.path.join('wp-content', 'uploads', 'screenshots')
                    os.makedirs(screenshots_dir, exist_ok=True)
                    print(f"  Created screenshots directory: {screenshots_dir}")
                
                # Check if already processed
                if os.path.exists(output_path) and not force:
                    print(f"  Skipped (already exists): {output_path}")
                    skipped_count += 1
                    continue
                
                # Process the image
                result = process_single_image(
                    downloaded_path, 
                    output_path, 
                    nudenet_detector, 
                    yolo_model,
                    image_type,
                    force,
                    draw_rectangles,
                    draw_labels
                )
                
                if result['success']:
                    processed_count += 1
                    print(f"  Success: {output_path}")
                    print(f"    NudeNet detections: {result['nudenet_detections']}")
                    print(f"    YOLO detections: {result['yolo_detections']}")
                    print(f"    WordPress files: {len(result['wordpress_files'])}")
                else:
                    error_count += 1
                    print(f"  Failed: {result['message']}")
                
            except Exception as e:
                error_count += 1
                print(f"  Error processing {image_data['url']}: {e}")
                continue
        
        # Summary
        print(f"\n=== Processing Summary ===")
        print(f"Total images: {len(image_urls)}")
        print(f"Downloaded: {download_count}")
        if not download_only:
            print(f"Processed: {processed_count}")
            print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        
        # Show database statistics
        if not download_only:
            db_tracker.get_processing_stats()
        
        return {
            'success': True,
            'total_images': len(image_urls),
            'downloaded': download_count,
            'processed': processed_count if not download_only else 0,
            'skipped': skipped_count if not download_only else 0,
            'errors': error_count
        }
        
    except Exception as e:
        print(f"Error in sliding_json: {e}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }

def category_thumbnails(json_url, output_dir="processed_images", base_url=None, force=False, download_only=False, draw_rectangles=False, draw_labels=False):
    """
    Process category thumbnail images from a JSON URL using enhanced detection.
    
    Args:
        json_url (str): URL to JSON file containing category thumbnail data
        output_dir (str): Directory to save processed images
        base_url (str): Base URL for converting relative paths to absolute URLs
        force (bool): Force reprocessing even if output already exists
        download_only (bool): Only download images, do not process them
        draw_rectangles (bool): Whether to draw rectangle borders for debugging
        draw_labels (bool): Whether to draw labels on rectangles for debugging
        
    Returns:
        dict: Processing summary
    """
    try:
        print(f"=== Category Thumbnails Processing (Enhanced) ===")
        print(f"JSON URL: {json_url}")
        print(f"Output directory: {output_dir}")
        print(f"Base URL: {base_url}")
        print(f"Force reprocessing: {force}")
        print(f"Download only: {download_only}")
        print(f"Draw rectangles: {draw_rectangles}")
        print(f"Draw labels: {draw_labels}")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create backup directory for original images
        backup_dir = os.path.join(output_dir, "backup", "category")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Download JSON data
        print("Downloading JSON data...")
        response = requests.get(json_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"JSON data loaded successfully")
        
        # Initialize detectors
        if not download_only:
            print("Initializing enhanced detectors...")
            nudenet_detector = NudeNetDetector(
                confidence_threshold=0.05,
                pixel_size=15,
                padding=10
            )
            
            # Initialize YOLO model with error handling
            yolo_model = None
            if ENABLE_YOLO_DETECTION:
                try:
                    yolo_model = YOLO("yolo_v8_model/runs/detect/train15/weights/best.pt")
                    print("YOLO model initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize YOLO model: {e}")
                    print("Continuing without YOLO detection...")
                    yolo_model = None
            else:
                print("YOLO detection disabled in configuration")
            
            print("Enhanced detectors initialized successfully")
        
        # Process images
        processed_count = 0
        skipped_count = 0
        error_count = 0
        download_count = 0
        
        # Extract image URLs from JSON data
        image_urls = []
        
        # Handle the category thumbnails JSON structure
        if isinstance(data, list):
            # Direct list structure: [{"term_id": 1364, "count": 14, "slug": "...", "thumb": "..."}, ...]
            for item in data:
                if isinstance(item, dict) and 'thumb' in item:
                    url = item['thumb']
                    if url:
                        # Handle relative URLs with base_url
                        if base_url and not url.startswith(('http://', 'https://')):
                            url = base_url.rstrip('/') + '/' + url.lstrip('/')
                        image_urls.append({
                            'url': url,
                            'type': 'category_thumb',
                            'term_id': item.get('term_id', 'unknown'),
                            'slug': item.get('slug', 'unknown'),
                            'count': item.get('count', 0)
                        })
        
        elif isinstance(data, dict):
            # Check for nested structures
            if 'data' in data and isinstance(data['data'], list):
                # Structure: {"data": [{"term_id": 1364, "count": 14, "slug": "...", "thumb": "..."}, ...]}
                for item in data['data']:
                    if isinstance(item, dict) and 'thumb' in item:
                        url = item['thumb']
                        if url:
                            if base_url and not url.startswith(('http://', 'https://')):
                                url = base_url.rstrip('/') + '/' + url.lstrip('/')
                            image_urls.append({
                                'url': url,
                                'type': 'category_thumb',
                                'term_id': item.get('term_id', 'unknown'),
                                'slug': item.get('slug', 'unknown'),
                                'count': item.get('count', 0)
                            })
            
            elif 'thumbnails' in data and isinstance(data['thumbnails'], list):
                # Structure: {"thumbnails": [{"term_id": 1364, "count": 14, "slug": "...", "thumb": "..."}, ...]}
                for item in data['thumbnails']:
                    if isinstance(item, dict) and 'thumb' in item:
                        url = item['thumb']
                        if url:
                            if base_url and not url.startswith(('http://', 'https://')):
                                url = base_url.rstrip('/') + '/' + url.lstrip('/')
                            image_urls.append({
                                'url': url,
                                'type': 'category_thumb',
                                'term_id': item.get('term_id', 'unknown'),
                                'slug': item.get('slug', 'unknown'),
                                'count': item.get('count', 0)
                            })
        
        print(f"Found {len(image_urls)} category thumbnail URLs to process")
        
        for i, image_data in enumerate(image_urls, 1):
            try:
                print(f"\n[{i}/{len(image_urls)}] Processing: {image_data['url']}")
                print(f"  Term ID: {image_data['term_id']}, Slug: {image_data['slug']}, Count: {image_data['count']}")
                
                # Download image
                downloaded_path = download_image(image_data['url'])
                if not downloaded_path:
                    error_count += 1
                    continue
                
                download_count += 1
                
                # Create backup of original image
                backup_filename = os.path.basename(downloaded_path)
                backup_path = os.path.join(backup_dir, backup_filename)
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(downloaded_path, backup_path)
                    print(f"  📁 Backed up to: {backup_path}")
                
                if download_only:
                    continue
                
                # Determine output path with WordPress structure for category thumbnails
                filename = os.path.basename(downloaded_path)
                
                # Save in wp-content/uploads/cthumbnails
                wp_upload_dir = os.path.join('wp-content', 'uploads', 'cthumbnails')
                output_path = os.path.join(wp_upload_dir, filename)
                
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Check if already processed
                if os.path.exists(output_path) and not force:
                    print(f"  Skipped (already exists): {output_path}")
                    skipped_count += 1
                    continue
                
                # Process the image with enhanced detection
                result = process_single_image_enhanced(
                    downloaded_path, 
                    output_path, 
                    nudenet_detector, 
                    yolo_model,
                    'category_thumb',
                    force,
                    draw_rectangles,
                    draw_labels
                )
                
                if result['success']:
                    processed_count += 1
                    print(f"  Success: {output_path}")
                    print(f"    NudeNet detections: {result['nudenet_detections']}")
                    print(f"    YOLO detections: {result['yolo_detections']}")
                    print(f"    WordPress files: {len(result['wordpress_files'])}")
                else:
                    error_count += 1
                    print(f"  Failed: {result['message']}")
                
            except Exception as e:
                error_count += 1
                print(f"  Error processing {image_data['url']}: {e}")
                continue
        
        # Summary
        print(f"\n=== Category Thumbnails Processing Summary ===")
        print(f"Total thumbnails: {len(image_urls)}")
        print(f"Downloaded: {download_count}")
        if not download_only:
            print(f"Processed: {processed_count}")
            print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        
        # Show database statistics
        if not download_only:
            db_tracker.get_processing_stats()
        
        return {
            'success': True,
            'total_thumbnails': len(image_urls),
            'downloaded': download_count,
            'processed': processed_count if not download_only else 0,
            'skipped': skipped_count if not download_only else 0,
            'errors': error_count
        }
        
    except Exception as e:
        print(f"Error in category_thumbnails: {e}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }

def sliding_single(image_path, output_dir="processed_images", image_type=None, force=False, draw_rectangles=False, draw_labels=False):
    """
    Process a single image using sliding window detection.
    
    Args:
        image_path (str): Path to input image file
        output_dir (str): Directory to save processed images
        image_type (str): Type of image (screenshot_full_url, review_full_image, category_thumb, etc.)
        force (bool): Force reprocessing even if output already exists
        draw_rectangles (bool): Whether to draw rectangle borders for debugging
        draw_labels (bool): Whether to draw labels on rectangles for debugging
        
    Returns:
        dict: Processing summary
    """
    try:
        print(f"=== Sliding Single Image Processing ===")
        print(f"Input image: {image_path}")
        print(f"Output directory: {output_dir}")
        print(f"Image type: {image_type}")
        print(f"Force reprocessing: {force}")
        print(f"Draw rectangles: {draw_rectangles}")
        print(f"Draw labels: {draw_labels}")
        print()
        
        # Check if input is a URL or file path
        is_url = image_path.startswith(('http://', 'https://'))
        
        if is_url:
            print(f"  Detected URL: {image_path}")
            # Download the image
            downloaded_path = download_image(image_path)
            if not downloaded_path:
                return {
                    'success': False,
                    'message': f"Failed to download image from URL: {image_path}"
                }
            print(f"  Downloaded to: {downloaded_path}")
            local_image_path = downloaded_path
        else:
            # Check if local file exists
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'message': f"Input file not found: {image_path}"
                }
            local_image_path = image_path
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create backup directory for original images
        backup_dir = os.path.join(output_dir, "backup", "single")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Initialize detectors
        print("Initializing detectors...")
        nudenet_detector = NudeNetDetector(
            confidence_threshold=0.05,
            pixel_size=15,
            padding=10
        )
        
        # Initialize YOLO model with error handling
        yolo_model = None
        if ENABLE_YOLO_DETECTION:
            try:
                yolo_model = YOLO("yolo_v8_model/runs/detect/train15/weights/best.pt")
                print("YOLO model initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize YOLO model: {e}")
                print("Continuing without YOLO detection...")
                yolo_model = None
        else:
            print("YOLO detection disabled in configuration")
        
        print("Detectors initialized successfully")
        
        # Determine image type if not provided
        if image_type is None:
            # Try to infer from filename or path
            filename = os.path.basename(local_image_path).lower()
            if 'screenshot' in filename or 'screen' in filename:
                image_type = 'screenshot_full_url'
            elif 'review' in filename:
                image_type = 'review_full_image'
            elif 'thumb' in filename or 'category' in filename:
                image_type = 'category_thumb'
            else:
                image_type = 'screenshot_full_url'  # Default
        
        print(f"  Detected image type: {image_type}")
        
        # Determine output path with WordPress structure
        filename = os.path.basename(local_image_path)
        
        if image_type == 'review_full_image':
            # Save in wp-content/uploads/screenshots
            wp_upload_dir = os.path.join('wp-content', 'uploads', 'screenshots')
            output_path = os.path.join(wp_upload_dir, filename)
            print(f"  Screenshots Output directory: {wp_upload_dir}")
        elif image_type == 'category_thumb':
            # Save in wp-content/uploads/cthumbnails
            wp_upload_dir = os.path.join('wp-content', 'uploads', 'cthumbnails')
            output_path = os.path.join(wp_upload_dir, filename)
            print(f"  Category thumbnails Output directory: {wp_upload_dir}")
        else:
            # Save in wp-content/uploads
            wp_upload_dir = os.path.join('wp-content', 'uploads')
            output_path = os.path.join(wp_upload_dir, filename)
            print(f"  Output directory: {wp_upload_dir}")
        
        print(f"  Output path: {output_path}")
        
        # Create output directory structure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Also create the screenshots folder for review_full_image type
        if image_type == 'review_full_image':
            screenshots_dir = os.path.join('wp-content', 'uploads', 'screenshots')
            os.makedirs(screenshots_dir, exist_ok=True)
            print(f"  Created screenshots directory: {screenshots_dir}")
        
        # Check if already processed
        if os.path.exists(output_path) and not force:
            print(f"  Skipped (already exists): {output_path}")
            return {
                'success': True,
                'total_images': 1,
                'downloaded': 1 if is_url else 0,
                'processed': 0,
                'skipped': 1,
                'errors': 0
            }
        
        # Create backup of original image
        backup_filename = os.path.basename(local_image_path)
        backup_path = os.path.join(backup_dir, backup_filename)
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(local_image_path, backup_path)
            print(f"  📁 Backed up to: {backup_path}")
        
        # Process the image
        print(f"Processing image: {filename}")
        result = process_single_image(
            local_image_path, 
            output_path, 
            nudenet_detector, 
            yolo_model,
            image_type,
            force,
            draw_rectangles,
            draw_labels
        )
        
        if result['success']:
            print(f"  Success: {output_path}")
            print(f"    NudeNet detections: {result['nudenet_detections']}")
            print(f"    YOLO detections: {result['yolo_detections']}")
            print(f"    WordPress files: {len(result['wordpress_files'])}")
            
            # Summary
            print(f"\n=== Processing Summary ===")
            print(f"Total images: 1")
            print(f"Downloaded: {1 if is_url else 0}")
            print(f"Processed: 1")
            print(f"Errors: 0")
            
            # Show database statistics
            db_tracker.get_processing_stats()
            
            return {
                'success': True,
                'total_images': 1,
                'downloaded': 1 if is_url else 0,
                'processed': 1,
                'skipped': 0,
                'errors': 0
            }
        else:
            print(f"  Failed: {result['message']}")
            return {
                'success': False,
                'message': result['message']
            }
        
    except Exception as e:
        print(f"Error in sliding_single: {e}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }

def blog_images(json_url, output_dir="processed_images", base_url=None, force=False, download_only=False, draw_rectangles=False, draw_labels=False):
    """
    Process blog images from a JSON URL using enhanced detection.
    
    Args:
        json_url (str): URL to JSON file containing blog image data
        output_dir (str): Directory to save processed images
        base_url (str): Base URL for converting relative paths to absolute URLs
        force (bool): Force reprocessing even if output already exists
        download_only (bool): Only download images, do not process them
        draw_rectangles (bool): Whether to draw rectangle borders for debugging
        draw_labels (bool): Whether to draw labels on rectangles for debugging
        
    Returns:
        dict: Processing summary
    """
    try:
        print(f"=== Blog Images Processing ===")
        print(f"JSON URL: {json_url}")
        print(f"Output directory: {output_dir}")
        print(f"Base URL: {base_url}")
        print(f"Force reprocessing: {force}")
        print(f"Download only: {download_only}")
        print(f"Draw rectangles: {draw_rectangles}")
        print(f"Draw labels: {draw_labels}")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create backup directory for original images
        backup_dir = os.path.join(output_dir, "backup", "blogs")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Download JSON data
        print("Downloading JSON data...")
        response = requests.get(json_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"JSON data loaded successfully")
        
        # Initialize detectors
        if not download_only:
            print("Initializing detectors...")
            nudenet_detector = NudeNetDetector(
                confidence_threshold=0.05,
                pixel_size=15,
                padding=10
            )
            
            # Initialize YOLO model with error handling
            yolo_model = None
            if ENABLE_YOLO_DETECTION:
                try:
                    yolo_model = YOLO("yolo_v8_model/runs/detect/train15/weights/best.pt")
                    print("YOLO model initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize YOLO model: {e}")
                    print("Continuing without YOLO detection...")
                    yolo_model = None
            else:
                print("YOLO detection disabled in configuration")
            
            print("Detectors initialized successfully")
        
        # Process images
        processed_count = 0
        skipped_count = 0
        error_count = 0
        download_count = 0
        
        # Extract image URLs from JSON data
        image_urls = []
        
        # Handle the new blog images JSON structure with WordPress sizes
        if isinstance(data, list):
            # New structure: [{"slug": "blog_slug", "images": ["url1", "url2", ...]}, ...]
            for item in data:
                if isinstance(item, dict) and 'slug' in item and 'images' in item:
                    slug = item['slug']
                    images_data = item['images']
                    
                    if isinstance(images_data, list):
                        # Process each URL in the images array
                        for i, url in enumerate(images_data):
                            if url and isinstance(url, str):
                                # Handle relative URLs with base_url
                                if base_url and not url.startswith(('http://', 'https://')):
                                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                
                                image_urls.append({
                                    'url': url,
                                    'type': 'blog_image',
                                    'slug': slug,
                                    'image_type': f'image_{i+1}',
                                    'size_name': f'image_{i+1}'
                                })
                    elif isinstance(images_data, dict):
                        # Legacy structure: {"thumbnail": "url", "medium": "url", ...}
                        for size_name, url in images_data.items():
                            if url and isinstance(url, str):
                                # Handle relative URLs with base_url
                                if base_url and not url.startswith(('http://', 'https://')):
                                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                
                                image_urls.append({
                                    'url': url,
                                    'type': 'blog_image',
                                    'slug': slug,
                                    'image_type': size_name,
                                    'size_name': size_name
                                })
                    elif isinstance(images_data, list):
                        # New structure: ["url1", "url2", ...]
                        for i, url in enumerate(images_data):
                            if url and isinstance(url, str):
                                if base_url and not url.startswith(('http://', 'https://')):
                                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                
                                image_urls.append({
                                    'url': url,
                                    'type': 'blog_image',
                                    'slug': slug,
                                    'image_type': f'image_{i+1}',
                                    'size_name': f'image_{i+1}'
                                })
        
        elif isinstance(data, dict):
            # Check for nested structures
            if 'data' in data and isinstance(data['data'], list):
                # Structure: {"data": [{"slug": "blog_slug", "images": {"thumbnail": "url", ...}}, ...]}
                for item in data['data']:
                    if isinstance(item, dict) and 'slug' in item and 'images' in item:
                        slug = item['slug']
                        images_data = item['images']
                        
                        if isinstance(images_data, dict):
                            for size_name, url in images_data.items():
                                if url and isinstance(url, str):
                                    if base_url and not url.startswith(('http://', 'https://')):
                                        url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                    
                                    image_urls.append({
                                        'url': url,
                                        'type': 'blog_image',
                                        'slug': slug,
                                        'image_type': size_name,
                                        'size_name': size_name
                                    })
                        elif isinstance(images_data, list):
                            # New structure: ["url1", "url2", ...]
                            for i, url in enumerate(images_data):
                                if url and isinstance(url, str):
                                    if base_url and not url.startswith(('http://', 'https://')):
                                        url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                    
                                    image_urls.append({
                                        'url': url,
                                        'type': 'blog_image',
                                        'slug': slug,
                                        'image_type': f'image_{i+1}',
                                        'size_name': f'image_{i+1}'
                                    })
            
            elif 'blog_images' in data and isinstance(data['blog_images'], list):
                # Structure: {"blog_images": [{"slug": "blog_slug", "images": ["url1", "url2", ...]}, ...]}
                for item in data['blog_images']:
                    if isinstance(item, dict) and 'slug' in item and 'images' in item:
                        slug = item['slug']
                        images_data = item['images']
                        
                        if isinstance(images_data, dict):
                            for size_name, url in images_data.items():
                                if url and isinstance(url, str):
                                    if base_url and not url.startswith(('http://', 'https://')):
                                        url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                    
                                    image_urls.append({
                                        'url': url,
                                        'type': 'blog_image',
                                        'slug': slug,
                                        'image_type': size_name,
                                        'size_name': size_name
                                    })
                        elif isinstance(images_data, list):
                            # New structure: ["url1", "url2", ...]
                            for i, url in enumerate(images_data):
                                if url and isinstance(url, str):
                                    if base_url and not url.startswith(('http://', 'https://')):
                                        url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                    
                                    image_urls.append({
                                        'url': url,
                                        'type': 'blog_image',
                                        'slug': slug,
                                        'image_type': f'image_{i+1}',
                                        'size_name': f'image_{i+1}'
                                    })
        
        print(f"Found {len(image_urls)} blog image URLs to process")
        
        for i, image_data in enumerate(image_urls, 1):
            try:
                print(f"\n[{i}/{len(image_urls)}] Processing: {image_data['url']}")
                print(f"  Blog ID: {image_data['slug']}, Size: {image_data['size_name']}")
                
                # Download image
                downloaded_path = download_image(image_data['url'])
                if not downloaded_path:
                    error_count += 1
                    continue
                
                download_count += 1
                
                # Create backup of original image
                backup_filename = os.path.basename(downloaded_path)
                backup_path = os.path.join(backup_dir, backup_filename)
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(downloaded_path, backup_path)
                    print(f"  📁 Backed up to: {backup_path}")
                
                if download_only:
                    continue
                
                # Determine output path with WordPress structure for blog images
                filename = os.path.basename(downloaded_path)
                
                # Use original filename (preserve case and original name)
                new_filename = filename
                
                # Save in wp-content/uploads/blog-images
                wp_upload_dir = os.path.join('wp-content', 'uploads', 'blog-images')
                output_path = os.path.join(wp_upload_dir, new_filename)
                
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Check if already processed
                if os.path.exists(output_path) and not force:
                    print(f"  Skipped (already exists): {output_path}")
                    skipped_count += 1
                    continue
                
                # Process the image with enhanced detection (no resizing)
                result = process_single_image_enhanced(
                    downloaded_path, 
                    output_path, 
                    nudenet_detector, 
                    yolo_model,
                    image_data['image_type'],
                    force,
                    draw_rectangles,
                    draw_labels
                )
                
                if result['success']:
                    processed_count += 1
                    print(f"  Success: {output_path}")
                    print(f"    NudeNet detections: {result['nudenet_detections']}")
                    print(f"    YOLO detections: {result['yolo_detections']}")
                    print(f"    WordPress files: {len(result['wordpress_files'])}")
                else:
                    error_count += 1
                    print(f"  Failed: {result['message']}")
                
            except Exception as e:
                error_count += 1
                print(f"  Error processing {image_data['url']}: {e}")
                continue
        
        # Summary
        print(f"\n=== Blog Images Processing Summary ===")
        print(f"Total blog images: {len(image_urls)}")
        print(f"Downloaded: {download_count}")
        if not download_only:
            print(f"Processed: {processed_count}")
            print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        
        # Show database statistics
        if not download_only:
            db_tracker.get_processing_stats()
        
        return {
            'success': True,
            'total_images': len(image_urls),
            'downloaded': download_count,
            'processed': processed_count if not download_only else 0,
            'skipped': skipped_count if not download_only else 0,
            'errors': error_count
        }
        
    except Exception as e:
        print(f"Error in blog_images: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def coupon_images(json_url, output_dir="processed_images", base_url=None, force=False, download_only=False, draw_rectangles=False, draw_labels=False):
    """
    Process coupon images from a JSON URL using enhanced detection.
    
    Args:
        json_url (str): URL to JSON file containing coupon image data
        output_dir (str): Directory to save processed images
        base_url (str): Base URL for converting relative paths to absolute URLs
        force (bool): Force reprocessing even if output already exists
        download_only (bool): Only download images, do not process them
        draw_rectangles (bool): Whether to draw rectangle borders for debugging
        draw_labels (bool): Whether to draw labels on rectangles for debugging
        
    Returns:
        dict: Processing summary
    """
    try:
        print(f"=== Coupon Images Processing ===")
        print(f"JSON URL: {json_url}")
        print(f"Output directory: {output_dir}")
        print(f"Base URL: {base_url}")
        print(f"Force reprocessing: {force}")
        print(f"Download only: {download_only}")
        print(f"Draw rectangles: {draw_rectangles}")
        print(f"Draw labels: {draw_labels}")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create backup directory for original images
        backup_dir = os.path.join(output_dir, "backup", "coupons")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Download JSON data
        print("Downloading JSON data...")
        response = requests.get(json_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"JSON data loaded successfully")
        
        # Initialize detectors
        if not download_only:
            print("Initializing detectors...")
            nudenet_detector = NudeNetDetector(
                confidence_threshold=0.05,
                pixel_size=15,
                padding=10
            )
            
            # Initialize YOLO model with error handling
            yolo_model = None
            if ENABLE_YOLO_DETECTION:
                try:
                    yolo_model = YOLO("yolo_v8_model/runs/detect/train15/weights/best.pt")
                    print("YOLO model initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize YOLO model: {e}")
                    print("Continuing without YOLO detection...")
                    yolo_model = None
            else:
                print("YOLO detection disabled in configuration")
            
            print("Detectors initialized successfully")
        
        # Process images
        processed_count = 0
        skipped_count = 0
        error_count = 0
        download_count = 0
        
        # Extract image URLs from JSON data
        image_urls = []
        
        # Handle the coupon images JSON structure with WordPress sizes
        if isinstance(data, list):
            # New structure: [{"slug": "coupon_slug", "images": ["url1", "url2", ...]}, ...]
            for item in data:
                if isinstance(item, dict) and 'slug' in item and 'images' in item:
                    slug = item['slug']
                    images_data = item['images']
                    
                    if isinstance(images_data, list):
                        # Process each URL in the images array
                        for i, url in enumerate(images_data):
                            if url and isinstance(url, str):
                                # Handle relative URLs with base_url
                                if base_url and not url.startswith(('http://', 'https://')):
                                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                
                                image_urls.append({
                                    'url': url,
                                    'type': 'coupon_image',
                                    'slug': slug,
                                    'image_type': f'image_{i+1}',
                                    'size_name': f'image_{i+1}'
                                })
                    elif isinstance(images_data, dict):
                        # Legacy structure: {"thumbnail": "url", "medium": "url", ...}
                        for size_name, url in images_data.items():
                            if url and isinstance(url, str):
                                # Handle relative URLs with base_url
                                if base_url and not url.startswith(('http://', 'https://')):
                                    url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                
                                image_urls.append({
                                    'url': url,
                                    'type': 'coupon_image',
                                    'slug': slug,
                                    'image_type': size_name,
                                    'size_name': size_name
                                })
        
        elif isinstance(data, dict):
            # Check for nested structures
            if 'data' in data and isinstance(data['data'], list):
                # Structure: {"data": [{"slug": "coupon_slug", "images": ["url1", "url2", ...]}, ...]}
                for item in data['data']:
                    if isinstance(item, dict) and 'slug' in item and 'images' in item:
                        slug = item['slug']
                        images_data = item['images']
                        
                        if isinstance(images_data, list):
                            # Process each URL in the images array
                            for i, url in enumerate(images_data):
                                if url and isinstance(url, str):
                                    if base_url and not url.startswith(('http://', 'https://')):
                                        url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                    
                                    image_urls.append({
                                        'url': url,
                                        'type': 'coupon_image',
                                        'slug': slug,
                                        'image_type': f'image_{i+1}',
                                        'size_name': f'image_{i+1}'
                                    })
                        elif isinstance(images_data, dict):
                            # Legacy structure: {"thumbnail": "url", "medium": "url", ...}
                            for size_name, url in images_data.items():
                                if url and isinstance(url, str):
                                    if base_url and not url.startswith(('http://', 'https://')):
                                        url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                    
                                    image_urls.append({
                                        'url': url,
                                        'type': 'coupon_image',
                                        'slug': slug,
                                        'image_type': size_name,
                                        'size_name': size_name
                                    })
            
            elif 'coupon_images' in data and isinstance(data['coupon_images'], list):
                # Structure: {"coupon_images": [{"slug": "coupon_slug", "images": ["url1", "url2", ...]}, ...]}
                for item in data['coupon_images']:
                    if isinstance(item, dict) and 'slug' in item and 'images' in item:
                        slug = item['slug']
                        images_data = item['images']
                        
                        if isinstance(images_data, list):
                            # Process each URL in the images array
                            for i, url in enumerate(images_data):
                                if url and isinstance(url, str):
                                    if base_url and not url.startswith(('http://', 'https://')):
                                        url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                    
                                    image_urls.append({
                                        'url': url,
                                        'type': 'coupon_image',
                                        'slug': slug,
                                        'image_type': f'image_{i+1}',
                                        'size_name': f'image_{i+1}'
                                    })
                        elif isinstance(images_data, dict):
                            # Legacy structure: {"thumbnail": "url", "medium": "url", ...}
                            for size_name, url in images_data.items():
                                if url and isinstance(url, str):
                                    if base_url and not url.startswith(('http://', 'https://')):
                                        url = base_url.rstrip('/') + '/' + url.lstrip('/')
                                    
                                    image_urls.append({
                                        'url': url,
                                        'type': 'coupon_image',
                                        'slug': slug,
                                        'image_type': size_name,
                                        'size_name': size_name
                                    })
        
        print(f"Found {len(image_urls)} coupon image URLs to process")
        
        for i, image_data in enumerate(image_urls, 1):
            try:
                print(f"\n[{i}/{len(image_urls)}] Processing: {image_data['url']}")
                print(f"  Coupon ID: {image_data['slug']}, Size: {image_data['size_name']}")
                
                # Download image
                downloaded_path = download_image(image_data['url'])
                if not downloaded_path:
                    error_count += 1
                    continue
                
                download_count += 1
                
                # Create backup of original image
                backup_filename = os.path.basename(downloaded_path)
                backup_path = os.path.join(backup_dir, backup_filename)
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(downloaded_path, backup_path)
                    print(f"  📁 Backed up to: {backup_path}")
                
                if download_only:
                    continue
                
                # Determine output path with WordPress structure for coupon images
                filename = os.path.basename(downloaded_path)
                
                # Use original filename (preserve case and original name)
                new_filename = filename
                
                # Save in wp-content/uploads/coupons
                wp_upload_dir = os.path.join('wp-content', 'uploads', 'coupons')
                output_path = os.path.join(wp_upload_dir, new_filename)
                
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Check if already processed
                if os.path.exists(output_path) and not force:
                    print(f"  Skipped (already exists): {output_path}")
                    skipped_count += 1
                    continue
                
                # Process the image with enhanced detection (no resizing)
                result = process_single_image_enhanced(
                    downloaded_path, 
                    output_path, 
                    nudenet_detector, 
                    yolo_model,
                    image_data['image_type'],
                    force,
                    draw_rectangles,
                    draw_labels
                )
                
                if result['success']:
                    processed_count += 1
                    print(f"  Success: {output_path}")
                    print(f"    NudeNet detections: {result['nudenet_detections']}")
                    print(f"    YOLO detections: {result['yolo_detections']}")
                    print(f"    WordPress files: {len(result['wordpress_files'])}")
                else:
                    error_count += 1
                    print(f"  Failed: {result['message']}")
                
            except Exception as e:
                error_count += 1
                print(f"  Error processing {image_data['url']}: {e}")
                continue
        
        # Summary
        print(f"\n=== Coupon Images Processing Summary ===")
        print(f"Total coupon images: {len(image_urls)}")
        print(f"Downloaded: {download_count}")
        if not download_only:
            print(f"Processed: {processed_count}")
            print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        
        # Show database statistics
        if not download_only:
            db_tracker.get_processing_stats()
        
        return {
            'success': True,
            'total_images': len(image_urls),
            'downloaded': download_count,
            'processed': processed_count if not download_only else 0,
            'skipped': skipped_count if not download_only else 0,
            'errors': error_count
        }
        
    except Exception as e:
        print(f"Error in coupon_images: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def single(image_path, output_dir="processed_images", image_type=None, force=False, draw_rectangles=False, draw_labels=False):
    """
    Process a single image (file path or URL) and save in uploads folder.
    
    Args:
        image_path (str): Path to input image file or URL
        output_dir (str): Directory to save processed images
        image_type (str): Type of image (screenshot_full_url, review_full_image, category_thumb, etc.)
        force (bool): Force reprocessing even if output already exists
        draw_rectangles (bool): Whether to draw rectangle borders for debugging
        draw_labels (bool): Whether to draw labels on rectangles for debugging
        
    Returns:
        dict: Processing summary
    """
    try:
        print(f"=== Single Image Processing ===")
        print(f"Input image: {image_path}")
        print(f"Output directory: {output_dir}")
        print(f"Image type: {image_type}")
        print(f"Force reprocessing: {force}")
        print(f"Draw rectangles: {draw_rectangles}")
        print(f"Draw labels: {draw_labels}")
        print()
        
        # Check if input is a URL or file path
        is_url = image_path.startswith(('http://', 'https://'))
        
        if is_url:
            print(f"  Detected URL: {image_path}")
            # Download the image
            downloaded_path = download_image(image_path)
            if not downloaded_path:
                return {
                    'success': False,
                    'message': f"Failed to download image from URL: {image_path}"
                }
            print(f"  Downloaded to: {downloaded_path}")
            local_image_path = downloaded_path
        else:
            # Check if local file exists
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'message': f"Input file not found: {image_path}"
                }
            local_image_path = image_path
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create backup directory for original images
        backup_dir = os.path.join(output_dir, "backup", "single")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Initialize detectors
        print("Initializing detectors...")
        nudenet_detector = NudeNetDetector(
            confidence_threshold=0.05,
            pixel_size=15,
            padding=10
        )
        
        # Initialize YOLO model with error handling
        yolo_model = None
        if ENABLE_YOLO_DETECTION:
            try:
                yolo_model = YOLO("yolo_v8_model/runs/detect/train15/weights/best.pt")
                print("YOLO model initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize YOLO model: {e}")
                print("Continuing without YOLO detection...")
                yolo_model = None
        else:
            print("YOLO detection disabled in configuration")
        
        print("Detectors initialized successfully")
        
        # Determine image type if not provided
        if image_type is None:
            # Try to infer from filename or path
            filename = os.path.basename(local_image_path).lower()
            if 'screenshot' in filename or 'screen' in filename:
                image_type = 'screenshot_full_url'
            elif 'review' in filename:
                image_type = 'review_full_image'
            elif 'thumb' in filename or 'category' in filename:
                image_type = 'category_thumb'
            else:
                image_type = 'screenshot_full_url'  # Default
        
        print(f"  Detected image type: {image_type}")
        
        # Create backup of original image
        backup_filename = os.path.basename(local_image_path)
        backup_path = os.path.join(backup_dir, backup_filename)
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(local_image_path, backup_path)
            print(f"  📁 Backed up to: {backup_path}")
        
        # Determine output path with uploads folder structure (same as sliding_json)
        filename = os.path.basename(local_image_path)
        
        if image_type == 'review_full_image':
            # Save in wp-content/uploads/screenshots
            wp_upload_dir = os.path.join('wp-content', 'uploads', 'screenshots')
            output_path = os.path.join(wp_upload_dir, filename)
            print(f"  Screenshots Output directory: {wp_upload_dir}")
        else:
            # Save in wp-content/uploads
            wp_upload_dir = os.path.join('wp-content', 'uploads')
            output_path = os.path.join(wp_upload_dir, filename)
        
        print(f"  Output directory: {wp_upload_dir}")
        print(f"  Output path: {output_path}")
        
        # Create output directory structure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Also create the screenshots folder for review_full_image type
        if image_type == 'review_full_image':
            screenshots_dir = os.path.join('wp-content', 'uploads', 'screenshots')
            os.makedirs(screenshots_dir, exist_ok=True)
            print(f"  Created screenshots directory: {screenshots_dir}")
        
        # Check if already processed
        if os.path.exists(output_path) and not force:
            print(f"  Skipped (already exists): {output_path}")
            return {
                'success': True,
                'total_images': 1,
                'downloaded': 1 if is_url else 0,
                'processed': 0,
                'skipped': 1,
                'errors': 0
            }
        
        # Process the image
        print(f"Processing image: {filename}")
        result = process_single_image(
            local_image_path, 
            output_path, 
            nudenet_detector, 
            yolo_model,
            image_type,
            force,
            draw_rectangles,
            draw_labels
        )
        
        if result['success']:
            print(f"  Success: {output_path}")
            print(f"    NudeNet detections: {result['nudenet_detections']}")
            print(f"    YOLO detections: {result['yolo_detections']}")
            print(f"    WordPress files: {len(result['wordpress_files'])}")
            
            # Summary
            print(f"\n=== Processing Summary ===")
            print(f"Total images: 1")
            print(f"Downloaded: {1 if is_url else 0}")
            print(f"Processed: 1")
            print(f"Errors: 0")
            
            # Show database statistics
            db_tracker.get_processing_stats()
            
            return {
                'success': True,
                'total_images': 1,
                'downloaded': 1 if is_url else 0,
                'processed': 1,
                'skipped': 0,
                'errors': 0
            }
        else:
            print(f"  Failed: {result['message']}")
            return {
                'success': False,
                'message': result['message']
            }
        
    except Exception as e:
        print(f"Error in single: {e}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }

def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(description='Enhanced Image Processing with NudeNet and YOLO')
    parser.add_argument('command', choices=['sliding-json', 'category-thumbnails', 'sliding-single', 'blog-images', 'coupon-images', 'single'], help='Command to execute')
    parser.add_argument('--json-url', help='URL to JSON file containing image data (for sliding-json, category-thumbnails, and blog-images)')
    parser.add_argument('--image-path', help='Path to input image file or URL (for sliding-single)')
    parser.add_argument('--image-type', help='Type of image (screenshot_full_url, review_full_image, category_thumb, etc.)')
    parser.add_argument('--output-dir', default='processed_images', help='Output directory for processed images')
    parser.add_argument('--base-url', help='Base URL for converting relative paths to absolute URLs')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if output already exists')
    parser.add_argument('--download-only', action='store_true', help='Only download images, do not process them')
    parser.add_argument('--disable-yolo', action='store_true', help='Disable YOLO detection (use only NudeNet)')
    parser.add_argument('--draw-rectangles', action='store_true', help='Draw rectangles around detected regions for debugging')
    parser.add_argument('--draw-labels', action='store_true', help='Draw labels on rectangles (requires --draw-rectangles)')
    
    args = parser.parse_args()
    
    # Update global YOLO configuration based on command line argument
    global ENABLE_YOLO_DETECTION
    if args.disable_yolo:
        ENABLE_YOLO_DETECTION = False
        print("⚠️ YOLO detection disabled via command line argument")
    
    if args.command == 'sliding-json':
        if not args.json_url:
            print("❌ --json-url is required for sliding-json command")
            return 1
            
        result = sliding_json(
            json_url=args.json_url,
            output_dir=args.output_dir,
            base_url=args.base_url,
            force=args.force,
            download_only=args.download_only,
            draw_rectangles=args.draw_rectangles,
            draw_labels=args.draw_labels
        )
        
        if not result['success']:
            print(f"❌ {result['message']}")
            return 1
        
        print("✅ Processing completed successfully!")
        return 0
        
    elif args.command == 'category-thumbnails':
        if not args.json_url:
            print("❌ --json-url is required for category-thumbnails command")
            return 1
            
        result = category_thumbnails(
            json_url=args.json_url,
            output_dir=args.output_dir,
            base_url=args.base_url,
            force=args.force,
            download_only=args.download_only,
            draw_rectangles=args.draw_rectangles,
            draw_labels=args.draw_labels
        )
        
        if not result['success']:
            print(f"❌ {result['message']}")
            return 1
        
        print("✅ Processing completed successfully!")
        return 0
        
    elif args.command == 'sliding-single':
        if not args.image_path:
            print("❌ --image-path is required for sliding-single command")
            return 1
            
        result = sliding_single(
            image_path=args.image_path,
            output_dir=args.output_dir,
            image_type=args.image_type,
            force=args.force,
            draw_rectangles=args.draw_rectangles,
            draw_labels=args.draw_labels
        )
        
        if not result['success']:
            print(f"❌ {result['message']}")
            return 1
        
        print("✅ Processing completed successfully!")
        return 0
        
    elif args.command == 'blog-images':
        if not args.json_url:
            print("❌ --json-url is required for blog-images command")
            return 1
            
        result = blog_images(
            json_url=args.json_url,
            output_dir=args.output_dir,
            base_url=args.base_url,
            force=args.force,
            download_only=args.download_only,
            draw_rectangles=args.draw_rectangles,
            draw_labels=args.draw_labels
        )
        
        if not result['success']:
            print(f"❌ {result['error']}")
            return 1
        
        print("✅ Processing completed successfully!")
        return 0
        
    elif args.command == 'coupon-images':
        if not args.json_url:
            print("❌ --json-url is required for coupon-images command")
            return 1
            
        result = coupon_images(
            json_url=args.json_url,
            output_dir=args.output_dir,
            base_url=args.base_url,
            force=args.force,
            download_only=args.download_only,
            draw_rectangles=args.draw_rectangles,
            draw_labels=args.draw_labels
        )
        
        if not result['success']:
            print(f"❌ {result['error']}")
            return 1
        
        print("✅ Processing completed successfully!")
        return 0
        
    elif args.command == 'single':
        if not args.image_path:
            print("❌ --image-path is required for single command")
            return 1
            
        result = single(
            image_path=args.image_path,
            output_dir=args.output_dir,
            image_type=args.image_type,
            force=args.force,
            draw_rectangles=args.draw_rectangles,
            draw_labels=args.draw_labels
        )
        
        if not result['success']:
            print(f"❌ {result['message']}")
            return 1
        
        print("✅ Processing completed successfully!")
        return 0

if __name__ == "__main__":
    exit(main()) 