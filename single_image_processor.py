#!/usr/bin/env python3
"""
Standalone Single Image Processor
This script processes a single image (URL or file path) with NudeNet and YOLO detection.
Saves processed images in wp-content/uploads folder.
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
import numpy as np

# Try to import required modules with error handling
try:
    from nudenet_detector import NudeNetDetector
    NUDENET_AVAILABLE = True
except ImportError:
    print("Warning: NudeNet detector not available. Install nudenet_detector module.")
    NUDENET_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: YOLO not available. Install ultralytics module.")
    YOLO_AVAILABLE = False

# Configuration
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

# Available NudeNet labels for reference
AVAILABLE_NUDENET_LABELS = [
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
]

def show_available_labels():
    """Display all available NudeNet labels."""
    print("Available NudeNet labels:")
    print("=" * 50)
    for i, label in enumerate(AVAILABLE_NUDENET_LABELS, 1):
        print(f"{i:2d}. {label}")
    print("=" * 50)
    print("Usage: --custom-labels FEMALE_BREAST_EXPOSED BUTTOCKS_EXPOSED")

def extract_folder_from_url(url):
    """
    Extract the folder path from a WordPress URL.
    
    Args:
        url (str): WordPress URL (e.g., https://www.mrporngeek.com/wp-content/uploads/screenshots/image.jpg)
        
    Returns:
        str: Folder path (e.g., 'screenshots', 'images', etc.) or 'screenshots' as default
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path
        
        # Look for wp-content/uploads/ in the path
        if '/wp-content/uploads/' in path:
            # Extract the part after wp-content/uploads/
            parts = path.split('/wp-content/uploads/')
            if len(parts) > 1:
                folder_part = parts[1]
                # Get the folder name (first part before the filename)
                folder_parts = folder_part.split('/')
                if len(folder_parts) > 1:
                    return folder_parts[0]  # Return the folder name
        
        # Default fallback
        return 'screenshots'
    except Exception as e:
        print(f"Warning: Could not extract folder from URL {url}: {e}")
        return 'screenshots'

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

def process_single_image_enhanced(input_path, output_path, nudenet_detector, yolo_model, image_type=None, force=False, draw_rectangles=False, draw_labels=False, disable_sliding=False, save_to_folder=True, extracted_folder=None, base_folder=None, disable_resize=False):
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
        disable_sliding (bool): Whether to disable sliding window method
        save_to_folder (bool): If True, save to wp-content/uploads/screenshots/, if False save to wp-content/uploads/blur/
        extracted_folder (str): Folder extracted from URL (e.g., 'screenshots', 'images')
        base_folder (str): Base folder to prepend to output path
        disable_resize (bool): Disable WordPress image resizing (only save the main processed image)
        
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
            use_sliding_window=not disable_sliding,  # Disable sliding if requested
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
        
        if not ENABLE_YOLO_DETECTION or yolo_model is None:
            print("    YOLO detection disabled or not available")
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
        
        # Create WordPress versions if image_type is specified and resize is not disabled
        wordpress_files = []
        if image_type and image_type != 'category_thumb' and not disable_resize:
            base_filename = os.path.splitext(os.path.basename(output_path))[0]
            # Extract folder from URL if it's a URL and save_to_folder is True
            extracted_folder = None
            if save_to_folder and extracted_folder is None:
                # Try to extract from the original image_path if it's a URL
                if input_path.startswith(('http://', 'https://')):
                    extracted_folder = extract_folder_from_url(input_path)
            
            wordpress_files = create_wordpress_versions(
                input_path, 
                output_path, 
                base_filename, 
                image_type,
                save_to_folder,
                extracted_folder,
                base_folder
            )
            print(f"  Created {len(wordpress_files)} WordPress-sized images")
        elif disable_resize:
            print(f"  Skipped WordPress resizing (disabled)")
        
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

def create_wordpress_versions(original_image_path, processed_image_path, base_filename, image_type=None, save_to_folder=True, extracted_folder=None, base_folder=None):
    """
    Create WordPress-sized images from the processed image based on image type.
    
    Args:
        original_image_path (str): Path to original image
        processed_image_path (str): Path to processed image
        base_filename (str): Base filename without extension
        image_type (str): Type of image ('review_full_image', 'screenshot_full_url', etc.)
        save_to_folder (bool): If True, save to wp-content/uploads/screenshots/, if False save to wp-content/uploads/blur/
        extracted_folder (str): Folder extracted from URL (e.g., 'screenshots', 'images')
        base_folder (str): Base folder to prepend to output path
        
    Returns:
        List of created file paths
    """
    created_files = []
    
    # WordPress sizes configuration
    WORDPRESS_SIZES = {
        'blog-tn': (170, 145, False),      # 510x315, cropped
        'category-thumb': (250, 212, True),  # 250x212, cropped
        'swiper-desktop': (590, 504, False)  # 590x504, not cropped
    }
    
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
        
        # Determine output directory based on save_to_folder parameter and extracted folder
        if save_to_folder:
            # Use extracted folder if provided, otherwise default to screenshots
            if extracted_folder:
                wp_upload_dir = os.path.join('wp-content', 'uploads', extracted_folder)
            else:
                wp_upload_dir = os.path.join('wp-content', 'uploads', 'screenshots')
        else:
            # Save in wp-content/uploads/blur
            wp_upload_dir = os.path.join('wp-content', 'uploads', 'blur')
        
        # Prepend base folder if provided
        if base_folder:
            wp_upload_dir = os.path.join(base_folder, wp_upload_dir)
        
        # Create output directory
        os.makedirs(wp_upload_dir, exist_ok=True)
        
        # Save resized image with correct format
        output_path = os.path.join(wp_upload_dir, filename)
        resized_image.save(output_path, save_format, **save_kwargs)
        created_files.append(output_path)
        print(f"  Created {size_name} size: {filename}")
    
    return created_files

def single_image_processor(image_path, output_dir="processed_images", image_type=None, force=False, draw_rectangles=False, draw_labels=False, disable_yolo=False, disable_sliding=False, disable_label_filter=False, custom_labels=None, save_to_folder=True, base_folder=None, disable_resize=False):
    """
    Process a single image (URL or file path) using enhanced detection.
    
    Args:
        image_path (str): Path to input image file or URL
        output_dir (str): Directory to save processed images
        image_type (str): Type of image (screenshot_full_url, review_full_image, category_thumb, etc.)
        force (bool): Force reprocessing even if output already exists
        draw_rectangles (bool): Whether to draw rectangle borders for debugging
        draw_labels (bool): Whether to draw labels on rectangles for debugging
        disable_yolo (bool): Disable YOLO detection for this command
        disable_sliding (bool): Disable sliding window method for this command
        disable_label_filter (bool): Disable label type filtering for this command
        custom_labels (list): List of custom labels to filter for (e.g., ['FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED'])
        save_to_folder (bool): If True, save to wp-content/uploads/screenshots/, if False save to wp-content/uploads/blur/
        base_folder (str): Base folder to prepend to output path (e.g., '/home/httpd/html/mrporngeek.com/public_html')
        disable_resize (bool): Disable WordPress image resizing (only save the main processed image)
        
    Returns:
        dict: Processing summary
    """
    try:
        print(f"=== Single Image Processing (Enhanced) ===")
        print(f"Input image: {image_path}")
        print(f"Output directory: {output_dir}")
        print(f"Image type: {image_type}")
        print(f"Save to folder: {save_to_folder}")
        print(f"Base folder: {base_folder}")
        print(f"Disable resize: {disable_resize}")
        print(f"Force reprocessing: {force}")
        print(f"Draw rectangles: {draw_rectangles}")
        print(f"Draw labels: {draw_labels}")
        print(f"Disable sliding: {disable_sliding}")
        print(f"Disable label filter: {disable_label_filter}")
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
        backup_dir = os.path.join(output_dir, "backup", "single-image")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Initialize detectors
        print("Initializing enhanced detectors...")
        
        if not NUDENET_AVAILABLE:
            return {
                'success': False,
                'message': "NudeNet detector not available. Please install the required module."
            }
        
        # Handle custom labels
        if custom_labels:
            print(f"  Using custom labels: {custom_labels}")
            
            # Validate custom labels
            invalid_labels = [label for label in custom_labels if label not in AVAILABLE_NUDENET_LABELS]
            if invalid_labels:
                print(f"  Warning: Invalid labels provided: {invalid_labels}")
                print(f"  Valid labels: {AVAILABLE_NUDENET_LABELS}")
                return {
                    'success': False,
                    'message': f"Invalid labels: {invalid_labels}. Use --show-labels to see available options."
                }
            
            # Create a custom NudeNetDetector with custom labels
            nudenet_detector = NudeNetDetector(
                confidence_threshold=0.3,
                pixel_size=15,
                padding=10,
                disable_label_filter=False  # We'll set custom labels manually
            )
            # Override the allowed labels with custom ones
            nudenet_detector.allowed_labels = set(custom_labels)
        else:
            nudenet_detector = NudeNetDetector(
                confidence_threshold=0.3,
                pixel_size=15,
                padding=10,
                disable_label_filter=disable_label_filter
            )
        
        # Initialize YOLO model with error handling
        yolo_model = None
        if ENABLE_YOLO_DETECTION and not disable_yolo and YOLO_AVAILABLE:
            try:
                yolo_model = YOLO("yolo_v8_model/runs/detect/train15/weights/best.pt")
                print("YOLO model initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize YOLO model: {e}")
                print("Continuing without YOLO detection...")
                yolo_model = None
        else:
            if disable_yolo:
                print("YOLO detection disabled for single-image command")
            elif not YOLO_AVAILABLE:
                print("YOLO not available - continuing without YOLO detection")
            else:
                print("YOLO detection disabled in configuration")
        
        print("Enhanced detectors initialized successfully")
        
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
        
        # Determine output path based on save_to_folder parameter and URL structure
        filename = os.path.basename(local_image_path)
        
        # Use original filename (preserve case and original name)
        new_filename = filename
        
        # Determine output directory based on save_to_folder parameter and URL
        if save_to_folder:
            # Extract folder from URL if it's a URL, otherwise use default
            if is_url:
                extracted_folder = extract_folder_from_url(image_path)
                wp_upload_dir = os.path.join('wp-content', 'uploads', extracted_folder)
                print(f"  Extracted folder from URL: {extracted_folder}")
            else:
                # For local files, use default screenshots folder
                wp_upload_dir = os.path.join('wp-content', 'uploads', 'screenshots')
        else:
            # Save in wp-content/uploads/blur
            wp_upload_dir = os.path.join('wp-content', 'uploads', 'blur')
        
        # Prepend base folder if provided
        if base_folder:
            wp_upload_dir = os.path.join(base_folder, wp_upload_dir)
            print(f"  Using base folder: {base_folder}")
        
        output_path = os.path.join(wp_upload_dir, new_filename)
        
        print(f"  Output directory: {wp_upload_dir}")
        print(f"  Output path: {output_path}")
        
        # Create output directory structure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
        
        # Process the image with enhanced detection (no resizing)
        print(f"Processing image: {filename}")
        # Extract folder from URL if it's a URL and save_to_folder is True
        extracted_folder = None
        if save_to_folder and is_url:
            extracted_folder = extract_folder_from_url(image_path)
        
        result = process_single_image_enhanced(
            local_image_path, 
            output_path, 
            nudenet_detector, 
            yolo_model,
            image_type,
            force,
            draw_rectangles,
            draw_labels,
            disable_sliding,
            save_to_folder,
            extracted_folder,
            base_folder,
            disable_resize
        )
        
        if result['success']:
            print(f"  Success: {output_path}")
            print(f"    NudeNet detections: {result['nudenet_detections']}")
            print(f"    YOLO detections: {result['yolo_detections']}")
            print(f"    WordPress files: {len(result['wordpress_files'])}")
            
            # Summary
            print(f"\n=== Single Image Processing Summary ===")
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
        print(f"Error in single_image_processor: {e}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }

def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(description='Standalone Single Image Processor with NudeNet and YOLO')
    parser.add_argument('--image-path', required=False, help='Path to input image file or URL')
    parser.add_argument('--image-type', help='Type of image (screenshot_full_url, review_full_image, category_thumb, etc.)')
    parser.add_argument('--output-dir', default='processed_images', help='Output directory for processed images')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if output already exists')
    parser.add_argument('--disable-yolo', action='store_true', help='Disable YOLO detection (use only NudeNet)')
    parser.add_argument('--disable-sliding', action='store_true', help='Disable sliding window method (use only full image detection)')
    parser.add_argument('--disable-label-filter', action='store_true', help='Disable label type filtering (process all detected labels)')
    parser.add_argument('--draw-rectangles', action='store_true', help='Draw rectangles around detected regions for debugging')
    parser.add_argument('--draw-labels', action='store_true', help='Draw labels on rectangles (requires --draw-rectangles)')
    parser.add_argument('--custom-labels', nargs='+', help='Custom labels to filter for (e.g., FEMALE_BREAST_EXPOSED BUTTOCKS_EXPOSED)')
    parser.add_argument('--show-labels', action='store_true', help='Show all available NudeNet labels and exit')
    parser.add_argument('--save-to-folder', action='store_true', help='Save to wp-content/uploads/screenshots/ (default)')
    parser.add_argument('--save-to-blur', action='store_true', help='Save to wp-content/uploads/blur/ (overrides --save-to-folder)')
    parser.add_argument('--base-folder', help='Base folder to prepend to output path (e.g., /home/httpd/html/mrporngeek.com/public_html)')
    parser.add_argument('--disable-resize', action='store_true', help='Disable WordPress image resizing (only save the main processed image)')
    
    args = parser.parse_args()
    
    # Show available labels if requested
    if args.show_labels:
        show_available_labels()
        return 0
    
    # Check if image-path is provided (required unless showing labels)
    if not args.image_path:
        parser.error("--image-path is required unless using --show-labels")
    
    # Update global YOLO configuration based on command line argument
    global ENABLE_YOLO_DETECTION
    if args.disable_yolo:
        ENABLE_YOLO_DETECTION = False
        print("⚠️ YOLO detection disabled via command line argument")
    
    # Determine save_to_folder based on command line arguments
    save_to_folder = True  # Default
    if args.save_to_blur:
        save_to_folder = False
        print("📁 Will save to wp-content/uploads/blur/")
    else:
        # Extract folder from URL if it's a URL
        if args.image_path and args.image_path.startswith(('http://', 'https://')):
            extracted_folder = extract_folder_from_url(args.image_path)
            print(f"📁 Will save to wp-content/uploads/{extracted_folder}/")
        else:
            print("📁 Will save to wp-content/uploads/screenshots/")
    
    result = single_image_processor(
        image_path=args.image_path,
        output_dir=args.output_dir,
        image_type=args.image_type,
        force=args.force,
        draw_rectangles=args.draw_rectangles,
        draw_labels=args.draw_labels,
        disable_yolo=args.disable_yolo,
        disable_sliding=args.disable_sliding,
        disable_label_filter=args.disable_label_filter,
        custom_labels=args.custom_labels,
        save_to_folder=save_to_folder,
        base_folder=args.base_folder,
        disable_resize=args.disable_resize
    )
    
    if not result['success']:
        print(f"❌ {result['message']}")
        return 1
    
    print("✅ Processing completed successfully!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 