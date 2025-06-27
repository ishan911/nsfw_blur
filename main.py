#!/usr/bin/env python3
"""
Enhanced Image Processing Script with NudeNet and YOLO Detection
This script combines NudeNet detection with YOLO detection for comprehensive content filtering.
Includes WordPress sizing and folder structure support.
"""

import os
import sys
import argparse
import json
import requests
from pathlib import Path
import cv2
from PIL import Image
from nudenet_detector import NudeNetDetector
from ultralytics import YOLO

# WordPress image sizes configuration
WORDPRESS_SIZES = {
    'blog-tn': (170, 145, False),      # 510x315, cropped
    'category-thumb': (250, 212, True),  # 250x212, cropped
    'swiper-desktop': (590, 504, False)  # 590x504, not cropped
}

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

def create_wordpress_sizes(original_image_path, processed_image_path, base_filename, image_type=None):
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

def process_single_image(input_path, output_path, nudenet_detector, yolo_model, image_type=None):
    """
    Process a single image with both NudeNet and YOLO detection.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to output image
        nudenet_detector (NudeNetDetector): NudeNet detector instance
        yolo_model (YOLO): YOLO model instance
        image_type (str): Type of image for WordPress sizing
        
    Returns:
        dict: Processing results
    """
    try:
        print(f"Processing: {input_path}")
        
        # Step 1: NudeNet detection and pixelation
        print("  Running NudeNet detection...")
        nudenet_result = nudenet_detector.process_image(
            input_path=input_path,
            output_path=output_path,
            use_sliding_window=True,
            draw_rectangles=False,
            draw_labels=False
        )
        
        if not nudenet_result['success']:
            print(f"  NudeNet processing failed: {nudenet_result['message']}")
            return nudenet_result
        
        print(f"  NudeNet detections: {nudenet_result['detection_count']}")
        
        # Step 2: YOLO detection and blurring
        print("  Running YOLO detection...")
        yolo_results = yolo_model(output_path)[0]
        
        if len(yolo_results.boxes) > 0:
            print(f"  YOLO detections: {len(yolo_results.boxes)}")
            
            # Apply Gaussian blur to YOLO detections
            img = cv2.imread(output_path)
            for box in yolo_results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                roi = img[y1:y2, x1:x2]
                roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
                img[y1:y2, x1:x2] = roi_blur
            
            # Save the final result
            cv2.imwrite(output_path, img)
            print(f"  Applied YOLO blurring to {len(yolo_results.boxes)} regions")
        else:
            print("  No YOLO detections found")
        
        # Step 3: Create WordPress sizes
        print("  Creating WordPress sizes...")
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        created_files = create_wordpress_sizes(input_path, output_path, base_filename, image_type)
        print(f"  Created {len(created_files)} WordPress-sized images")
        
        return {
            'success': True,
            'nudenet_detections': nudenet_result['detection_count'],
            'yolo_detections': len(yolo_results.boxes) if len(yolo_results.boxes) > 0 else 0,
            'total_detections': nudenet_result['detection_count'] + (len(yolo_results.boxes) if len(yolo_results.boxes) > 0 else 0),
            'wordpress_files': created_files,
            'message': f"Processed successfully - NudeNet: {nudenet_result['detection_count']}, YOLO: {len(yolo_results.boxes) if len(yolo_results.boxes) > 0 else 0}"
        }
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }

def sliding_json(json_url, output_dir="processed_images", base_url=None, force=False, download_only=False):
    """
    Process images from a JSON URL using sliding window detection.
    
    Args:
        json_url (str): URL to JSON file containing image data
        output_dir (str): Directory to save processed images
        base_url (str): Base URL for converting relative paths to absolute URLs
        force (bool): Force reprocessing even if output already exists
        download_only (bool): Only download images, do not process them
        
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
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
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
                padding=5
            )
            
            yolo_model = YOLO("yolo_v8_model/runs/detect/train15/weights/best.pt")
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
                    image_type
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

def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(description='Enhanced Image Processing with NudeNet and YOLO')
    parser.add_argument('command', choices=['sliding-json'], help='Command to execute')
    parser.add_argument('--json-url', required=True, help='URL to JSON file containing image data')
    parser.add_argument('--output-dir', default='processed_images', help='Output directory for processed images')
    parser.add_argument('--base-url', help='Base URL for converting relative paths to absolute URLs')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if output already exists')
    parser.add_argument('--download-only', action='store_true', help='Only download images, do not process them')
    
    args = parser.parse_args()
    
    if args.command == 'sliding-json':
        result = sliding_json(
            json_url=args.json_url,
            output_dir=args.output_dir,
            base_url=args.base_url,
            force=args.force,
            download_only=args.download_only
        )
        
        if not result['success']:
            print(f"❌ {result['message']}")
            return 1
        
        print("✅ Processing completed successfully!")
        return 0

if __name__ == "__main__":
    exit(main()) 