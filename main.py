#!/usr/bin/env python3
"""
Image Processing Script with Record Keeping
This script demonstrates how to use the ImageBlurrer class with database tracking.
Can be used as a standalone script with command-line arguments.
"""

import os
import sys
import argparse
from pathlib import Path
import shutil

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.processors import (
    ImageProcessor,
    SlidingWindowImageProcessor,
    CustomJSONImageProcessor,
    SlidingWindowCustomJSONImageProcessor,
    SlidingWindowWordPressImageProcessor
)
from src.processors.utils import list_available_parts, validate_paths


def restore_backup_files(backup_dir, dry_run=True):
    """
    Restore backup files from the backup directory to their original locations.
    
    Args:
        backup_dir (str): Path to the backup directory
        dry_run (bool): If True, only show what would be restored without actually restoring
    
    Returns:
        dict: Summary of restore operation
    """
    if not os.path.exists(backup_dir):
        return {"error": f"Backup directory not found: {backup_dir}"}
    
    # Find all backup files
    backup_files = []
    for root, dirs, files in os.walk(backup_dir):
        for file in files:
            if file.endswith('.backup'):
                backup_files.append(Path(root) / file)
    
    if not backup_files:
        return {"error": "No backup files found"}
    
    restored_count = 0
    skipped_count = 0
    error_count = 0
    
    for backup_file in backup_files:
        # Determine target path by removing .backup extension and backup directory prefix
        relative_path = backup_file.relative_to(backup_dir)
        target_path = Path('wp-content/uploads') / relative_path.with_suffix('')
        
        if dry_run:
            # Just show what would be restored
            pass
        else:
            # Actually restore the file
            try:
                # Create target directory if it doesn't exist
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if target already exists
                if target_path.exists():
                    skipped_count += 1
                    continue
                
                # Copy backup file to target location
                shutil.copy2(backup_file, target_path)
                restored_count += 1
                
            except Exception as e:
                error_count += 1
                continue
    
    if dry_run:
        return {
            "mode": "dry_run",
            "total_files": len(backup_files),
            "would_restore": len(backup_files)
        }
    else:
        return {
            "mode": "restore",
            "total_files": len(backup_files),
            "restored": restored_count,
            "skipped": skipped_count,
            "errors": error_count
        }


def main():
    parser = argparse.ArgumentParser(description='WordPress Image Processing Tool')
    parser.add_argument('command', choices=[
        'process-single', 'process-batch', 'process-wordpress', 
        'process-custom-json', 'sliding-json', 'process-single-slide', 'process-batch-slide',
        'sliding-single', 'sliding-batch', 'sliding-wordpress',
        'restore-backup'
    ], help='Command to execute')
    
    # Positional arguments for legacy commands
    parser.add_argument('input', nargs='?', help='Input file, directory, or JSON URL')
    parser.add_argument('output', nargs='?', help='Output file or directory')
    
    # Common arguments
    parser.add_argument('--input', '-i', help='Input file or directory (alternative to positional)')
    parser.add_argument('--output', '-o', help='Output file or directory (alternative to positional)')
    parser.add_argument('--window-size', type=int, default=512, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=256, help='Sliding window stride')
    parser.add_argument('--overlap-threshold', type=float, default=0.3, help='Overlap threshold for merging detections')
    parser.add_argument('--confidence-threshold', type=float, default=0.1, help='Minimum confidence for detections')
    
    # Detection method arguments
    parser.add_argument('--nudenet', action='store_true', help='Enable NudeNet detection')
    parser.add_argument('--yolo', action='store_true', help='Enable YOLO detection')
    parser.add_argument('--nudenet-confidence', type=float, default=0.6, help='NudeNet confidence threshold')
    parser.add_argument('--yolo-confidence', type=float, default=0.5, help='YOLO confidence threshold')
    parser.add_argument('--yolo-model', help='Path to YOLO model file')
    
    # Blur method arguments
    parser.add_argument('--blur-method', choices=['pixelate', 'blur'], default='pixelate', help='Blur method')
    parser.add_argument('--pixel-size', type=int, default=10, help='Pixel size for pixelation')
    
    # WordPress specific arguments
    parser.add_argument('--wordpress-dir', default='wp-content/uploads', help='WordPress uploads directory')
    
    # Custom JSON specific arguments
    parser.add_argument('--json-url', help='URL to JSON data (alternative to positional input)')
    parser.add_argument('--base-url', help='Base URL for converting relative paths to absolute URLs')
    parser.add_argument('--download-dir', default='downloads', help='Directory to download images')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if already processed')
    parser.add_argument('--download-only', action='store_true', help='Only download images, do not process them')
    
    # Backup restore arguments
    parser.add_argument('--backup-dir', default='wp-content/uploads/backup', help='Backup directory path')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually restore files (not dry run)')
    
    args = parser.parse_args()
    
    # Handle positional arguments for legacy commands
    if args.command in ['sliding-json', 'sliding-single', 'sliding-batch', 'sliding-wordpress']:
        if args.input and not args.json_url:
            args.json_url = args.input
        if args.output:
            args.output_dir = args.output
    
    if args.command == 'restore-backup':
        result = restore_backup_files(args.backup_dir, dry_run=not args.no_dry_run)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        if result["mode"] == "dry_run":
            print(f"📋 Dry run complete - would restore {result['would_restore']} files")
        else:
            print(f"✅ Restore complete:")
            print(f"  Restored: {result['restored']} files")
            print(f"  Skipped: {result['skipped']} files (already exist)")
            print(f"  Errors: {result['errors']} files")
        return
    
    # Initialize detection methods
    nudenet_enabled = args.nudenet
    yolo_enabled = args.yolo
    
    # If neither is specified, enable both by default
    if not nudenet_enabled and not yolo_enabled:
        nudenet_enabled = True
        yolo_enabled = True
    
    if args.command == 'sliding-json':
        if not args.json_url:
            print("Error: JSON URL is required for sliding-json command")
            return
        
        processor = SlidingWindowCustomJSONImageProcessor(
            json_url=args.json_url,
            json_file=None,
            base_url=args.base_url,
            model_path=args.yolo_model,
            database_path=None,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold,
            nudenet_enabled=nudenet_enabled,
            yolo_enabled=yolo_enabled,
            nudenet_confidence_threshold=args.nudenet_confidence,
            yolo_confidence_threshold=args.yolo_confidence,
            yolo_model_path=args.yolo_model,
            blur_method=args.blur_method,
            pixel_size=args.pixel_size
        )
        
        processor.process_custom_json_images(
            output_dir=args.output,
            pixel_size=args.pixel_size,
            force=args.force,
            download_only=args.download_only,
            use_yolo_detection=True,
            yolo_confidence_threshold=args.yolo_confidence,
            yolo_model_path=args.yolo_model
        )
        
    elif args.command == 'process-custom-json':
        if not args.json_url:
            print("Error: --json-url is required for process-custom-json command")
            return
        
        processor = SlidingWindowCustomJSONImageProcessor(
            json_url=args.json_url,
            json_file=None,
            base_url=args.base_url,
            model_path=args.yolo_model,
            database_path=None,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold,
            nudenet_enabled=nudenet_enabled,
            yolo_enabled=yolo_enabled,
            nudenet_confidence_threshold=args.nudenet_confidence,
            yolo_confidence_threshold=args.yolo_confidence,
            yolo_model_path=args.yolo_model,
            blur_method=args.blur_method,
            pixel_size=args.pixel_size
        )
        
        processor.process_custom_json_images(
            output_dir=args.output,
            pixel_size=args.pixel_size,
            force=args.force,
            download_only=args.download_only,
            use_yolo_detection=True,
            yolo_confidence_threshold=args.yolo_confidence,
            yolo_model_path=args.yolo_model
        )
        
    elif args.command == 'process-single-slide' or args.command == 'sliding-single':
        if not args.input or not args.output:
            print("Error: --input and --output are required for process-single-slide/sliding-single command")
            return
        
        processor = SlidingWindowImageProcessor(
            model_path=args.yolo_model,
            database_path=None,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold,
            nudenet_enabled=nudenet_enabled,
            yolo_enabled=yolo_enabled,
            nudenet_confidence_threshold=args.nudenet_confidence,
            yolo_confidence_threshold=args.yolo_confidence,
            yolo_model_path=args.yolo_model,
            blur_method=args.blur_method,
            pixel_size=args.pixel_size
        )
        
        processor.process_single_image(
            input_path=args.input,
            output_path=args.output,
            pixel_size=args.pixel_size,
            confidence_threshold=args.yolo_confidence,
            force=True
        )
        
    elif args.command == 'process-batch-slide' or args.command == 'sliding-batch':
        if not args.input or not args.output:
            print("Error: --input and --output are required for process-batch-slide/sliding-batch command")
            return
        
        input_dir = args.input
        output_dir = args.output
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory does not exist: {input_dir}")
            return
        
        processor = SlidingWindowImageProcessor(
            model_path=args.yolo_model,
            database_path=None,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold,
            nudenet_enabled=nudenet_enabled,
            yolo_enabled=yolo_enabled,
            nudenet_confidence_threshold=args.nudenet_confidence,
            yolo_confidence_threshold=args.yolo_confidence,
            yolo_model_path=args.yolo_model,
            blur_method=args.blur_method,
            pixel_size=args.pixel_size
        )
        
        processor.process_multiple_images(
            input_dir=input_dir,
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            confidence_threshold=args.yolo_confidence,
            force=True
        )
        
    elif args.command == 'process-wordpress' or args.command == 'sliding-wordpress':
        processor = SlidingWindowWordPressImageProcessor(
            wordpress_dir=args.wordpress_dir,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold,
            nudenet_enabled=nudenet_enabled,
            yolo_enabled=yolo_enabled,
            nudenet_confidence_threshold=args.nudenet_confidence,
            yolo_confidence_threshold=args.yolo_confidence,
            yolo_model_path=args.yolo_model,
            blur_method=args.blur_method,
            pixel_size=args.pixel_size
        )
        
        processor.process_wordpress_json_images(
            output_dir=args.output,
            pixel_size=args.pixel_size,
            confidence_threshold=args.yolo_confidence,
            force=True,
            debug=False
        )
        
    elif args.command == 'process-single':
        if not args.input or not args.output:
            print("Error: --input and --output are required for process-single command")
            return
        
        processor = SlidingWindowImageProcessor(
            model_path=args.yolo_model,
            database_path=None,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold,
            nudenet_enabled=nudenet_enabled,
            yolo_enabled=yolo_enabled,
            nudenet_confidence_threshold=args.nudenet_confidence,
            yolo_confidence_threshold=args.yolo_confidence,
            yolo_model_path=args.yolo_model,
            blur_method=args.blur_method,
            pixel_size=args.pixel_size
        )
        
        processor.process_single_image(
            input_path=args.input,
            output_path=args.output,
            pixel_size=args.pixel_size,
            confidence_threshold=args.yolo_confidence,
            force=True
        )
        
    elif args.command == 'process-batch':
        if not args.input or not args.output:
            print("Error: --input and --output are required for process-batch command")
            return
        
        input_dir = args.input
        output_dir = args.output
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory does not exist: {input_dir}")
            return
        
        processor = SlidingWindowImageProcessor(
            model_path=args.yolo_model,
            database_path=None,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold,
            nudenet_enabled=nudenet_enabled,
            yolo_enabled=yolo_enabled,
            nudenet_confidence_threshold=args.nudenet_confidence,
            yolo_confidence_threshold=args.yolo_confidence,
            yolo_model_path=args.yolo_model,
            blur_method=args.blur_method,
            pixel_size=args.pixel_size
        )
        
        processor.process_multiple_images(
            input_dir=input_dir,
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            confidence_threshold=args.yolo_confidence,
            force=True
        )


if __name__ == "__main__":
    main() 