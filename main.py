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


def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Image Processing Script with Record Keeping and Custom JSON Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s single input.jpg output.jpg
  %(prog)s batch data/input data/output
  %(prog)s single image.jpg result.jpg --pixel-size 15
  %(prog)s batch photos/ processed/ --force
  %(prog)s json https://example.com/api/images --base-url https://example.com
  %(prog)s json data/images.json --base-url https://example.com
  %(prog)s json data/images.json --download-only
  %(prog)s sliding-single input.jpg output.jpg --window-size 160 --stride 80
  %(prog)s sliding-batch data/input data/output --window-size 640 --stride 320
  %(prog)s sliding-json data/images.json --window-size 512 --stride 256
  %(prog)s sliding-wordpress data/images.json --window-size 512 --stride 256
  %(prog)s list-parts
  %(prog)s stats
  %(prog)s cleanup
  %(prog)s  # Run with default settings (batch processing)
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['single', 'batch', 'json', 'sliding-single', 'sliding-batch', 'sliding-json', 'sliding-wordpress', 'list-parts', 'stats', 'cleanup'],
        default='batch',
        help='Command to execute (default: batch)'
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Input file, directory path, JSON URL, or JSON file path'
    )
    
    parser.add_argument(
        'output',
        nargs='?',
        help='Output file or directory path'
    )
    
    parser.add_argument(
        '--model',
        default='models/640m.onnx',
        help='Path to the model file (default: models/640m.onnx)'
    )
    
    parser.add_argument(
        '--pixel-size',
        type=int,
        default=12,
        help='Pixelation size (default: 10)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if image was already processed'
    )
    
    parser.add_argument(
        '--database',
        default='data/processed_images.json',
        help='Path to the processing database (default: data/processed_images.json)'
    )
    
    # Sliding window specific arguments
    parser.add_argument(
        '--window-size',
        type=int,
        default=640,
        help='Sliding window size in pixels (default: 640)'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=320,
        help='Stride between sliding windows in pixels (default: 320)'
    )
    
    parser.add_argument(
        '--overlap-threshold',
        type=float,
        default=0.3,
        help='IoU threshold for merging overlapping detections (default: 0.3)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.1,
        help='Minimum confidence score for detections (default: 0.1)'
    )
    
    # Custom JSON specific arguments
    parser.add_argument(
        '--base-url',
        help='Base URL for converting relative paths to absolute URLs'
    )
    
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download images, do not process them'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output with detailed processing information'
    )
    
    args = parser.parse_args()
    
    # Handle non-processing commands
    if args.command == 'list-parts':
        list_available_parts()
        return
    elif args.command == 'stats':
        processor = ImageProcessor(database_path=args.database)
        processor.get_processing_stats()
        return
    elif args.command == 'cleanup':
        processor = ImageProcessor(database_path=args.database)
        processor.cleanup_orphaned_records()
        return
    
    # Handle sliding window custom JSON processing
    if args.command == 'sliding-json':
        if not args.input:
            parser.error("sliding-json command requires JSON URL or file path")
        
        # Determine if input is URL or file
        if args.input.startswith(('http://', 'https://')):
            json_url = args.input
            json_file = None
        else:
            json_url = None
            json_file = args.input
        
        json_processor = SlidingWindowCustomJSONImageProcessor(
            json_url=json_url,
            json_file=json_file,
            base_url=args.base_url,
            model_path=args.model,
            database_path=args.database,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold
        )
        print('Processing custom JSON images with sliding window... ', args.overlap_threshold)
        
        # Clean up orphaned records
        json_processor.cleanup_orphaned_records()
        
        # Show current statistics
        json_processor.get_processing_stats()
        
        # Process custom JSON images with sliding window
        output_dir = args.output if args.output else "data/custom_processed"
        json_processor.process_custom_json_images(
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            confidence_threshold=args.confidence_threshold,
            force=args.force,
            download_only=args.download_only
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        json_processor.get_processing_stats()
        return
    
    # Handle custom JSON processing (original)
    if args.command == 'json':
        if not args.input:
            parser.error("json command requires JSON URL or file path")
        
        # Determine if input is URL or file
        if args.input.startswith(('http://', 'https://')):
            json_url = args.input
            json_file = None
        else:
            json_url = None
            json_file = args.input
        
        json_processor = CustomJSONImageProcessor(
            json_url=json_url,
            json_file=json_file,
            base_url=args.base_url,
            model_path=args.model,
            database_path=args.database
        )
        
        # Clean up orphaned records
        json_processor.cleanup_orphaned_records()
        
        # Show current statistics
        json_processor.get_processing_stats()
        
        # Process custom JSON images
        output_dir = args.output if args.output else "data/custom_processed"
        json_processor.process_custom_json_images(
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            force=args.force,
            download_only=args.download_only
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        json_processor.get_processing_stats()
        return
    
    # Handle sliding window single image processing
    if args.command == 'sliding-single':
        if not args.input or not args.output:
            parser.error("sliding-single command requires input and output arguments")
        
        if not validate_paths(args.input, args.output, args.model):
            sys.exit(1)
        
        # Initialize sliding window processor
        processor = SlidingWindowImageProcessor(
            model_path=args.model, 
            database_path=args.database,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold
        )
        
        # Clean up orphaned records
        processor.cleanup_orphaned_records()
        
        # Show current statistics
        processor.get_processing_stats()
        
        print(f"Processing single image with sliding window: {args.input} -> {args.output}")
        print(f"Window size: {args.window_size}x{args.window_size}, Stride: {args.stride}")
        
        processor.process_single_image(
            input_path=args.input,
            output_path=args.output,
            pixel_size=args.pixel_size,
            confidence_threshold=args.confidence_threshold,
            force=args.force
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        processor.get_processing_stats()
        return
    
    # Handle sliding window batch processing
    if args.command == 'sliding-batch':
        # Use default paths if not provided
        input_dir = args.input if args.input else "data/input"
        output_dir = args.output if args.output else "data/output"
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        if not validate_paths(input_dir, output_dir, args.model):
            sys.exit(1)
        
        # Initialize sliding window processor
        processor = SlidingWindowImageProcessor(
            model_path=args.model, 
            database_path=args.database,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold
        )
        
        # Clean up orphaned records
        processor.cleanup_orphaned_records()
        
        # Show current statistics
        processor.get_processing_stats()
        
        print(f"Processing batch images with sliding window from: {input_dir} -> {output_dir}")
        print(f"Window size: {args.window_size}x{args.window_size}, Stride: {args.stride}")
        
        processor.process_multiple_images(
            input_dir=input_dir,
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            confidence_threshold=args.confidence_threshold,
            force=args.force
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        processor.get_processing_stats()
        return
    
    # Handle sliding window WordPress processing
    if args.command == 'sliding-wordpress':
        if not args.input:
            parser.error("sliding-wordpress command requires JSON URL or file path")
        
        # Determine if input is URL or file
        if args.input.startswith(('http://', 'https://')):
            json_url = args.input
            json_file = None
        else:
            json_url = None
            json_file = args.input
        
        wordpress_processor = SlidingWindowWordPressImageProcessor(
            json_url=json_url,
            json_file=json_file,
            base_url=args.base_url,
            model_path=args.model,
            database_path=args.database,
            window_size=args.window_size,
            stride=args.stride,
            overlap_threshold=args.overlap_threshold
        )
        print('Processing existing WordPress images with sliding window...')
        
        # Clean up orphaned records
        wordpress_processor.cleanup_orphaned_records()
        
        # Show current statistics
        wordpress_processor.get_processing_stats()
        
        # Process WordPress images with sliding window
        output_dir = args.output if args.output else "data/wordpress_processed"
        wordpress_processor.process_wordpress_json_images(
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            confidence_threshold=args.confidence_threshold,
            force=args.force,
            debug=args.debug
        )
        
        # Show updated statistics
        print("\n" + "="*50)
        wordpress_processor.get_processing_stats()
        return
    
    # Initialize processor for other commands (original functionality)
    processor = ImageProcessor(model_path=args.model, database_path=args.database)
    
    # Clean up orphaned records
    processor.cleanup_orphaned_records()
    
    # Show current statistics
    processor.get_processing_stats()
    
    # Handle processing commands (original functionality)
    if args.command == 'single':
        if not args.input or not args.output:
            parser.error("single command requires input and output arguments")
        
        if not validate_paths(args.input, args.output, args.model):
            sys.exit(1)
        
        print(f"Processing single image: {args.input} -> {args.output}")
        processor.process_single_image(
            input_path=args.input,
            output_path=args.output,
            pixel_size=args.pixel_size,
            force=args.force
        )
    
    elif args.command == 'batch':
        # Use default paths if not provided
        input_dir = args.input if args.input else "data/input"
        output_dir = args.output if args.output else "data/output"
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        if not validate_paths(input_dir, output_dir, args.model):
            sys.exit(1)
        
        print(f"Processing batch images from: {input_dir} -> {output_dir}")
        processor.process_multiple_images(
            input_dir=input_dir,
            output_dir=output_dir,
            pixel_size=args.pixel_size,
            force=args.force
        )
    
    # Show updated statistics
    print("\n" + "="*50)
    processor.get_processing_stats()


if __name__ == "__main__":
    main() 