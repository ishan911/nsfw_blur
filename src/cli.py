#!/usr/bin/env python3
"""
Command-line interface for the ImageBlurrer.
This script provides a command-line interface to process images.
"""

import argparse
import os
from pathlib import Path
from .blurrer import ImageBlurrer

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process images to detect and pixelate sensitive content.')
    
    # Add arguments
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--output', '-o', help='Output image path or directory')
    parser.add_argument('--pixel-size', '-p', type=int, default=10,
                      help='Size of pixels for pixelation effect (default: 10)')
    parser.add_argument('--recursive', '-r', action='store_true',
                      help='Process directories recursively')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize blurrer
    blurrer = ImageBlurrer()
    
    # Process input
    if os.path.isfile(args.input):
        # Process single file
        output_path = args.output or f"processed_{os.path.basename(args.input)}"
        try:
            result = blurrer.process_image(
                input_path=args.input,
                output_path=output_path,
                pixel_size=args.pixel_size
            )
            print(f"Image processed successfully. Saved to: {result}")
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            
    elif os.path.isdir(args.input):
        # Process directory
        output_dir = args.output or "processed_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg')
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(image_extensions):
                    input_path = os.path.join(root, file)
                    # Create relative path for output
                    rel_path = os.path.relpath(input_path, args.input)
                    output_path = os.path.join(output_dir, rel_path)
                    # Create output directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    try:
                        result = blurrer.process_image(
                            input_path=input_path,
                            output_path=output_path,
                            pixel_size=args.pixel_size
                        )
                        print(f"Processed {rel_path} successfully")
                    except Exception as e:
                        print(f"Error processing {rel_path}: {str(e)}")
                        
            if not args.recursive:
                break
    else:
        print(f"Error: Input path '{args.input}' does not exist")

if __name__ == "__main__":
    main() 