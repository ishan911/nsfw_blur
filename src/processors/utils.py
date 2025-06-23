#!/usr/bin/env python3
"""
Utilities for Image Processing

This module contains utility functions for image processing operations.
"""

import os


def list_available_parts():
    """List available body parts for blurring."""
    parts = [
        'BUTTOCKS_EXPOSED',
        'BUTTOCKS_COVERED',
        'FEMALE_BREAST_EXPOSED',
        'FEMALE_GENITALIA_EXPOSED',
        'ANUS_EXPOSED',
        'MALE_GENITALIA_EXPOSED'
    ]
    
    print("Available body parts for blurring:")
    for part in parts:
        print(f"  - {part}")
    print("\nNote: You can specify multiple parts separated by commas")


def validate_paths(input_path, output_path, model_path):
    """Validate file and directory paths."""
    # Check if input file/directory exists
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return False
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file does not exist: {model_path}")
        return False
    
    # Check if input is an image file (for single mode)
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Error: Input file is not a supported image format (jpg, jpeg, png): {input_path}")
            return False
    
    return True 