"""
Image Processors Module

This module contains various image processing classes for different use cases:
- Base ImageProcessor for standard image processing
- SlidingWindowImageProcessor for sliding window approach
- CustomJSONImageProcessor for processing images from JSON data
- SlidingWindowCustomJSONImageProcessor for sliding window JSON processing
- SlidingWindowWordPressImageProcessor for WordPress-specific processing
"""

from .base_processor import ImageProcessor
from .sliding_window_processor import SlidingWindowImageProcessor
from .custom_json_processor import CustomJSONImageProcessor
from .sliding_window_custom_json_processor import SlidingWindowCustomJSONImageProcessor
from .sliding_window_wordpress_processor import SlidingWindowWordPressImageProcessor

__all__ = [
    'ImageProcessor',
    'SlidingWindowImageProcessor', 
    'CustomJSONImageProcessor',
    'SlidingWindowCustomJSONImageProcessor',
    'SlidingWindowWordPressImageProcessor'
] 