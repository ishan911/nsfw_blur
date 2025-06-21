# Image Processors Module

This module contains various image processing classes for different use cases, organized in a structured and maintainable way.

## Structure

```
src/processors/
├── __init__.py                           # Module exports
├── base_processor.py                     # Base ImageProcessor class
├── sliding_window_processor.py           # SlidingWindowImageProcessor class
├── custom_json_processor.py              # CustomJSONImageProcessor class
├── sliding_window_custom_json_processor.py # SlidingWindowCustomJSONImageProcessor class
├── sliding_window_wordpress_processor.py # SlidingWindowWordPressImageProcessor class
├── utils.py                              # Utility functions
└── README.md                             # This file
```

## Classes

### ImageProcessor (base_processor.py)
The base class that provides standard image processing functionality with database tracking.

**Features:**
- Single and batch image processing
- Database tracking for processed images
- File hash checking to detect changes
- Processing statistics and cleanup utilities

### SlidingWindowImageProcessor (sliding_window_processor.py)
Extends ImageProcessor to provide sliding window image processing functionality.

**Features:**
- All features from ImageProcessor
- Sliding window approach for large images
- Configurable window size, stride, and overlap threshold
- Confidence threshold support

### CustomJSONImageProcessor (custom_json_processor.py)
Extends ImageProcessor to process images specified in JSON format.

**Features:**
- JSON data fetching from URLs or files
- Image URL extraction from JSON items
- Image downloading functionality
- Support for relative and absolute URLs

### SlidingWindowCustomJSONImageProcessor (sliding_window_custom_json_processor.py)
Combines sliding window processing with JSON-based image handling.

**Features:**
- All features from CustomJSONImageProcessor
- Sliding window processing for JSON-specified images
- Enhanced processing statistics

### SlidingWindowWordPressImageProcessor (sliding_window_wordpress_processor.py)
Specialized processor for existing WordPress images.

**Features:**
- All features from SlidingWindowCustomJSONImageProcessor
- WordPress uploads folder integration
- Automatic path resolution for WordPress image types

## Usage

### Basic Usage
```python
from src.processors import ImageProcessor

# Create processor instance
processor = ImageProcessor(
    model_path='models/640m.onnx',
    database_path='data/processed_images.json'
)

# Process single image
processor.process_single_image(
    input_path='input.jpg',
    output_path='output.jpg',
    pixel_size=10
)
```

### Sliding Window Usage
```python
from src.processors import SlidingWindowImageProcessor

# Create sliding window processor
processor = SlidingWindowImageProcessor(
    window_size=640,
    stride=320,
    overlap_threshold=0.3
)

# Process with sliding window
processor.process_single_image(
    input_path='large_image.jpg',
    output_path='processed_large_image.jpg',
    confidence_threshold=0.1
)
```

### JSON Processing Usage
```python
from src.processors import CustomJSONImageProcessor

# Create JSON processor
processor = CustomJSONImageProcessor(
    json_url='https://api.example.com/images',
    base_url='https://example.com'
)

# Process JSON-specified images
processor.process_custom_json_images(
    output_dir='data/processed',
    pixel_size=10,
    download_only=False
)
```

## Utility Functions

### list_available_parts()
Lists all available body parts that can be blurred.

### validate_paths(input_path, output_path, model_path)
Validates file and directory paths for processing.

## Database Tracking

All processors maintain a JSON database of processed images with:
- Input and output file paths
- File hashes for change detection
- Processing parameters
- Timestamps
- File sizes

This allows for:
- Skipping already processed images
- Detecting when source files have changed
- Tracking processing statistics
- Cleaning up orphaned records

## Inheritance Hierarchy

```
ImageProcessor (base)
├── SlidingWindowImageProcessor
├── CustomJSONImageProcessor
    └── SlidingWindowCustomJSONImageProcessor
        └── SlidingWindowWordPressImageProcessor
```

Each class extends the functionality of its parent while maintaining compatibility with the base interface. 