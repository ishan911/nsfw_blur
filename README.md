# Image Blurring API

A Python-based image processing API that uses a NudeDetector model to detect and blur sensitive content in images. The API supports both single image processing and batch processing with record keeping.

## Features

- **Single Image Processing**: Process individual images with customizable pixelation
- **Batch Processing**: Process multiple images in a directory
- **Sliding Window Detection**: Advanced detection for large images using overlapping windows
- **WordPress Integration**: Process existing images from WordPress uploads folder
- **JSON Processing**: Process images from JSON data (URLs or local files)
- **Record Keeping**: Track processed images to avoid reprocessing
- **WordPress Image Sizing**: Create WordPress-compatible image sizes
- **Customizable Parameters**: Adjust detection sensitivity, pixelation size, and more

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the model file to `models/640m.onnx`

## Usage

### Basic Commands

```bash
# Process a single image
python main.py single input.jpg output.jpg

# Process multiple images in a directory
python main.py batch data/input data/output

# Process images from JSON data
python main.py json data/images.json --base-url https://example.com
```

### Sliding Window Commands

```bash
# Process single image with sliding window
python main.py sliding-single input.jpg output.jpg --window-size 160 --stride 80

# Process batch with sliding window
python main.py sliding-batch data/input data/output --window-size 640 --stride 320

# Process JSON data with sliding window
python main.py sliding-json data/images.json --window-size 512 --stride 256
```

### WordPress Integration

```bash
# Process existing images from wp-content/uploads folder
python main.py sliding-wordpress data/images.json --window-size 512 --stride 256
```

The `sliding-wordpress` command works like `sliding-json` but instead of downloading images from URLs, it processes existing images from the WordPress `wp-content/uploads` folder using relative paths.

**Key Features:**
- Looks for images in `wp-content/uploads/` for `screenshot_full_url` images
- Looks for images in `wp-content/uploads/screenshots/` for `review_full_image` images
- Creates WordPress-sized images (170x145, 250x212 for screenshots; 590x504 for reviews)
- Saves processed images in the root `wp-content/uploads` folder structure

### Advanced Options

```bash
# Customize detection parameters
python main.py sliding-wordpress data/images.json \
  --window-size 640 \
  --stride 320 \
  --overlap-threshold 0.3 \
  --confidence-threshold 0.15 \
  --pixel-size 15 \
  --force

# Process from remote JSON URL
python main.py sliding-wordpress https://api.example.com/images.json \
  --output processed_images
```

## Parameters

- `--window-size`: Size of sliding window in pixels (default: 640)
- `--stride`: Stride between windows in pixels (default: 320)
- `--overlap-threshold`: IoU threshold for merging detections (default: 0.3)
- `--confidence-threshold`: Minimum confidence for detections (default: 0.1)
- `--pixel-size`: Pixelation block size (default: 12)
- `--force`: Force reprocessing even if already processed
- `--output`: Output directory for processed images

## WordPress Folder Structure

The system expects images to be organized as follows:

```
wp-content/
├── uploads/
│   ├── screenshot1.jpg          # screenshot_full_url images
│   ├── screenshot2.png
│   └── screenshots/
│       ├── review1.jpg          # review_full_image images
│       └── review2.png
```

Processed images will be saved in the same structure with WordPress-sized variants.

## Testing

Run the test script to verify WordPress integration:

```bash
python test_wordpress.py
```

This will create test data and demonstrate the sliding-wordpress functionality. 