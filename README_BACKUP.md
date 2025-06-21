# WordPress Image Processing with Backup Functionality

This document explains the backup functionality added to the WordPress image processor.

## Overview

The `SlidingWindowWordPressImageProcessor` now includes automatic backup creation before processing any images. This ensures that original images are safely preserved before any modifications are made.

## Backup Functionality

### How It Works

1. **Automatic Backup Creation**: Before processing any image, the system automatically creates a backup of the original file
2. **Organized Backup Structure**: Backups are organized in a `backups/` directory that mirrors the original WordPress uploads structure
3. **Timestamped Backups**: Each backup includes a timestamp to prevent overwrites
4. **Complete Path Preservation**: The backup maintains the original directory structure

### Backup Directory Structure

```
backups/
├── theartporn-featured-image_backup_20241201_143022.jpg
├── screenshots/
│   └── theartpornfeaturedimage_watermark_backup_20241201_143025.jpg
└── other-images/
    └── another-image_backup_20241201_143030.jpg
```

### Backup Naming Convention

- **Format**: `{original_filename}_backup_{YYYYMMDD_HHMMSS}.{extension}`
- **Example**: `theartporn-featured-image_backup_20241201_143022.jpg`

## Usage

### Command Line Usage

```bash
# Process WordPress images with backup functionality
python main.py sliding-wordpress data/wordpress_images.json \
    --window-size 640 \
    --stride 320 \
    --pixel-size 10 \
    --confidence-threshold 0.1
```

### Programmatic Usage

```python
from src.processors import SlidingWindowWordPressImageProcessor

# Initialize processor
processor = SlidingWindowWordPressImageProcessor(
    json_file="data/wordpress_images.json",
    model_path="models/640m.onnx",
    window_size=640,
    stride=320
)

# Process images (backups created automatically)
processor.process_wordpress_json_images(
    pixel_size=10,
    confidence_threshold=0.1,
    force=False
)
```

## WordPress Directory Structure

The processor expects images to be located in the standard WordPress uploads structure:

```
wp-content/
└── uploads/
    ├── image1.jpg                    # Featured images
    ├── image2.jpg
    └── screenshots/
        ├── screenshot1.jpg           # Screenshot images
        └── screenshot2.jpg
```

## JSON Data Format

The processor reads image data from JSON files with this format:

```json
[
    {
        "slug": "theartporn",
        "screenshot_full_url": "/wp-content/uploads/theartporn-featured-image.jpg",
        "review_full_image": "/wp-content/uploads/screenshots/theartpornfeaturedimage_watermark.jpg"
    }
]
```

## Processing Flow

1. **JSON Parsing**: Read image URLs from JSON file
2. **Path Resolution**: Convert relative URLs to local file paths
3. **File Existence Check**: Verify images exist in wp-content/uploads
4. **Backup Creation**: Create timestamped backup before processing
5. **Image Processing**: Apply sliding window detection and pixelation
6. **In-Place Update**: Replace original image with processed version
7. **Database Tracking**: Record processing details

## Safety Features

### Backup Verification

- **File Existence Check**: Verifies original file exists before backup
- **Copy Verification**: Uses `shutil.copy2()` for reliable file copying
- **Error Handling**: Graceful handling of backup failures

### Processing Safety

- **No Backup, No Processing**: Processing stops if backup creation fails
- **Database Tracking**: All processed images are tracked to prevent duplicate processing
- **Force Option**: Use `--force` flag to reprocess already processed images

## Recovery

### Restoring from Backup

To restore an image from backup:

```bash
# Find the backup file
ls backups/

# Restore the image
cp backups/theartporn-featured-image_backup_20241201_143022.jpg \
   wp-content/uploads/theartporn-featured-image.jpg
```

### Batch Restore

```python
import os
import shutil

# Restore all images from backups
backup_dir = "backups"
for root, dirs, files in os.walk(backup_dir):
    for file in files:
        if file.endswith("_backup_"):
            # Extract original filename
            original_name = file.split("_backup_")[0] + "." + file.split(".")[-1]
            
            # Determine original path
            relative_path = os.path.relpath(root, backup_dir)
            if relative_path == ".":
                original_path = f"wp-content/uploads/{original_name}"
            else:
                original_path = f"wp-content/uploads/{relative_path}/{original_name}"
            
            # Restore file
            backup_path = os.path.join(root, file)
            shutil.copy2(backup_path, original_path)
            print(f"Restored: {original_path}")
```

## Configuration Options

### Backup Settings

- **Backup Directory**: `backups/` (configurable in code)
- **Timestamp Format**: `YYYYMMDD_HHMMSS`
- **File Preservation**: Uses `shutil.copy2()` to preserve metadata

### Processing Settings

- **Window Size**: Size of sliding window (default: 640px)
- **Stride**: Step size between windows (default: 320px)
- **Pixel Size**: Size of pixelation blocks (default: 10px)
- **Confidence Threshold**: Minimum detection confidence (default: 0.1)

## Testing

Run the test script to verify backup functionality:

```bash
python test_wordpress_backup.py
```

This will:
1. Create test WordPress directory structure
2. Generate test images
3. Test backup creation
4. Test WordPress processing
5. Show backup directory structure

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure write permissions for backup directory
2. **Disk Space**: Ensure sufficient space for backups
3. **File Not Found**: Verify images exist in wp-content/uploads
4. **Backup Failures**: Check file permissions and disk space

### Debug Mode

Enable verbose logging by modifying the processor code to include more detailed output during backup operations. 