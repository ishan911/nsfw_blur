# NudeNet Detector Class

A clean, object-oriented wrapper for NudeNet detection and pixelation functionality.

## Features

- **Easy to use**: Simple class interface for detection and pixelation
- **Configurable**: Customizable confidence thresholds, pixel sizes, and padding
- **Multiple detection methods**: Enhanced detection, sliding window, and simple detection
- **Flexible output**: Control rectangle borders and labels
- **Privacy protection**: Automatic pixelation with configurable padding

## Quick Start

### Basic Usage

```python
from nudenet_detector import NudeNetDetector

# Create detector
detector = NudeNetDetector(
    confidence_threshold=0.05,  # Detection sensitivity
    pixel_size=15,              # Pixelation intensity
    padding=5                   # Extra padding around detections
)

# Process an image
result = detector.process_image(
    input_path="input.jpg",
    output_path="output.jpg",
    use_sliding_window=True,    # Better detection
    draw_rectangles=False,      # Clean output
    draw_labels=False           # No labels
)

print(f"Found {result['detection_count']} detections")
```

### Advanced Usage

```python
# Create detector with custom settings
detector = NudeNetDetector(
    confidence_threshold=0.01,  # Very sensitive
    pixel_size=20,              # Heavy pixelation
    padding=10                  # Large padding
)

# Just run detection without pixelation
detections = detector.detect_enhanced("input.jpg")
print(f"Found {len(detections)} detections")

# Use sliding window for comprehensive detection
sliding_detections = detector.detect_with_sliding_window("input.jpg")
print(f"Found {len(sliding_detections)} detections with sliding window")

# Pixelate with visual indicators
detector.pixelate_image(
    "input.jpg", 
    detections, 
    "output_with_borders.jpg",
    draw_rectangles=True,
    draw_labels=True
)
```

## Class Methods

### Initialization

```python
NudeNetDetector(
    confidence_threshold=0.05,  # Minimum confidence (0.01-1.0)
    pixel_size=15,              # Pixel block size (5-25)
    padding=5                   # Padding around detections
)
```

### Main Methods

#### `process_image(input_path, output_path, use_sliding_window=True, draw_rectangles=False, draw_labels=False)`
Complete image processing pipeline.

**Returns:**
```python
{
    'success': True/False,
    'detection_count': 5,
    'detections': [...],
    'message': 'Processed 5 detections'
}
```

#### `detect_enhanced(image_path, enhancement_factor=1.5)`
Enhanced detection with multiple preprocessing techniques.

#### `detect_simple(image_path)`
Simple detection without preprocessing.

#### `detect_with_sliding_window(input_path, window_size=(256, 256), step_size=128)`
Sliding window detection for comprehensive coverage.

#### `pixelate_image(image_path, detections, output_path, draw_rectangles=False, draw_labels=False)`
Pixelate detected regions in an image.

## Configuration Options

### Detection Settings
- `confidence_threshold`: Lower values = more detections (0.01-1.0)
- `use_sliding_window`: Better detection but slower
- `enhancement_factor`: Image preprocessing intensity

### Pixelation Settings
- `pixel_size`: Higher values = more pixelated (5-25)
- `padding`: Extra pixels around detections (0-20)

### Visual Settings
- `draw_rectangles`: Show colored borders around detections
- `draw_labels`: Show detection info and confidence scores

## Examples

### Clean Privacy Protection
```python
detector = NudeNetDetector(confidence_threshold=0.05, pixel_size=15, padding=5)
result = detector.process_image(
    "input.jpg", "output.jpg",
    use_sliding_window=True,
    draw_rectangles=False,
    draw_labels=False
)
```

### Analysis Mode
```python
detector = NudeNetDetector(confidence_threshold=0.01, pixel_size=10, padding=0)
result = detector.process_image(
    "input.jpg", "analysis.jpg",
    use_sliding_window=False,
    draw_rectangles=True,
    draw_labels=True
)
```

### Heavy Pixelation
```python
detector = NudeNetDetector(confidence_threshold=0.05, pixel_size=25, padding=10)
result = detector.process_image(
    "input.jpg", "heavily_pixelated.jpg",
    use_sliding_window=True,
    draw_rectangles=False,
    draw_labels=False
)
```

## Output Modes

1. **Clean Privacy Protection**: No rectangles or labels, just pixelation
2. **Analysis Mode**: Rectangles and labels for detection review
3. **Detection Only**: Just run detection without pixelation
4. **Custom**: Mix and match settings as needed

## Performance Tips

- **Fast processing**: Use `use_sliding_window=False`
- **Best detection**: Use `use_sliding_window=True`
- **Large images**: Increase `window_size` and `step_size`
- **Memory usage**: Lower `enhancement_factor` for large images

## Error Handling

The class includes comprehensive error handling:
- Invalid image paths
- Detection failures
- Pixelation errors
- File I/O issues

All methods return structured results with success status and error messages.

## Dependencies

- OpenCV (`cv2`)
- NumPy
- PIL (Pillow)
- NudeNet (`nudenet`)

## Installation

```bash
pip install opencv-python numpy pillow nudenet
``` 