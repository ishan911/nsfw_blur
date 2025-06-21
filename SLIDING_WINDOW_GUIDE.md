# Sliding Window Cropping Method for Image Detection

## Overview

The sliding window cropping method is an advanced technique that improves detection accuracy for large images by processing them in smaller, overlapping windows. This approach is particularly useful when:

- Images are larger than the model's optimal input size
- Objects of interest are small relative to the image size
- You need to ensure no content is missed due to image scaling

## How It Works

### 1. Window Generation
The algorithm creates a grid of overlapping windows across the image:
- **Window Size**: Configurable size (default: 640x640 pixels)
- **Stride**: Distance between window centers (default: 320 pixels = 50% overlap)
- **Overlap**: Ensures no content falls between windows

### 2. Detection Process
For each window:
1. Crop the window from the original image
2. Run detection on the cropped window
3. Translate detection coordinates back to original image space
4. Collect all detections

### 3. Detection Merging
- Uses Intersection over Union (IoU) to identify overlapping detections
- Merges detections with IoU > threshold (default: 0.3)
- Keeps the detection with highest confidence score

## Implementation

### Basic Usage

```python
from src.blurrer import SlidingWindowBlurrer

# Initialize with default parameters
blurrer = SlidingWindowBlurrer(
    model_path='models/640m.onnx',
    parts=['FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED'],
    window_size=640,
    stride=320,
    overlap_threshold=0.3
)

# Process image
blurrer.process_image(
    input_path='input.jpg',
    output_path='output.png',
    pixel_size=10,
    confidence_threshold=0.1
)
```

### Advanced Configuration

```python
# High-resolution processing
blurrer_hr = SlidingWindowBlurrer(
    window_size=512,      # Smaller windows for finer detail
    stride=256,           # More overlap (50%)
    overlap_threshold=0.4  # Higher threshold for better merging
)

# Fast processing for large images
blurrer_fast = SlidingWindowBlurrer(
    window_size=800,      # Larger windows
    stride=600,           # Less overlap (25%)
    overlap_threshold=0.2  # Lower threshold
)
```

## Parameters Explained

### Window Size (`window_size`)
- **Purpose**: Size of each sliding window
- **Range**: 256-1024 pixels (recommended)
- **Trade-off**: 
  - Larger windows = faster processing, may miss small objects
  - Smaller windows = slower processing, better for small objects

### Stride (`stride`)
- **Purpose**: Distance between window centers
- **Range**: Should be ≤ window_size
- **Overlap**: `(window_size - stride) / window_size * 100`
- **Trade-off**:
  - Larger stride = faster processing, may miss objects at window boundaries
  - Smaller stride = slower processing, better coverage

### Overlap Threshold (`overlap_threshold`)
- **Purpose**: IoU threshold for merging duplicate detections
- **Range**: 0.0-1.0 (recommended: 0.2-0.5)
- **Trade-off**:
  - Lower threshold = more aggressive merging, fewer false positives
  - Higher threshold = less merging, may have duplicates

## Performance Considerations

### Processing Time
Processing time scales with:
- Number of windows: `(image_width / stride) * (image_height / stride)`
- Window size: Larger windows process faster
- Image resolution: Higher resolution = more windows

### Memory Usage
- Each window is processed independently
- Temporary files are created and cleaned up automatically
- Memory usage is proportional to window size

### Accuracy vs Speed Trade-offs

| Configuration | Speed | Accuracy | Use Case |
|---------------|-------|----------|----------|
| Large windows, high stride | Fast | Lower | Batch processing |
| Medium windows, medium stride | Balanced | Balanced | General use |
| Small windows, low stride | Slow | Higher | High-precision detection |

## Visualization

You can visualize the sliding windows to understand the coverage:

```python
# Process with window visualization
blurrer.process_image_with_visualization(
    input_path='input.jpg',
    output_path='output_with_windows.png',
    show_windows=True  # Draws red rectangles around each window
)
```

## Best Practices

### 1. Choose Window Size Based on Object Size
- **Small objects** (< 100px): Use 256-512 pixel windows
- **Medium objects** (100-300px): Use 512-640 pixel windows  
- **Large objects** (> 300px): Use 640-800 pixel windows

### 2. Adjust Stride for Coverage
- **High overlap** (75%): stride = window_size * 0.25
- **Medium overlap** (50%): stride = window_size * 0.5
- **Low overlap** (25%): stride = window_size * 0.75

### 3. Tune Confidence Threshold
- **Conservative**: 0.15-0.2 (fewer false positives)
- **Balanced**: 0.1-0.15 (default)
- **Aggressive**: 0.05-0.1 (more detections)

### 4. Optimize for Your Use Case
- **Real-time processing**: Use larger windows, higher stride
- **Batch processing**: Use balanced configuration
- **High-precision**: Use smaller windows, lower stride

## Example Configurations

### Fast Processing
```python
SlidingWindowBlurrer(
    window_size=800,
    stride=600,
    overlap_threshold=0.2
)
```

### Balanced Processing
```python
SlidingWindowBlurrer(
    window_size=640,
    stride=320,
    overlap_threshold=0.3
)
```

### High-Precision Processing
```python
SlidingWindowBlurrer(
    window_size=512,
    stride=256,
    overlap_threshold=0.4
)
```

## Troubleshooting

### Common Issues

1. **Too many windows**: Increase stride or window size
2. **Missed detections**: Decrease stride or window size
3. **Duplicate detections**: Increase overlap_threshold
4. **Slow processing**: Increase window size or stride

### Performance Optimization

1. **Use appropriate window size** for your object sizes
2. **Balance overlap** between coverage and speed
3. **Adjust confidence threshold** based on your requirements
4. **Consider image preprocessing** (resizing) for very large images

## Comparison with Original Method

| Aspect | Original Method | Sliding Window Method |
|--------|----------------|----------------------|
| **Large Images** | May miss content | Better coverage |
| **Small Objects** | Limited detection | Improved detection |
| **Processing Speed** | Fast | Slower but more thorough |
| **Memory Usage** | Low | Moderate |
| **Accuracy** | Good for standard images | Better for complex images |

## Conclusion

The sliding window method significantly improves detection accuracy for large images and small objects. While it requires more processing time, the trade-off is worthwhile for applications requiring high detection precision. Choose parameters based on your specific requirements for speed vs. accuracy. 