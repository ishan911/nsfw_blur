# NudeNet Detection with Pixelation

This enhanced version of the NudeNet detection script now includes pixelation functionality to blur detected sensitive content.

## Features

### 🔍 Enhanced Detection
- **Multiple preprocessing techniques** for better detection on darker bodies
- **Upscaling detection** for small body parts (1.5x, 2.0x, 2.5x, 3.0x)
- **Sliding window detection** for comprehensive coverage
- **Adaptive confidence thresholds**

### 🎨 Pixelation Options
- **Pixelate detected regions** instead of just drawing rectangles
- **Adjustable pixel size** (higher values = more pixelated)
- **Adaptive pixelation** based on detection confidence
- **Multiple output formats**:
  - Pixelation + rectangle borders + labels
  - Pixelation only (clean output)
  - Rectangle borders only (no pixelation)

## Usage

### Basic Usage

```python
# Run the main script with pixelation enabled
python detect_all_parts.py
```

### Configuration Options

In the `main()` function, you can configure:

```python
# Pixelation configuration
pixelate_detections = True      # Enable/disable pixelation
pixel_size = 15                # Size of pixel blocks (5-25 recommended)
draw_rectangle_borders = False # Show colored borders around detections
draw_labels = False            # Show labels (only if draw_rectangle_borders=True)
```

### Output Modes

The script now supports multiple output modes:

1. **Pixelation Only** (`draw_rectangle_borders=False`, `draw_labels=False`)
   - Clean pixelated output with no visual indicators
   - Perfect for privacy protection

2. **Pixelation + Rectangles** (`draw_rectangle_borders=True`, `draw_labels=False`)
   - Pixelated regions with colored borders
   - Shows detection areas without text labels

3. **Full Detection** (`draw_rectangle_borders=True`, `draw_labels=True`)
   - Pixelated regions with borders and labels
   - Shows detection info and confidence scores

4. **Rectangles Only** (`pixelate=False`, `draw_rectangle_borders=True`)
   - Original detection rectangles without pixelation
   - For analysis purposes

### Test Different Settings

```python
# Run the test script to see different pixelation effects
python test_pixelation.py
```

This will generate multiple output files with different pixel sizes for comparison.

## Output Files

### Main Script Outputs
- `nudenet_detections.jpg` - Full detection with pixelation and labels (if enabled)
- `nudenet_pixelated_only.jpg` - Clean pixelated version (no labels)
- `sliding_window_combined_detections.jpg` - Sliding window results

### Test Script Outputs
- `test_pixelation_{size}_with_labels.jpg` - Pixelation + rectangles + labels
- `test_pixelation_{size}_only.jpg` - Pixelation only (clean)
- `test_no_pixelation.jpg` - Rectangles only (no pixelation)
- `test_pixelation_only.jpg` - Pixelation only (no rectangles or labels)

## Pixelation Settings Guide

| Pixel Size | Effect | Use Case |
|------------|--------|----------|
| 5-10       | Light pixelation | Subtle blurring |
| 10-15      | Medium pixelation | Standard privacy protection |
| 15-20      | Heavy pixelation | Strong privacy protection |
| 20-25      | Very heavy pixelation | Maximum privacy protection |

## Technical Details

### Pixelation Algorithm
1. **Region Extraction**: Extract the detected bounding box region
2. **Downsampling**: Resize to smaller dimensions using `cv2.INTER_AREA`
3. **Upsampling**: Resize back to original size using `cv2.INTER_NEAREST`
4. **Replacement**: Replace the original region with pixelated version

### Adaptive Pixelation
The script automatically adjusts pixel size based on:
- **Detection confidence**: Higher confidence = larger pixel blocks
- **Region size**: Smaller regions get smaller pixel blocks
- **User setting**: Base pixel size from configuration

### Supported Body Parts
The script detects and can pixelate:
- Female/Male genitalia (exposed/covered)
- Female breasts (exposed/covered)
- Buttocks (exposed/covered)
- Anus (exposed/covered)
- Feet (exposed/covered)
- Armpits (exposed/covered)

## Examples

### Before and After
```
Original Image → Detection → Pixelation
```

### Different Output Styles
1. **With Labels**: Shows detection info and confidence scores
2. **Pixelation Only**: Clean output for privacy protection
3. **Rectangles Only**: For analysis without pixelation

## Requirements

- OpenCV (`cv2`)
- NumPy
- PIL (Pillow)
- NudeNet (`nudenet`)

## Installation

```bash
pip install opencv-python numpy pillow nudenet
```

## Notes

- **Performance**: Pixelation adds minimal processing time
- **Quality**: Higher pixel sizes reduce image quality but increase privacy
- **Compatibility**: Works with all image formats supported by OpenCV
- **Memory**: No significant memory overhead for pixelation

## Troubleshooting

### No Detections Found
- Lower the confidence threshold (try 0.01-0.05)
- Enable enhanced detection
- Try sliding window approach

### Poor Pixelation Quality
- Increase pixel size for stronger effect
- Check if detection regions are valid
- Ensure image is not corrupted

### Performance Issues
- Reduce image resolution
- Disable sliding window for large images
- Use smaller pixel sizes 