from PIL import Image, ImageFilter
from nudenet import NudeDetector

detector = NudeDetector(model_path="models/640m.onnx")

def convert_box_to_array( box_str):
        """Convert box string to array of coordinates."""
        # Remove any brackets and split by comma
        box_str = box_str.strip('[]()')
        # Split the string and convert each element to float
        return [float(x) for x in box_str.split(',')]

def blur_region(image, region, blur_radius=20):
        """Blur a specific region in the image."""
        x, y, w, h = region
        region_img = image.crop((x, y, x + w, y + h))
        blurred_region = region_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        image.paste(blurred_region, (x, y))
        return image

def pixelate_region(image, region, pixel_size=10):
        """Create a pixelated effect in a specific region of the image."""
        x, y, w, h = region
        
        # Add 10px extra to all 4 sides
        extra_padding = 10
        x_expanded = max(0, x - extra_padding)
        y_expanded = max(0, y - extra_padding)
        w_expanded = min(w + 2 * extra_padding, image.width - x_expanded)
        h_expanded = min(h + 2 * extra_padding, image.height - y_expanded)
        
        # Crop the expanded region
        region_img = image.crop((x_expanded, y_expanded, x_expanded + w_expanded, y_expanded + h_expanded))
        
        # Calculate new size (smaller size = more pixelated)
        small_w = max(w_expanded // pixel_size, 1)
        small_h = max(h_expanded // pixel_size, 1)
        
        # Resize down
        small_img = region_img.resize((small_w, small_h), Image.Resampling.NEAREST)
        
        # Resize back up
        pixelated_region = small_img.resize((w_expanded, h_expanded), Image.Resampling.NEAREST)
        
        # Paste the pixelated region back
        image.paste(pixelated_region, (x_expanded, y_expanded))
        return image


inputPath = "data/input/TheyAreHugePiercing_watermark.jpg"
outputPath = "output.png"

results = detector.detect(inputPath)
image = Image.open(inputPath)

areas = [
    #   'FEMALE_GENI
      'FEMALE_BREAST_EXPOSED',
      'FEMALE_GENITALIA_EXPOSED',
      'ANUS_EXPOSED',
      'MALE_GENITALIA_EXPOSED',
      'FEMALE_GENITALIA_EXPOSED',
]

print(results)

for result in results:
    if result['class'] in areas:
        print(result)
        if result['score'] > 0.1:
            # print(f"valid {result['class']} ${result['box']}")
            # image = blur_region(image, result['box'])
            image = pixelate_region(image, result['box'])

if outputPath:
    image.save(outputPath)

# detector.censor("images/2.png", output_path="output.png")