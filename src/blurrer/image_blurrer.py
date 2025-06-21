import os
from PIL import Image, ImageFilter
from nudenet import NudeDetector

class ImageBlurrer:
    def __init__(self, model_path="models/640m.onnx", parts = []):
        print("Initializing ImageBlurrer")
        self.detector = NudeDetector(model_path=model_path)
        self.parts = parts

    def blur_region(self, image, region, blur_radius=20):
        """Blur a specific region in the image."""
        x, y, w, h = region
        region_img = image.crop((x, y, x + w, y + h))
        blurred_region = region_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        image.paste(blurred_region, (x, y))
        return image

    def pixelate_region(self, image, region, pixel_size=10):
        """Create a pixelated effect in a specific region of the image."""
        x, y, w, h = region
        
        # Crop the region
        region_img = image.crop((x, y, x + w, y + h))
        
        # Calculate new size (smaller size = more pixelated)
        small_w = max(w // pixel_size, 1)
        small_h = max(h // pixel_size, 1)
        
        # Resize down
        small_img = region_img.resize((small_w, small_h), Image.Resampling.NEAREST)
        
        # Resize back up
        pixelated_region = small_img.resize((w, h), Image.Resampling.NEAREST)
        
        # Paste the pixelated region back
        image.paste(pixelated_region, (x, y))
        return image

    def process_image(self, input_path, output_path=None, pixel_size=10):
        """Process an image to detect and pixelate sensitive content."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
            
        # Load and process image
        image = Image.open(input_path)

        # Detect sensitive content
        results = self.detector.detect(input_path)

        for result in results:
            if not self.parts or result['class'] in self.parts:
                if result['score'] > 0.1:
                    image = self.pixelate_region(image, result['box'], pixel_size)

        if output_path:
            image.save(output_path)

        return image
