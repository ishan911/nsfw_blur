from ultralytics import YOLO
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolov8n.pt")

# Training with comprehensive data augmentation
model.train(
    data="train.yaml", 
    epochs=50, 
    imgsz=1280, 
    device=device, 
    batch=4,
    
    # Data Augmentation Parameters
    augment=True,  # Enable augmentation
    
    # Geometric Augmentations
    degrees=10.0,      # Image rotation (+/- degrees)
    translate=0.1,     # Image translation (+/- fraction)
    scale=0.5,         # Image scaling (+/- gain)
    shear=2.0,         # Image shear (+/- degrees)
    perspective=0.0,   # Image perspective (+/- fraction), range 0-0.001
    flipud=0.0,        # Image flip up-down (probability)
    fliplr=0.5,        # Image flip left-right (probability)
    mosaic=1.0,        # Image mosaic (probability)
    mixup=0.0,         # Image mixup (probability)
    copy_paste=0.0,    # Segment copy-paste (probability)
    
    # Color Augmentations
    hsv_h=0.015,       # Image HSV-Hue augmentation (fraction)
    hsv_s=0.7,         # Image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,         # Image HSV-Value augmentation (fraction)
    auto_augment='randaugment',  # Auto-augment policy (randaugment, autoaugment, augmix)
    
    # Additional Augmentations
    erasing=0.0,       # Random erasing (probability)
    crop_fraction=1.0, # Image crop fraction (for training)
    
    # Training Parameters
    patience=50,       # Early stopping patience
    save=True,         # Save checkpoints
    save_period=10,    # Save checkpoint every x epochs
    verbose=True,      # Verbose output
    plots=True,        # Save plots
)