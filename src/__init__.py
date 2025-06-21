# src/__init__.py

# Version of the entire project
__version__ = '1.0.0'

# Import main components to make them available at the top level
from .blurrer import ImageBlurrer

# Define what should be available when someone does "from src import *"
__all__ = ['ImageBlurrer']