# Changelog

## [Latest Update] - Blog Images JSON Format & Backup Functionality

### New Features

#### 1. Updated Blog Images JSON Format Support
- **Enhanced JSON parsing** to handle the new WordPress image sizes format
- **New structure support**: 
  ```json
  [
    {
      "slug": "blog-post-slug",
      "images": {
        "thumbnail": "url",
        "medium": "url",
        "large": "url",
        "full": "url",
        // ... other WordPress sizes
      }
    }
  ]
  ```

#### 2. WordPress Image Sizes Integration
- **Maintains WordPress naming patterns** for all image sizes
- **Preserves folder structure**: `wp-content/uploads/blog-images/`
- **Supports all WordPress image sizes**:
  - thumbnail, medium, medium_large, large
  - 1536x1536, 2048x2048
  - blog-tn, category-thumb, category-mobile-thumb
  - thumb-200, thumb-280, thumb-300
  - swiper-desktop, swiper-mobile
  - banner-logo, post-thumbnail, full

#### 3. Backup Functionality for All Commands
- **Automatic backup creation** for all original images
- **Organized backup structure** by command type:
  - `backup/blogs/` - Blog images backups
  - `backup/category/` - Category thumbnail backups  
  - `backup/sliding/` - Sliding JSON image backups
  - `backup/single/` - Single image backups
- **Backup naming pattern**: `{slug}_{size_name}_{original_filename}`
- **Commands with backup support**:
  - `blog-images` - Backs up all WordPress size images in `backup/blogs/`
  - `sliding-json` - Backs up screenshot and review images in `backup/sliding/`
  - `category-thumbnails` - Backs up category thumbnail images in `backup/category/`
  - `sliding-single` - Backs up single processed images in `backup/single/`

### Technical Improvements

#### 1. Enhanced Error Handling
- **Better error messages** for JSON parsing failures
- **Graceful handling** of missing or malformed JSON data
- **Improved logging** with detailed processing information

#### 2. Database Integration
- **Enhanced tracking** of processed images with WordPress size information
- **Better statistics** showing processing results by image type
- **Improved database records** with size-specific metadata

#### 3. File Organization
- **Consistent naming patterns** across all commands
- **WordPress-compatible folder structure**
- **Organized backup system** with command-specific folders

### Usage Examples

#### Blog Images Command
```bash
python main.py blog-images --json-url "https://example.com/blog-images.json" --base-url "https://www.mrporngeek.com"
```

#### Output Structure
```
processed_images/
├── backup/
│   ├── blogs/
│   │   ├── why-ai-porn-chatbots-are-the-internets-new-obsession_thumbnail_ai-porn-chatbots-featured-image-150x150.jpg
│   │   ├── why-ai-porn-chatbots-are-the-internets-new-obsession_medium_ai-porn-chatbots-featured-image-280x82.jpg
│   │   └── ...
│   ├── category/
│   │   ├── category-slug_category_thumb_thumbnail.jpg
│   │   └── ...
│   ├── sliding/
│   │   ├── slug_screenshot_full_url_screenshot.jpg
│   │   └── ...
│   └── single/
│       ├── screenshot_full_url_single-image.jpg
│       └── ...
└── wp-content/
    └── uploads/
        └── blog-images/
            ├── why-ai-porn-chatbots-are-the-internets-new-obsession_thumbnail.jpg
            ├── why-ai-porn-chatbots-are-the-internets-new-obsession_medium.jpg
            └── ...
```

### Backward Compatibility
- **Maintains support** for previous JSON formats
- **Automatic format detection** and parsing
- **No breaking changes** to existing functionality

### Testing
- **Verified JSON parsing** with sample data
- **Confirmed backup functionality** across all commands
- **Tested WordPress size handling** with various image formats 