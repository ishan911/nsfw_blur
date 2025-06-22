# Building CentOS Executable

This guide explains how to build a CentOS-compatible executable from your Mac.

## Prerequisites

1. **Docker Desktop** installed on your Mac
2. **Git** (optional, for version control)

## Quick Build

1. **Clone or navigate to your project directory:**
   ```bash
   cd /path/to/your/blurapi/project
   ```

2. **Make the build script executable:**
   ```bash
   chmod +x build.sh
   ```

3. **Run the build script:**
   ```bash
   ./build.sh
   ```

4. **Wait for completion** (this may take 10-20 minutes on first run)

## What the Build Process Does

1. **Creates a Docker container** with CentOS 7
2. **Installs all dependencies** (Python, OpenCV, ONNX Runtime, etc.)
3. **Uses PyInstaller** to create a single-file executable
4. **Extracts the executable** to your local directory as `blurapi-centos`

## Manual Build (Alternative)

If you prefer to run the steps manually:

```bash
# Build Docker image
docker build -t blurapi-builder .

# Create container
docker create --name blurapi-container blurapi-builder

# Extract executable
docker cp blurapi-container:/output/blurapi ./blurapi-centos

# Clean up
docker rm blurapi-container

# Make executable
chmod +x ./blurapi-centos
```

## Deploying to CentOS

1. **Copy the executable to your CentOS server:**
   ```bash
   scp blurapi-centos user@your-centos-server:/path/to/destination/
   ```

2. **Copy your model file:**
   ```bash
   scp -r models/ user@your-centos-server:/path/to/destination/
   ```

3. **On CentOS server, make it executable:**
   ```bash
   chmod +x blurapi-centos
   ```

4. **Test the executable:**
   ```bash
   ./blurapi-centos --help
   ```

## Usage on CentOS

```bash
# Process WordPress images
./blurapi-centos sliding-wordpress data/wordpress_images.json

# With debug output
./blurapi-centos sliding-wordpress data/wordpress_images.json --debug

# Process single image
./blurapi-centos single input.jpg output.jpg

# Process batch of images
./blurapi-centos batch input_folder/ output_folder/
```

## Troubleshooting

### Build Issues

1. **Docker not running:**
   ```bash
   # Start Docker Desktop
   open -a Docker
   ```

2. **Permission denied:**
   ```bash
   chmod +x build.sh
   ```

3. **Out of disk space:**
   ```bash
   # Clean Docker images
   docker system prune -a
   ```

### Runtime Issues on CentOS

1. **Missing libraries:**
   ```bash
   # Install basic libraries
   sudo yum install glibc libstdc++
   ```

2. **Permission denied:**
   ```bash
   chmod +x blurapi-centos
   ```

3. **Model not found:**
   ```bash
   # Ensure models/ directory exists with your ONNX file
   ls -la models/
   ```

## File Structure

After building, you should have:

```
your-project/
├── blurapi-centos          # The executable
├── models/
│   └── 640m.onnx          # Your model file
├── data/
│   └── wordpress_images.json
└── wp-content/
    └── uploads/
        └── backup/         # Backups will be created here
```

## Performance Notes

- **First run** may be slower as the executable extracts dependencies
- **Memory usage** will be higher than Python script (includes Python runtime)
- **File size** will be larger (~100-200MB) due to bundled dependencies

## Security Considerations

- The executable contains all dependencies and is self-contained
- No need to install Python or other dependencies on CentOS
- Consider the executable as a binary distribution of your code

## Alternative: Direct PyInstaller (Mac-only)

If you only need a Mac executable:

```bash
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller --onefile --add-data "models:models" main.py

# Result: dist/main
```

**Note:** This will only work on Mac, not CentOS. 