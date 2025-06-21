# JSON Fetcher Deployment Guide

This guide explains how to deploy and use the JSON fetching scripts on your web server.

## Files Overview

1. **`fetch_json_examples.py`** - Comprehensive Python script with all JSON fetching functions
2. **`fetch_json.sh`** - Shell script wrapper with command-line options
3. **`json_fetcher.py`** - Standalone Python script for easy deployment

## Deployment Options

### Option 1: Standalone Python Script (Recommended)

The `json_fetcher.py` script is designed for easy deployment:

```bash
# Make executable
chmod +x examples/json_fetcher.py

# Usage examples:
./json_fetcher.py get https://jsonplaceholder.typicode.com/posts/1
./json_fetcher.py file data/config.json
./json_fetcher.py post https://api.example.com/data '{"key":"value"}'
./json_fetcher.py save '{"name":"test"}' output.json
```

**Features:**
- No external dependencies (falls back to built-in urllib if requests not available)
- Simple command-line interface
- Error handling and validation
- Can save output to files

### Option 2: Shell Script Wrapper

The `fetch_json.sh` script provides a more feature-rich interface:

```bash
# Make executable
chmod +x examples/fetch_json.sh

# Usage examples:
./fetch_json.sh --url https://jsonplaceholder.typicode.com/posts/1
./fetch_json.sh --file data/config.json
./fetch_json.sh --post https://api.example.com/data --data '{"key":"value"}'
./fetch_json.sh --save output.json --data '{"name":"test"}'
```

**Features:**
- Automatic dependency checking and installation
- Support for custom headers
- More detailed error messages
- Runs all examples if no arguments provided

### Option 3: Direct Python Execution

Run the comprehensive script directly:

```bash
python3 examples/fetch_json_examples.py
```

## Web Server Integration

### 1. CGI Script Setup

To use as a CGI script on Apache:

```apache
# In your Apache configuration or .htaccess
AddHandler cgi-script .py
Options +ExecCGI
```

Then place the script in your CGI directory and access via URL.

### 2. Cron Job Integration

Add to crontab for automated JSON fetching:

```bash
# Fetch JSON every hour and save to file
0 * * * * /path/to/json_fetcher.py get https://api.example.com/data /var/www/data.json

# Process JSON file daily
0 0 * * * /path/to/json_fetcher.py file /var/www/data.json > /var/log/json_processing.log
```

### 3. Web Application Integration

Import functions in your web application:

```python
import sys
sys.path.append('/path/to/examples')
from fetch_json_examples import fetch_json_from_url, save_json_to_file

# Use in your web app
data = fetch_json_from_url('https://api.example.com/data')
if data:
    save_json_to_file(data, '/var/www/cache/data.json')
```

## Security Considerations

1. **Input Validation**: Always validate URLs and file paths
2. **File Permissions**: Set appropriate permissions on output files
3. **Error Handling**: Implement proper error handling in production
4. **Rate Limiting**: Add delays between requests to avoid overwhelming APIs
5. **Authentication**: Use environment variables for API keys

## Environment Setup

### Minimal Requirements

```bash
# Install Python 3 (if not already installed)
sudo apt-get install python3 python3-pip  # Ubuntu/Debian
sudo yum install python3 python3-pip      # CentOS/RHEL

# Install requests library (optional, script will work without it)
pip3 install requests
```

### Production Recommendations

```bash
# Create virtual environment
python3 -m venv /opt/json_fetcher
source /opt/json_fetcher/bin/activate

# Install dependencies
pip install requests

# Set up logging
mkdir -p /var/log/json_fetcher
chown www-data:www-data /var/log/json_fetcher
```

## Troubleshooting

### Common Issues

1. **Permission Denied**: Make sure scripts are executable
   ```bash
   chmod +x *.py *.sh
   ```

2. **Python Not Found**: Ensure python3 is in PATH
   ```bash
   which python3
   ```

3. **Import Errors**: Install missing dependencies
   ```bash
   pip3 install requests
   ```

4. **Network Issues**: Check firewall and proxy settings

### Debug Mode

Add debug output to scripts:

```bash
# For shell script
bash -x ./fetch_json.sh --url https://example.com

# For Python script
python3 -v json_fetcher.py get https://example.com
```

## Examples for Common Use Cases

### 1. API Data Synchronization

```bash
# Fetch data from API and save to database directory
./json_fetcher.py get https://api.example.com/users /var/www/db/users.json
```

### 2. Configuration Management

```bash
# Load configuration from file
./json_fetcher.py file /etc/app/config.json
```

### 3. Webhook Processing

```bash
# Send data to webhook endpoint
./json_fetcher.py post https://webhook.example.com '{"event":"update","data":"value"}'
```

### 4. Data Backup

```bash
# Save current data to backup file
./json_fetcher.py save '{"timestamp":"2024-01-01","data":"backup"}' /backup/data_$(date +%Y%m%d).json
``` 