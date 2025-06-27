#!/usr/bin/env python3
"""
Batch Processing Script for JSON URLs

This script reads JSON URLs from a text file, processes them using the sliding-json command,
and removes them from the file once processing is complete.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def read_urls_from_file(file_path):
    """
    Read URLs from a text file, skipping empty lines and comments.
    
    Args:
        file_path (str): Path to the text file containing URLs
        
    Returns:
        list: List of URLs (stripped of whitespace)
    """
    urls = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Strip whitespace and skip empty lines
                line = line.strip()
                if not line:
                    continue
                
                # Skip comment lines (starting with #)
                if line.startswith('#'):
                    continue
                
                urls.append(line)
        
        print(f"Read {len(urls)} URLs from {file_path}")
        return urls
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def remove_url_from_file(file_path, url_to_remove):
    """
    Remove a specific URL from the file.
    
    Args:
        file_path (str): Path to the text file
        url_to_remove (str): URL to remove from the file
        
    Returns:
        bool: True if URL was removed, False otherwise
    """
    try:
        # Read all lines
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out the URL to remove (case-insensitive)
        original_count = len(lines)
        filtered_lines = []
        
        for line in lines:
            if line.strip().lower() != url_to_remove.lower():
                filtered_lines.append(line)
        
        # Write back the filtered lines
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
        
        removed_count = original_count - len(filtered_lines)
        if removed_count > 0:
            print(f"  Removed {removed_count} URL(s) from {file_path}")
            return True
        else:
            print(f"  Warning: URL not found in file to remove")
            return False
            
    except Exception as e:
        print(f"  Error removing URL from file: {e}")
        return False

def run_sliding_json(url, base_url=None, output_dir="processed_images", force=False, download_only=False):
    """
    Run the sliding-json command for a single URL.
    
    Args:
        url (str): JSON URL to process
        base_url (str): Base URL for relative paths
        output_dir (str): Output directory
        force (bool): Force reprocessing
        download_only (bool): Only download images
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Build the command
        cmd = [
            sys.executable, 'main.py', 'sliding-json',
            '--json-url', url
        ]
        
        if base_url:
            cmd.extend(['--base-url', base_url])
        
        if output_dir != "processed_images":
            cmd.extend(['--output-dir', output_dir])
        
        if force:
            cmd.append('--force')
        
        if download_only:
            cmd.append('--download-only')
        
        print(f"  Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"  Errors: {result.stderr}")
        
        # Check return code
        if result.returncode == 0:
            print(f"  ✅ Successfully processed: {url}")
            return True
        else:
            print(f"  ❌ Failed to process: {url} (return code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout processing: {url}")
        return False
    except Exception as e:
        print(f"  ❌ Error processing {url}: {e}")
        return False

def process_urls_batch(urls_file, base_url=None, output_dir="processed_images", force=False, 
                      download_only=False, delay=5, max_retries=3):
    """
    Process all URLs from the file in batch.
    
    Args:
        urls_file (str): Path to the text file containing URLs
        base_url (str): Base URL for relative paths
        output_dir (str): Output directory
        force (bool): Force reprocessing
        download_only (bool): Only download images
        delay (int): Delay between processing URLs (seconds)
        max_retries (int): Maximum number of retries for failed URLs
        
    Returns:
        dict: Processing summary
    """
    print(f"=== Batch Processing Started ===")
    print(f"URLs file: {urls_file}")
    print(f"Base URL: {base_url}")
    print(f"Output directory: {output_dir}")
    print(f"Force reprocessing: {force}")
    print(f"Download only: {download_only}")
    print(f"Delay between URLs: {delay} seconds")
    print(f"Max retries: {max_retries}")
    print()
    
    # Read URLs from file
    urls = read_urls_from_file(urls_file)
    if not urls:
        print("No URLs found to process.")
        return {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
    
    # Processing statistics
    total_urls = len(urls)
    successful = 0
    failed = 0
    skipped = 0
    
    print(f"Starting to process {total_urls} URLs...")
    print()
    
    # Process each URL
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{total_urls}] Processing: {url}")
        
        # Try to process the URL
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            if retry_count > 0:
                print(f"  Retry {retry_count}/{max_retries}...")
            
            success = run_sliding_json(
                url=url,
                base_url=base_url,
                output_dir=output_dir,
                force=force,
                download_only=download_only
            )
            
            if not success:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  Waiting {delay * 2} seconds before retry...")
                    time.sleep(delay * 2)
        
        # Handle result
        if success:
            successful += 1
            # Remove URL from file only if processing was successful
            remove_url_from_file(urls_file, url)
            
            # Show remaining URLs count
            remaining_urls = total_urls - successful
            if remaining_urls > 0:
                print(f"  📊 Progress: {successful}/{total_urls} completed, {remaining_urls} remaining")
            else:
                print(f"  🎉 All URLs processed successfully!")
        else:
            failed += 1
            print(f"  ❌ Failed after {max_retries} retries, keeping URL in file")
        
        # Add delay between URLs (except for the last one)
        if i < total_urls:
            print(f"  Waiting {delay} seconds before next URL...")
            time.sleep(delay)
        
        print()
    
    # Summary
    print(f"=== Batch Processing Summary ===")
    print(f"Total URLs: {total_urls}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} URL(s) failed and remain in the file for manual review.")
    
    return {
        'total': total_urls,
        'successful': successful,
        'failed': failed,
        'skipped': skipped
    }

def create_sample_urls_file(file_path):
    """
    Create a sample URLs file with example entries.
    
    Args:
        file_path (str): Path to create the sample file
    """
    sample_content = """# JSON URLs to process
# One URL per line
# Lines starting with # are comments and will be ignored
# Empty lines are also ignored

# Example URLs (replace with your actual URLs):
https://api.example.com/images1.json
https://api.example.com/images2.json
https://api.example.com/images3.json

# You can also use local files:
data/local_images.json
"""
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print(f"Created sample URLs file: {file_path}")
        print("Please edit this file with your actual JSON URLs.")
    except Exception as e:
        print(f"Error creating sample file: {e}")

def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='Batch process JSON URLs using sliding-json command',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process URLs from a file
  python batch_process.py urls.txt --base-url "https://your-domain.com"
  
  # Create a sample URLs file
  python batch_process.py --create-sample urls.txt
  
  # Process with custom settings
  python batch_process.py urls.txt --base-url "https://your-domain.com" --output-dir "my_output" --force --delay 10
        """
    )
    
    parser.add_argument('urls_file', nargs='?', help='Path to text file containing JSON URLs')
    parser.add_argument('--base-url', help='Base URL for converting relative paths to absolute URLs')
    parser.add_argument('--output-dir', default='processed_images', help='Output directory for processed images')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if output already exists')
    parser.add_argument('--download-only', action='store_true', help='Only download images, do not process them')
    parser.add_argument('--delay', type=int, default=5, help='Delay between processing URLs in seconds (default: 5)')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries for failed URLs (default: 3)')
    parser.add_argument('--create-sample', help='Create a sample URLs file with the specified name')
    
    args = parser.parse_args()
    
    # Handle create-sample option
    if args.create_sample:
        create_sample_urls_file(args.create_sample)
        return 0
    
    # Check if URLs file is provided
    if not args.urls_file:
        print("Error: Please provide a URLs file or use --create-sample to create one.")
        print("Use --help for more information.")
        return 1
    
    # Check if URLs file exists
    if not os.path.exists(args.urls_file):
        print(f"Error: URLs file not found: {args.urls_file}")
        print("Use --create-sample to create a sample file.")
        return 1
    
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("Error: main.py not found in current directory.")
        print("Please run this script from the directory containing main.py")
        return 1
    
    # Process URLs
    result = process_urls_batch(
        urls_file=args.urls_file,
        base_url=args.base_url,
        output_dir=args.output_dir,
        force=args.force,
        download_only=args.download_only,
        delay=args.delay,
        max_retries=args.max_retries
    )
    
    # Return appropriate exit code
    if result['failed'] > 0:
        print(f"\n⚠️  Some URLs failed. Check the file for remaining URLs.")
        return 1
    else:
        print(f"\n✅ All URLs processed successfully!")
        return 0

if __name__ == "__main__":
    exit(main()) 