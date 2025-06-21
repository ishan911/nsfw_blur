#!/usr/bin/env python3
"""
Database Viewer for Processed Images
This script helps you view and manage the processed images database.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

def load_database(database_path='data/processed_images.json'):
    """Load the processed images database."""
    try:
        if os.path.exists(database_path):
            with open(database_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Database not found: {database_path}")
            return {}
    except Exception as e:
        print(f"Error loading database: {e}")
        return {}

def show_summary(processed_images):
    """Show a summary of the database."""
    if not processed_images:
        print("No processed images found in database.")
        return
    
    total_images = len(processed_images)
    total_size = sum(record.get('file_size', 0) for record in processed_images.values())
    
    print(f"Database Summary:")
    print(f"  Total images: {total_images}")
    print(f"  Total size: {total_size / (1024*1024):.2f} MB")
    print(f"  Average size: {(total_size / total_images) / (1024*1024):.2f} MB per image")

def show_recent(processed_images, limit=10):
    """Show recent processing activity."""
    if not processed_images:
        return
    
    recent = sorted(
        processed_images.items(),
        key=lambda x: x[1]['processed_at'],
        reverse=True
    )[:limit]
    
    print(f"\nRecent Processing Activity (last {len(recent)}):")
    print("-" * 80)
    for i, (input_path, record) in enumerate(recent, 1):
        filename = os.path.basename(input_path)
        output_filename = os.path.basename(record['output_path'])
        processed_time = datetime.fromisoformat(record['processed_at']).strftime('%Y-%m-%d %H:%M:%S')
        file_size = record.get('file_size', 0) / (1024*1024)
        
        print(f"{i:2d}. {filename}")
        print(f"     Output: {output_filename}")
        print(f"     Size: {file_size:.2f} MB")
        print(f"     Pixel Size: {record['pixel_size']}")
        print(f"     Processed: {processed_time}")
        print()

def show_by_size(processed_images, limit=10):
    """Show images sorted by file size."""
    if not processed_images:
        return
    
    sorted_by_size = sorted(
        processed_images.items(),
        key=lambda x: x[1].get('file_size', 0),
        reverse=True
    )[:limit]
    
    print(f"\nLargest Images (top {len(sorted_by_size)}):")
    print("-" * 80)
    for i, (input_path, record) in enumerate(sorted_by_size, 1):
        filename = os.path.basename(input_path)
        file_size = record.get('file_size', 0) / (1024*1024)
        processed_time = datetime.fromisoformat(record['processed_at']).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{i:2d}. {filename} ({file_size:.2f} MB) - {processed_time}")

def show_orphaned_records(processed_images):
    """Show records for files that no longer exist."""
    orphaned = []
    for input_path in processed_images:
        if not os.path.exists(input_path):
            orphaned.append(input_path)
    
    if orphaned:
        print(f"\nOrphaned Records ({len(orphaned)} files no longer exist):")
        print("-" * 80)
        for i, input_path in enumerate(orphaned, 1):
            filename = os.path.basename(input_path)
            record = processed_images[input_path]
            processed_time = datetime.fromisoformat(record['processed_at']).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{i:2d}. {filename} (processed: {processed_time})")
    else:
        print("\nNo orphaned records found.")

def search_by_filename(processed_images, search_term):
    """Search for images by filename."""
    matches = []
    search_term_lower = search_term.lower()
    
    for input_path, record in processed_images.items():
        filename = os.path.basename(input_path)
        if search_term_lower in filename.lower():
            matches.append((input_path, record))
    
    if matches:
        print(f"\nSearch Results for '{search_term}' ({len(matches)} matches):")
        print("-" * 80)
        for i, (input_path, record) in enumerate(matches, 1):
            filename = os.path.basename(input_path)
            output_filename = os.path.basename(record['output_path'])
            processed_time = datetime.fromisoformat(record['processed_at']).strftime('%Y-%m-%d %H:%M:%S')
            file_size = record.get('file_size', 0) / (1024*1024)
            
            print(f"{i:2d}. {filename}")
            print(f"     Output: {output_filename}")
            print(f"     Size: {file_size:.2f} MB")
            print(f"     Processed: {processed_time}")
            print()
    else:
        print(f"\nNo matches found for '{search_term}'")

def export_list(processed_images, output_file):
    """Export the database as a simple list."""
    try:
        with open(output_file, 'w') as f:
            f.write("Processed Images Database Export\n")
            f.write("=" * 50 + "\n\n")
            
            for input_path, record in processed_images.items():
                filename = os.path.basename(input_path)
                output_filename = os.path.basename(record['output_path'])
                processed_time = datetime.fromisoformat(record['processed_at']).strftime('%Y-%m-%d %H:%M:%S')
                file_size = record.get('file_size', 0) / (1024*1024)
                
                f.write(f"Input: {filename}\n")
                f.write(f"Output: {output_filename}\n")
                f.write(f"Size: {file_size:.2f} MB\n")
                f.write(f"Pixel Size: {record['pixel_size']}\n")
                f.write(f"Processed: {processed_time}\n")
                f.write("-" * 30 + "\n")
        
        print(f"Database exported to: {output_file}")
    except Exception as e:
        print(f"Error exporting database: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Database Viewer for Processed Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --summary
  %(prog)s --recent 5
  %(prog)s --largest 10
  %(prog)s --search "example"
  %(prog)s --orphaned
  %(prog)s --export output.txt
        """
    )
    
    parser.add_argument(
        '--database',
        default='data/processed_images.json',
        help='Path to the database file (default: data/processed_images.json)'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show database summary'
    )
    
    parser.add_argument(
        '--recent',
        type=int,
        metavar='N',
        help='Show N most recent entries'
    )
    
    parser.add_argument(
        '--largest',
        type=int,
        metavar='N',
        help='Show N largest images'
    )
    
    parser.add_argument(
        '--search',
        metavar='TERM',
        help='Search for images containing TERM in filename'
    )
    
    parser.add_argument(
        '--orphaned',
        action='store_true',
        help='Show orphaned records (files that no longer exist)'
    )
    
    parser.add_argument(
        '--export',
        metavar='FILE',
        help='Export database to FILE'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Show all information (summary, recent, largest, orphaned)'
    )
    
    args = parser.parse_args()
    
    # Load database
    processed_images = load_database(args.database)
    
    if not processed_images:
        print("No data to display.")
        return
    
    # Show requested information
    if args.all or args.summary:
        show_summary(processed_images)
    
    if args.all or args.recent:
        limit = args.recent if args.recent else 10
        show_recent(processed_images, limit)
    
    if args.all or args.largest:
        limit = args.largest if args.largest else 10
        show_by_size(processed_images, limit)
    
    if args.all or args.orphaned:
        show_orphaned_records(processed_images)
    
    if args.search:
        search_by_filename(processed_images, args.search)
    
    if args.export:
        export_list(processed_images, args.export)
    
    # If no specific options, show summary and recent
    if not any([args.summary, args.recent, args.largest, args.orphaned, args.search, args.export, args.all]):
        show_summary(processed_images)
        show_recent(processed_images, 5)

if __name__ == "__main__":
    main() 