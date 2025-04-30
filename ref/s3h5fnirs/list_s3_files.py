#!/usr/bin/env python3
"""
Script to list H5 files in the S3 bucket.

This script:
1. Lists all H5 files in the specified S3 prefix
2. Displays key information about each file

Usage:
python list_s3_files.py [--prefix PREFIX] [--limit N]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("list_s3_files")


def list_s3_files(bucket, prefix='curated-h5/', limit=None, name_filter=None):
    """List files in an S3 bucket.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix to search
        limit: Maximum number of files to list
        name_filter: Optional string to filter file names by
        
    Returns:
        List of file information dictionaries
    """
    # Add parent directory to path to import utils
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from utils.aws import init_s3_client
    
    logger.info(f"Listing files in s3://{bucket}/{prefix}")
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # List files
        paginator = s3_client.get_paginator('list_objects_v2')
        files = []
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Get file name from key
                    key = obj['Key']
                    file_name = key.split('/')[-1]
                    
                    # Apply name filter if provided
                    if name_filter and name_filter.lower() not in file_name.lower():
                        continue
                    
                    # Collect file info
                    file_info = {
                        'key': key,
                        'name': file_name,
                        'size_mb': obj['Size'] / (1024 * 1024),
                        'last_modified': obj['LastModified']
                    }
                    files.append(file_info)
                    
                    # Check limit
                    if limit and len(files) >= limit:
                        break
            
            if limit and len(files) >= limit:
                break
        
        return files
    
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        return []


def print_file_list(files):
    """Print a formatted list of files.
    
    Args:
        files: List of file information dictionaries
    """
    print("\n" + "="*100)
    print(f"S3 FILES LISTING")
    print("="*100)
    
    if not files:
        print("No files found")
        return
    
    print(f"Found {len(files)} files")
    print("\n{:<60} {:<10} {:<30}".format("File Name", "Size (MB)", "Last Modified"))
    print("-"*100)
    
    for file_info in files:
        name = file_info['name']
        size_mb = file_info['size_mb']
        last_modified = file_info['last_modified'].strftime("%Y-%m-%d %H:%M:%S")
        
        print("{:<60} {:<10.2f} {:<30}".format(name, size_mb, last_modified))
    
    print("="*100)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="List files in an S3 bucket")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--prefix", default="curated-h5/", help="S3 prefix to search")
    parser.add_argument("--limit", type=int, help="Maximum number of files to list")
    parser.add_argument("--filter", help="String to filter file names by")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # List files
    files = list_s3_files(args.bucket, args.prefix, args.limit, args.filter)
    
    # Print results
    print_file_list(files)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())