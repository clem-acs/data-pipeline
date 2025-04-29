#!/usr/bin/env python3
"""
Script to test S3 connection and count sessions available in the bucket.
"""
import os
import sys
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the sys.path to import from the root package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.aws import get_aws_credentials

def test_s3_connection(bucket_name="conduit-data-dev", prefix="data-collector/new-sessions/"):
    """
    Test connection to S3 and count the number of sessions available.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): Path prefix to filter objects
        
    Returns:
        int: Number of unique sessions
    """
    # Get AWS credentials from environment variables
    print("Retrieving AWS credentials...")
    aws_credentials = get_aws_credentials()
    
    if not aws_credentials["aws_access_key_id"] or not aws_credentials["aws_secret_access_key"]:
        print("ERROR: AWS credentials not found in environment variables.")
        return None
    
    print("Initializing S3 client...")
    try:
        # Create S3 client with credentials
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_credentials["aws_access_key_id"],
            aws_secret_access_key=aws_credentials["aws_secret_access_key"],
            region_name="us-east-1"  # Default region, change if needed
        )
        
        # Test connection by listing buckets
        response = s3_client.list_buckets()
        print(f"Connection successful! Found {len(response['Buckets'])} buckets.")
        
        # Verify bucket exists
        if bucket_name not in [b['Name'] for b in response['Buckets']]:
            print(f"WARNING: Bucket '{bucket_name}' not found in your account.")
            return None
            
        # List all objects in the bucket with the given prefix
        print(f"Fetching sessions from s3://{bucket_name}/{prefix}...")
        
        # Use paginator for handling large numbers of objects
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # Store unique session IDs
        session_ids = set()
        total_objects = 0
        
        # Process each page of results
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                total_objects += len(page['Contents'])
                
                # Extract session IDs from object keys
                for obj in page['Contents']:
                    key = obj['Key']
                    # Assuming directory structure: prefix/session_id/...
                    parts = key.split('/')
                    if len(parts) > 2:
                        session_id = parts[2]  # Get the session ID from the path
                        session_ids.add(session_id)
        
        # Print results
        print(f"Total objects: {total_objects}")
        print(f"Total unique sessions: {len(session_ids)}")
        
        return len(session_ids)
        
    except Exception as e:
        print(f"ERROR: Failed to connect to S3 or fetch data: {e}")
        return None

if __name__ == "__main__":
    test_s3_connection()