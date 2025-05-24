"""
AWS utility functions for pipeline transforms.
"""

import os
import hashlib
import logging
import boto3
import dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple, Union

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

logger = logging.getLogger("pipeline.utils.aws")


def get_aws_credentials():
    """Get AWS credentials from environment variables.
    
    Returns:
        Dict with AWS credentials
    """
    credentials = {
        'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
        'region_name': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    }
    
    # Check if credentials are available
    if not credentials['aws_access_key_id'] or not credentials['aws_secret_access_key']:
        logger.warning("AWS credentials not found in environment variables")
    
    return credentials


def init_s3_client():
    """Initialize S3 client using environment credentials.
    
    Returns:
        Boto3 S3 client
    """
    credentials = get_aws_credentials()
    return boto3.client('s3', **credentials)


def init_dynamodb_resource():
    """Initialize DynamoDB resource using environment credentials.
    
    Returns:
        Boto3 DynamoDB resource
    """
    credentials = get_aws_credentials()
    return boto3.resource('dynamodb', **credentials)


def init_dynamodb_client():
    """Initialize DynamoDB client using environment credentials.
    
    Returns:
        Boto3 DynamoDB client
    """
    credentials = get_aws_credentials()
    return boto3.client('dynamodb', **credentials)


def ensure_dynamodb_table_exists(dynamodb_client, table_name: str, key_schema: List[Dict[str, Any]], 
                                attribute_definitions: List[Dict[str, Any]], 
                                gsis: Optional[List[Dict[str, Any]]] = None):
    """Create a DynamoDB table if it doesn't exist.
    
    Args:
        dynamodb_client: DynamoDB client
        table_name: Name of the table to check/create
        key_schema: KeySchema for the table
        attribute_definitions: AttributeDefinitions for the table
        gsis: Optional list of GlobalSecondaryIndexes
    """
    try:
        # Check if table exists
        dynamodb_client.describe_table(TableName=table_name)
        logger.info(f"DynamoDB table '{table_name}' already exists.")
    except dynamodb_client.exceptions.ResourceNotFoundException:
        # Create table if it doesn't exist
        logger.info(f"Creating DynamoDB table '{table_name}'...")
        
        create_args = {
            'TableName': table_name,
            'KeySchema': key_schema,
            'AttributeDefinitions': attribute_definitions,
            'ProvisionedThroughput': {
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        }
        
        # Add GSIs if provided
        if gsis:
            create_args['GlobalSecondaryIndexes'] = gsis
            
        table = dynamodb_client.create_table(**create_args)
        
        # Wait for table creation to complete
        logger.info(f"Waiting for table '{table_name}' creation to complete...")
        waiter = dynamodb_client.get_waiter('table_exists')
        waiter.wait(TableName=table_name)
        logger.info(f"Table '{table_name}' created successfully.")


def ensure_pipeline_table_exists(dynamodb_client, table_name: str = 'conduit-pipeline-metadata'):
    """Create the pipeline metadata table if it doesn't exist.
    
    Args:
        dynamodb_client: DynamoDB client
        table_name: Name of the table to check/create
    """
    key_schema = [
        {'AttributeName': 'data_id', 'KeyType': 'HASH'},
        {'AttributeName': 'transform_id', 'KeyType': 'RANGE'}
    ]
    attribute_definitions = [
        {'AttributeName': 'data_id', 'AttributeType': 'S'},
        {'AttributeName': 'transform_id', 'AttributeType': 'S'},
        {'AttributeName': 'status', 'AttributeType': 'S'},
        {'AttributeName': 'timestamp', 'AttributeType': 'S'},
    ]
    gsis = [
        {
            'IndexName': 'TransformIndex',
            'KeySchema': [
                {'AttributeName': 'transform_id', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
            ],
            'Projection': {
                'ProjectionType': 'ALL'
            },
            'ProvisionedThroughput': {
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        },
        {
            'IndexName': 'StatusIndex',
            'KeySchema': [
                {'AttributeName': 'status', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
            ],
            'Projection': {
                'ProjectionType': 'ALL'
            },
            'ProvisionedThroughput': {
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        }
    ]
    
    ensure_dynamodb_table_exists(dynamodb_client, table_name, key_schema, attribute_definitions, gsis)


def ensure_script_versions_table_exists(dynamodb_client, table_name: str = 'conduit-script-versions'):
    """Create the script versions table if it doesn't exist.
    
    Args:
        dynamodb_client: DynamoDB client
        table_name: Name of the table to check/create
    """
    key_schema = [
        {'AttributeName': 'script_id', 'KeyType': 'HASH'},
        {'AttributeName': 'script_version', 'KeyType': 'RANGE'}
    ]
    attribute_definitions = [
        {'AttributeName': 'script_id', 'AttributeType': 'S'},
        {'AttributeName': 'script_version', 'AttributeType': 'S'},
    ]
    
    ensure_dynamodb_table_exists(dynamodb_client, table_name, key_schema, attribute_definitions)


def verify_script_md5(s3_client, bucket: str, key: str, expected_md5: str) -> bool:
    """Verify that a script in S3 matches the expected MD5 hash.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        expected_md5: Expected MD5 hash of the script
        
    Returns:
        Boolean indicating if the MD5 hash matches
    """
    try:
        # Get object metadata (includes ETag which is MD5 for small objects)
        response = s3_client.head_object(Bucket=bucket, Key=key)
        
        # Sometimes we need to get the actual content and compute the hash
        # especially if the object was uploaded with server-side encryption
        get_response = s3_client.get_object(Bucket=bucket, Key=key)
        content = get_response['Body'].read()
        actual_md5 = hashlib.md5(content).hexdigest()
        
        # Compare hashes
        if actual_md5 == expected_md5:
            logger.debug(f"MD5 hash verified for s3://{bucket}/{key}")
            return True
        else:
            logger.warning(f"MD5 mismatch for s3://{bucket}/{key}")
            logger.debug(f"Expected: {expected_md5}, Actual: {actual_md5}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying MD5 hash: {e}")
        return False


def check_script_version_exists(dynamodb, script_table_name: str, script_id: str, script_version: str) -> Tuple[bool, Optional[Dict]]:
    """Check if a script version exists in DynamoDB.
    
    Args:
        dynamodb: DynamoDB resource
        script_table_name: Name of the script versions table
        script_id: Script ID to check
        script_version: Script version to check
        
    Returns:
        Tuple of (bool indicating if version exists, the item if it exists)
    """
    try:
        script_table = dynamodb.Table(script_table_name)
        response = script_table.get_item(
            Key={
                'script_id': script_id,
                'script_version': script_version
            }
        )
        
        exists = 'Item' in response
        item = response.get('Item', None)
        
        return exists, item
    except Exception as e:
        logger.error(f"Error checking script version: {e}")
        # If we can't check, assume it doesn't exist
        return False, None


def upload_script_with_md5_verification(s3_client, dynamodb_resource, 
                                     script_path: str, script_id: str, script_name: str, script_version: str,
                                     s3_bucket: str, script_prefix: str, script_table_name: str,
                                     dry_run: bool = False) -> Tuple[str, bool]:
    """Upload a script to S3 only if the version doesn't exist or MD5 hash has changed.
    
    Args:
        s3_client: Boto3 S3 client
        dynamodb_resource: DynamoDB resource
        script_path: Path to the script file
        script_id: Script ID (e.g., "0A")
        script_name: Script name (e.g., "curate_sessions")
        script_version: Script version (e.g., "v0")
        s3_bucket: S3 bucket name
        script_prefix: S3 prefix for scripts
        script_table_name: DynamoDB table name for script versions
        dry_run: If True, don't actually upload or write to DynamoDB
        
    Returns:
        Tuple of (script S3 path, boolean indicating if upload was performed)
    """
    # Check if script version exists in DynamoDB
    script_exists, existing_item = check_script_version_exists(
        dynamodb_resource, script_table_name, script_id, script_version
    )
    
    # Define S3 key and path
    s3_script_key = f"{script_prefix}{script_id}_{script_name}_{script_version}.py"
    s3_path = f"s3://{s3_bucket}/{s3_script_key}"
    
    # If script doesn't exist, upload it
    if not script_exists:
        if dry_run:
            logger.info(f"[DRY RUN] Would upload script to {s3_path}")
            return s3_path, False
            
        # Read the file content
        with open(script_path, 'rb') as script_file:
            script_content = script_file.read()
            
        # Calculate MD5 hash
        md5_hash = hashlib.md5(script_content).hexdigest()
            
        # Upload to S3
        logger.info(f"Uploading script to {s3_path}")
        s3_client.put_object(
            Bucket=s3_bucket, 
            Key=s3_script_key, 
            Body=script_content
        )

        # Record in script versions table
        script_table = dynamodb_resource.Table(script_table_name)
        script_table.put_item(Item={
            'script_id': script_id,
            'script_version': script_version,
            's3_path': s3_path,
            'upload_date': datetime.now().isoformat(),
            'md5_hash': md5_hash
        })
        
        logger.info(f"Recorded script version in DynamoDB")
        return s3_path, True
        
    # Script exists, verify MD5 hash
    elif existing_item and 'md5_hash' in existing_item:
        expected_md5 = existing_item['md5_hash']
        s3_key = existing_item['s3_path'].replace(f"s3://{s3_bucket}/", "")
        
        # Read the current script file to get its MD5
        with open(script_path, 'rb') as script_file:
            script_content = script_file.read()
            
        current_md5 = hashlib.md5(script_content).hexdigest()
        
        if current_md5 == expected_md5:
            # MD5 matches the recorded version
            logger.info(f"Script version {script_id} {script_version} already exists with matching MD5, reusing")
            return existing_item['s3_path'], False
        
        # MD5 differs, verify against S3 object
        s3_matches = verify_script_md5(s3_client, s3_bucket, s3_key, expected_md5)
        
        if not s3_matches:
            logger.warning(f"Script in S3 does not match recorded MD5 hash")
            
        if current_md5 != expected_md5:
            logger.warning(f"Current script content differs from recorded version (MD5 mismatch)")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would update script at {s3_path} with new content")
                return s3_path, False
                
            # Update the script in S3
            logger.info(f"Updating script at {s3_path} with new content")
            s3_client.put_object(
                Bucket=s3_bucket, 
                Key=s3_script_key, 
                Body=script_content
            )
            
            # Update the MD5 hash in DynamoDB
            script_table = dynamodb_resource.Table(script_table_name)
            script_table.update_item(
                Key={
                    'script_id': script_id,
                    'script_version': script_version
                },
                UpdateExpression="set md5_hash = :m, upload_date = :d",
                ExpressionAttributeValues={
                    ':m': current_md5,
                    ':d': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Updated script version with new MD5 hash")
            return s3_path, True
            
    # Script exists but doesn't have MD5 hash
    else:
        logger.info(f"Script version {script_id} {script_version} already exists, reusing S3 path")
        return existing_item['s3_path'], False