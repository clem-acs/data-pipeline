#!/usr/bin/env python3

import argparse
import logging
import sys
import hashlib
import json
import os
from datetime import datetime

try:
    # When running as an installed package
    # Assuming utils is in the same parent directory or installed
    from utils.logging import setup_logging
    from utils.aws import init_dynamodb_client, init_dynamodb_resource, \
                          ensure_pipeline_table_exists, ensure_script_versions_table_exists
except ImportError:
    # When running as a script from the root directory
    # This allows direct execution if 'utils' is a subdirectory
    sys.path.append('.') # Add current directory to path to find utils
    from utils.logging import setup_logging
    from utils.aws import init_dynamodb_client, init_dynamodb_resource, \
                          ensure_pipeline_table_exists, ensure_script_versions_table_exists

logger = logging.getLogger(__name__)

# Generic DynamoDB operation handlers

def handle_put_item(args):
    """Handles the put-item command."""
    logger.info(f"Putting item into table '{args.table_name}' with item JSON: {args.item_json}")
    try:
        item_dict = json.loads(args.item_json)
        dynamodb = init_dynamodb_resource(endpoint_url=args.dynamodb_endpoint)
        table = dynamodb.Table(args.table_name)
        table.put_item(Item=item_dict)
        logger.info(f"Successfully put item into table '{args.table_name}'.")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON string for item: {args.item_json}", exc_info=args.verbose)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error putting item into table '{args.table_name}': {e}", exc_info=args.verbose)
        sys.exit(1)

def handle_get_item(args):
    """Handles the get-item command."""
    logger.info(f"Getting item from table '{args.table_name}' with key JSON: {args.key_json}")
    try:
        key_dict = json.loads(args.key_json)
        dynamodb = init_dynamodb_resource(endpoint_url=args.dynamodb_endpoint)
        table = dynamodb.Table(args.table_name)
        response = table.get_item(Key=key_dict)
        item = response.get('Item')
        if item:
            logger.info(f"Item found in table '{args.table_name}':")
            print(json.dumps(item, indent=2, default=str))
        else:
            logger.info(f"Item not found in table '{args.table_name}' with key {args.key_json}.")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON string for key: {args.key_json}", exc_info=args.verbose)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error getting item from table '{args.table_name}': {e}", exc_info=args.verbose)
        sys.exit(1)

def handle_scan_table(args):
    """Handles the scan-table command."""
    logger.info(f"Scanning table '{args.table_name}'")
    try:
        dynamodb = init_dynamodb_resource(endpoint_url=args.dynamodb_endpoint)
        table = dynamodb.Table(args.table_name)
        response = table.scan()
        items = response.get('Items', [])
        logger.info(f"Found {len(items)} items in table '{args.table_name}':")
        if items:
            print(json.dumps(items, indent=2, default=str))
        else:
            logger.info(f"No items found in table '{args.table_name}'.")
        # Note: For larger tables, pagination should be handled here.
        # Example: while 'LastEvaluatedKey' in response: response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey']) ...
    except Exception as e:
        logger.error(f"Error scanning table '{args.table_name}': {e}", exc_info=args.verbose)
        sys.exit(1)

def handle_delete_item(args):
    """Handles the delete-item command."""
    logger.info(f"Deleting item from table '{args.table_name}' with key JSON: {args.key_json}")
    try:
        key_dict = json.loads(args.key_json)
        dynamodb = init_dynamodb_resource(endpoint_url=args.dynamodb_endpoint)
        table = dynamodb.Table(args.table_name)
        table.delete_item(Key=key_dict)
        logger.info(f"Successfully deleted item with key {args.key_json} from table '{args.table_name}'.")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON string for key: {args.key_json}", exc_info=args.verbose)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error deleting item from table '{args.table_name}': {e}", exc_info=args.verbose)
        sys.exit(1)

def handle_create_table(args):
    """Handles the create-table command."""
    logger.info(f"Creating table '{args.table_name}'")
    try:
        key_schema = json.loads(args.key_schema)
        attribute_definitions = json.loads(args.attribute_definitions)
        provisioned_throughput = json.loads(args.provisioned_throughput)

        client = init_dynamodb_client(endpoint_url=args.dynamodb_endpoint)
        
        logger.debug(f"Creating table with params: TableName={args.table_name}, KeySchema={key_schema}, AttributeDefinitions={attribute_definitions}, ProvisionedThroughput={provisioned_throughput}")
        
        client.create_table(
            TableName=args.table_name,
            KeySchema=key_schema,
            AttributeDefinitions=attribute_definitions,
            ProvisionedThroughput=provisioned_throughput
        )
        
        logger.info(f"Waiting for table '{args.table_name}' to be created...")
        waiter = client.get_waiter('table_exists')
        waiter.wait(TableName=args.table_name)
        logger.info(f"Table '{args.table_name}' created successfully.")
        
    except json.JSONDecodeError as je:
        logger.error(f"Invalid JSON string provided for create-table arguments: {je}", exc_info=args.verbose)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating table '{args.table_name}': {e}", exc_info=args.verbose)
        sys.exit(1)

# Script and DB Init handlers (from previous steps)
def handle_register_script(args):
    """Handles the register-script command."""
    logger.info(f"Registering script: ID={args.script_id}, Version={args.script_version}, Name={args.script_name}, Path={args.script_path}")

    if not os.path.exists(args.script_path):
        logger.error(f"Script path does not exist: {args.script_path}")
        sys.exit(1)

    try:
        with open(args.script_path, 'rb') as f:
            content = f.read()
        md5_hash = hashlib.md5(content).hexdigest()
        logger.debug(f"Calculated MD5 hash: {md5_hash}")

        dynamodb = init_dynamodb_resource(endpoint_url=args.dynamodb_endpoint)
        table = dynamodb.Table('conduit-script-versions')

        item = {
            'script_id': args.script_id,
            'script_version': args.script_version,
            's3_path': args.script_path,  # Using local path for 's3_path' as specified
            'upload_date': datetime.now().isoformat(),
            'md5_hash': md5_hash,
            'script_name': args.script_name
        }

        table.put_item(Item=item)
        logger.info(f"Successfully registered script '{args.script_name}' (ID: {args.script_id}, Version: {args.script_version}) "
                    f"with MD5 hash {md5_hash} from path {args.script_path}.")

    except FileNotFoundError:
        logger.error(f"File not found during script registration: {args.script_path}", exc_info=args.verbose)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during script registration: {e}", exc_info=args.verbose)
        sys.exit(1)

def handle_check_script(args):
    """Handles the check-script command."""
    logger.info(f"Checking script: ID={args.script_id}, Version={args.script_version}")

    try:
        dynamodb = init_dynamodb_resource(endpoint_url=args.dynamodb_endpoint)
        table = dynamodb.Table('conduit-script-versions')

        response = table.get_item(
            Key={
                'script_id': args.script_id,
                'script_version': args.script_version
            }
        )

        item = response.get('Item')
        if item:
            logger.info("Script found.")
            # Pretty print the JSON item
            print(json.dumps(item, indent=2, default=str)) # use default=str for datetime if not natively serializable
        else:
            logger.info("Script not found.")
            
    except Exception as e:
        logger.error(f"Error during script check: {e}", exc_info=args.verbose)
        sys.exit(1)

def handle_init_local_db(args):
    """Handles the init-local-db command."""
    logger.info(f"Starting local DynamoDB initialization using endpoint: {args.dynamodb_endpoint}")
    
    try:
        dynamodb_client = init_dynamodb_client(endpoint_url=args.dynamodb_endpoint)
        logger.debug("DynamoDB client initialized.")

        logger.info("Ensuring 'conduit-pipeline-metadata' table exists...")
        ensure_pipeline_table_exists(dynamodb_client, table_name='conduit-pipeline-metadata')
        # ensure_pipeline_table_exists already logs success, so no extra logging here unless it fails
        logger.info("'conduit-pipeline-metadata' table check complete.")

        logger.info("Ensuring 'conduit-script-versions' table exists...")
        ensure_script_versions_table_exists(dynamodb_client, table_name='conduit-script-versions')
        logger.info("'conduit-script-versions' table check complete.")
        
        logger.info("Local DynamoDB initialization process finished.")
        
    except Exception as e:
        logger.error(f"Error during local DynamoDB initialization: {e}", exc_info=args.verbose)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test CLI for interacting with local DynamoDB.")
    parser.add_argument(
        "--dynamodb-endpoint",
        default="http://localhost:8000",
        help="Local DynamoDB endpoint URL. Default: http://localhost:8000"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging."
    )

    # Initialize logging
    # The setup_logging function should ideally take a log level argument
    # For now, we'll set it based on verbose flag after parsing unknown args once
    # or pass the verbose flag directly if setup_logging supports it.
    
    # Temporary parsing to set up logging level early
    # No longer need to parse known_args separately for logging. 
    # Logging will be set up after all args are parsed.
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=False)
    
    # Subparser for init-local-db
    init_db_parser = subparsers.add_parser(
        'init-local-db', 
        help="Initializes the local DynamoDB by creating necessary tables (conduit-pipeline-metadata, conduit-script-versions)."
    )
    init_db_parser.set_defaults(func=handle_init_local_db)

    # Subparser for register-script
    register_script_parser = subparsers.add_parser(
        'register-script',
        help="Registers a script in the local DynamoDB, mimicking script upload."
    )
    register_script_parser.add_argument('--script-path', required=True, help="Path to the local script file.")
    register_script_parser.add_argument('--script-id', required=True, help="Script ID (e.g., '0A').")
    register_script_parser.add_argument('--script-name', required=True, help="Script name (e.g., 'curate_sessions').")
    register_script_parser.add_argument('--script-version', required=True, help="Script version (e.g., 'v0').")
    register_script_parser.set_defaults(func=handle_register_script)

    # Subparser for check-script
    check_script_parser = subparsers.add_parser(
        'check-script',
        help="Checks for a script version in the local DynamoDB."
    )
    check_script_parser.add_argument('--script-id', required=True, help="Script ID to check.")
    check_script_parser.add_argument('--script-version', required=True, help="Script version to check.")
    check_script_parser.set_defaults(func=handle_check_script)

    # Subparser for put-item
    put_item_parser = subparsers.add_parser('put-item', help="Puts an item into a specified local DynamoDB table.")
    put_item_parser.add_argument('--table-name', required=True, help="Name of the DynamoDB table.")
    put_item_parser.add_argument('--item-json', required=True, help="JSON string of the item to put.")
    put_item_parser.set_defaults(func=handle_put_item)

    # Subparser for get-item
    get_item_parser = subparsers.add_parser('get-item', help="Gets an item from a specified local DynamoDB table.")
    get_item_parser.add_argument('--table-name', required=True, help="Name of the DynamoDB table.")
    get_item_parser.add_argument('--key-json', required=True, help='JSON string of the key (e.g., \'{"id": "value"}\').')
    get_item_parser.set_defaults(func=handle_get_item)

    # Subparser for scan-table
    scan_table_parser = subparsers.add_parser('scan-table', help="Scans a specified local DynamoDB table.")
    scan_table_parser.add_argument('--table-name', required=True, help="Name of the DynamoDB table.")
    scan_table_parser.set_defaults(func=handle_scan_table)

    # Subparser for delete-item
    delete_item_parser = subparsers.add_parser('delete-item', help="Deletes an item from a specified local DynamoDB table.")
    delete_item_parser.add_argument('--table-name', required=True, help="Name of the DynamoDB table.")
    delete_item_parser.add_argument('--key-json', required=True, help="JSON string of the key for the item to delete.")
    delete_item_parser.set_defaults(func=handle_delete_item)

    # Subparser for create-table
    create_table_parser = subparsers.add_parser('create-table', help="Creates a new table in the local DynamoDB.")
    create_table_parser.add_argument('--table-name', required=True, help="Name for the new table.")
    create_table_parser.add_argument('--key-schema', required=True, help='JSON string for KeySchema (e.g., \'[{"AttributeName": "id", "KeyType": "HASH"}]\').')
    create_table_parser.add_argument('--attribute-definitions', required=True, help='JSON string for AttributeDefinitions (e.g., \'[{"AttributeName": "id", "AttributeType": "S"}]\').')
    create_table_parser.add_argument('--provisioned-throughput', default='{"ReadCapacityUnits": 1, "WriteCapacityUnits": 1}', help='JSON string for ProvisionedThroughput (default: \'{"ReadCapacityUnits": 1, "WriteCapacityUnits": 1}\').')
    create_table_parser.set_defaults(func=handle_create_table)

    # Parse all arguments
    parsed_args = parser.parse_args()

    # Setup logging based on parsed arguments
    if parsed_args.verbose:
        setup_logging(level=logging.DEBUG)
        logger.debug("Verbose logging enabled.")
    else:
        setup_logging(level=logging.INFO)
    
    logger.debug(f"Parsed arguments: {parsed_args}")

    # Execute command
    if parsed_args.command is None:
        logger.info("No command specified. Use --help for options.")
        parser.print_help()
        sys.exit(1)
    elif hasattr(parsed_args, 'func'):
        parsed_args.func(parsed_args)
    else:
        # This case should ideally not be reached if 'required=True' for subparsers,
        # or if all commands set a function.
        logger.warning(f"Command '{parsed_args.command}' is recognized but has no handler function.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
