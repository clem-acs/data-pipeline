import unittest
import subprocess
import json
import boto3
import os # For environment variables if needed for endpoint
import tempfile
import shutil
from datetime import datetime

# Define the default local DynamoDB endpoint for tests
DYNAMODB_ENDPOINT = os.environ.get('TEST_DYNAMODB_ENDPOINT', 'http://localhost:8000')

class TestTestCli(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        self.dynamodb_client = boto3.client('dynamodb', endpoint_url=DYNAMODB_ENDPOINT)
        # Clean up any pre-existing tables (optional, good for idempotency)
        self._delete_table_if_exists('conduit-pipeline-metadata')
        self._delete_table_if_exists('conduit-script-versions')
        # Add other tables if they might persist from failed tests

    def tearDown(self):
        """Clean up after each test."""
        self._delete_table_if_exists('conduit-pipeline-metadata')
        self._delete_table_if_exists('conduit-script-versions')
        # Add other tables created during tests

    def _delete_table_if_exists(self, table_name):
        try:
            self.dynamodb_client.delete_table(TableName=table_name)
            # Wait for table to be deleted (optional, good for quick re-runs)
            waiter = self.dynamodb_client.get_waiter('table_not_exists')
            waiter.wait(TableName=table_name, WaiterConfig={'Delay': 1, 'MaxAttempts': 10})
        except self.dynamodb_client.exceptions.ResourceNotFoundException:
            pass # Table doesn't exist, which is fine
        except Exception as e:
            print(f"Error deleting table {table_name} during test cleanup: {e}")


    def run_cli_command(self, command_args):
        """Helper to run test_cli.py commands."""
        base_command = ['python', '../test_cli.py', '--dynamodb-endpoint', DYNAMODB_ENDPOINT]
        full_command = base_command + command_args
        try:
            result = subprocess.run(full_command, capture_output=True, text=True, check=True, cwd='tests') # Run from tests dir
            return result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            print(f"CLI Error Output: {e.stderr}")
            raise

    def test_01_init_local_db(self):
        """Test the init-local-db command."""
        stdout, stderr = self.run_cli_command(['init-local-db'])
        
        self.assertIn("Initializing local DynamoDB tables...", stdout)
        self.assertIn("Table 'conduit-pipeline-metadata' created/verified.", stdout)
        self.assertIn("Table 'conduit-script-versions' created/verified.", stdout)

        # Verify tables exist using boto3
        try:
            response_pipeline = self.dynamodb_client.describe_table(TableName='conduit-pipeline-metadata')
            self.assertEqual(response_pipeline['Table']['TableName'], 'conduit-pipeline-metadata')
            
            response_scripts = self.dynamodb_client.describe_table(TableName='conduit-script-versions')
            self.assertEqual(response_scripts['Table']['TableName'], 'conduit-script-versions')
        except Exception as e:
            self.fail(f"describe_table failed after init-local-db: {e}")

    def test_02_register_and_check_script(self):
        """Test registering a new script and then checking it."""
        # Create a dummy script file
        temp_dir = tempfile.mkdtemp()
        dummy_script_path = os.path.join(temp_dir, "dummy_script.py")
        with open(dummy_script_path, "w") as f:
            f.write("print('hello')")

        script_id = "TestScript01"
        script_name = "MyDummyScript"
        script_version = "v0.0.1"

        # Test register-script
        stdout_reg, stderr_reg = self.run_cli_command([
            'register-script',
            '--script-path', dummy_script_path,
            '--script-id', script_id,
            '--script-name', script_name,
            '--script-version', script_version
        ])
        
        self.assertIn(f"Registering script: ID={script_id}, Version={script_version}, Name={script_name}, Path={dummy_script_path}", stdout_reg)
        self.assertIn(f"Successfully registered script '{script_name}' (ID: {script_id}, Version: {script_version})", stdout_reg)
        self.assertIn("MD5 hash", stdout_reg) 

        # Test check-script
        stdout_check, stderr_check = self.run_cli_command([
            'check-script',
            '--script-id', script_id,
            '--script-version', script_version
        ])

        self.assertIn("Script found.", stdout_check)
        self.assertIn(f'"script_id": "{script_id}"', stdout_check)
        self.assertIn(f'"script_version": "{script_version}"', stdout_check)
        self.assertIn(f'"s3_path": "{dummy_script_path}"', stdout_check) 
        self.assertIn(f'"script_name": "{script_name}"', stdout_check)
        self.assertIn('"md5_hash":', stdout_check)

        shutil.rmtree(temp_dir)

    def test_03_check_nonexistent_script(self):
        """Test checking a script that hasn't been registered."""
        script_id = "NonExistentS01"
        script_version = "v1.0"
        stdout, stderr = self.run_cli_command([
            'check-script',
            '--script-id', script_id,
            '--script-version', script_version
        ])
        self.assertIn(f"Checking script: ID={script_id}, Version={script_version}", stdout)
        self.assertIn("Script not found.", stdout)

    def test_04_create_custom_table(self):
        """Test the create-table command."""
        table_name = "MyTestCustomTable"
        self._delete_table_if_exists(table_name) 

        key_schema_json = '[{"AttributeName": "customId", "KeyType": "HASH"}]'
        attr_defs_json = '[{"AttributeName": "customId", "AttributeType": "S"}]'
        
        stdout, stderr = self.run_cli_command([
            'create-table',
            '--table-name', table_name,
            '--key-schema', key_schema_json,
            '--attribute-definitions', attr_defs_json
        ])

        self.assertIn(f"Creating table '{table_name}'", stdout)
        self.assertIn(f"Table '{table_name}' created successfully.", stdout)

        try:
            response = self.dynamodb_client.describe_table(TableName=table_name)
            self.assertEqual(response['Table']['TableName'], table_name)
            self.assertEqual(response['Table']['KeySchema'], json.loads(key_schema_json))
            defined_attrs = json.loads(attr_defs_json)
            actual_attrs = response['Table']['AttributeDefinitions']
            for da in defined_attrs:
                self.assertIn(da, actual_attrs)
        except Exception as e:
            self.fail(f"describe_table failed for {table_name}: {e}")
        finally:
            self._delete_table_if_exists(table_name)

    def test_05_put_get_scan_delete_item(self):
        """Test put-item, get-item, scan-table, and delete-item commands."""
        table_name = "ItemTestTable"
        self._delete_table_if_exists(table_name) # Clean before

        # 1. Create a table for item operations
        key_schema_json = '[{"AttributeName": "itemId", "KeyType": "HASH"}]'
        attr_defs_json = '[{"AttributeName": "itemId", "AttributeType": "S"}]'
        self.run_cli_command([
            'create-table',
            '--table-name', table_name,
            '--key-schema', key_schema_json,
            '--attribute-definitions', attr_defs_json
        ])
        # Wait for table to be active (important before putting items)
        waiter = self.dynamodb_client.get_waiter('table_exists')
        waiter.wait(TableName=table_name, WaiterConfig={'Delay': 1, 'MaxAttempts': 10})

        item_id = "testItem001"
        item_data = "Hello DynamoDB"
        item_json = f'{{"itemId": "{item_id}", "data": "{item_data}"}}'
        key_json = f'{{"itemId": "{item_id}"}}'

        # 2. Test put-item
        stdout_put, _ = self.run_cli_command([
            'put-item',
            '--table-name', table_name,
            '--item-json', item_json
        ])
        self.assertIn(f"Successfully put item into {table_name}", stdout_put)
        
        # Verify with boto3 directly (optional but good)
        try:
            response = self.dynamodb_client.get_item(TableName=table_name, Key={"itemId": {"S": item_id}})
            self.assertEqual(response['Item']['data']['S'], item_data)
        except Exception as e:
            self.fail(f"boto3.get_item failed after cli put-item: {e}")

        # 3. Test get-item
        stdout_get, _ = self.run_cli_command([
            'get-item',
            '--table-name', table_name,
            '--key-json', key_json
        ])
        self.assertIn(f'"itemId": "{item_id}"', stdout_get)
        self.assertIn(f'"data": "{item_data}"', stdout_get)

        # 4. Test scan-table
        stdout_scan, _ = self.run_cli_command([
            'scan-table',
            '--table-name', table_name
        ])
        self.assertIn(f'"itemId": "{item_id}"', stdout_scan)
        self.assertIn(f'"data": "{item_data}"', stdout_scan)
        
        # Put another item to test scan more thoroughly
        item_id_2 = "testItem002"
        item_data_2 = "Another item"
        item_json_2 = f'{{"itemId": "{item_id_2}", "data": "{item_data_2}"}}'
        self.run_cli_command([
            'put-item',
            '--table-name', table_name,
            '--item-json', item_json_2
        ])
        stdout_scan_2, _ = self.run_cli_command([
            'scan-table',
            '--table-name', table_name
        ])
        self.assertIn(f'"itemId": "{item_id}"', stdout_scan_2)
        self.assertIn(f'"data": "{item_data}"', stdout_scan_2)
        self.assertIn(f'"itemId": "{item_id_2}"', stdout_scan_2)
        self.assertIn(f'"data": "{item_data_2}"', stdout_scan_2)


        # 5. Test delete-item
        stdout_delete, _ = self.run_cli_command([
            'delete-item',
            '--table-name', table_name,
            '--key-json', key_json # Deleting the first item
        ])
        self.assertIn(f"Successfully deleted item from {table_name}", stdout_delete)

        # Verify with get-item that it's gone
        stdout_get_after_delete, _ = self.run_cli_command([
            'get-item',
            '--table-name', table_name,
            '--key-json', key_json
        ])
        self.assertIn("Item not found.", stdout_get_after_delete)
        
        # Verify with boto3 that it's gone (optional)
        try:
            response = self.dynamodb_client.get_item(TableName=table_name, Key={"itemId": {"S": item_id}})
            self.assertNotIn('Item', response, "Item should have been deleted.")
        except Exception as e:
            self.fail(f"boto3.get_item failed checking deletion: {e}")

        # Clean up the table
        self._delete_table_if_exists(table_name)

    def test_06_get_nonexistent_item(self):
        """Test get-item for an item that doesn't exist."""
        table_name = "NoItemTable"
        self._delete_table_if_exists(table_name)

        key_schema_json = '[{"AttributeName": "id", "KeyType": "HASH"}]'
        attr_defs_json = '[{"AttributeName": "id", "AttributeType": "S"}]'
        self.run_cli_command([
            'create-table',
            '--table-name', table_name,
            '--key-schema', key_schema_json,
            '--attribute-definitions', attr_defs_json
        ])
        waiter = self.dynamodb_client.get_waiter('table_exists')
        waiter.wait(TableName=table_name, WaiterConfig={'Delay': 1, 'MaxAttempts': 10})
        
        key_json = '{"id": "nonexistent"}'
        stdout_get, _ = self.run_cli_command([
            'get-item',
            '--table-name', table_name,
            '--key-json', key_json
        ])
        self.assertIn("Item not found.", stdout_get)
        self._delete_table_if_exists(table_name)

if __name__ == '__main__':
    unittest.main()
