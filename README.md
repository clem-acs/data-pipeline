# Project Title (Update if needed)

## Local Development Environment

This project includes a local DynamoDB setup for testing script version control and DynamoDB behavior without connecting to AWS.

### Prerequisites
- Docker
- Docker Compose

### Setting up Local DynamoDB
1.  **Start the DynamoDB container:**
    Open your terminal in the root directory of this project and run:
    ```bash
    docker-compose up -d
    ```
    This will start a local DynamoDB instance accessible at `http://localhost:8000`.
    Data will be persisted in the `./docker/dynamodb_data` directory.

2.  **View Logs (Optional):**
    To view the logs from the DynamoDB container:
    ```bash
    docker-compose logs -f dynamodb-local
    ```

3.  **Stop the DynamoDB container:**
    To stop the container:
    ```bash
    docker-compose down
    ```
    The data stored in `./docker/dynamodb_data` will persist across restarts. To start fresh, delete this directory before restarting.

### Using the Test CLI (`test_cli.py`)

The `test_cli.py` script provides tools to interact with the local DynamoDB instance for testing purposes.

**Prerequisites:**
- Ensure the local DynamoDB is running (see "Setting up Local DynamoDB" above).
- Python 3.x installed.
- Required Python packages (boto3). You can typically install boto3 using pip:
  ```bash
  pip install boto3
  ```
  (Consider adding a `requirements.txt` to the project if more dependencies arise).

**General Usage:**
```bash
python test_cli.py [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

**Global Options:**
- `--dynamodb-endpoint URL`: Specify the local DynamoDB endpoint (default: `http://localhost:8000`).
- `--verbose`: Enable verbose DEBUG level logging.

**Commands:**

1.  **`init-local-db`**: Initializes the local DynamoDB by creating standard tables.
    ```bash
    python test_cli.py init-local-db
    ```

2.  **`register-script`**: Registers a script in the local DynamoDB.
    ```bash
    python test_cli.py register-script \
        --script-path /path/to/your/script.py \
        --script-id "S01" \
        --script-name "MyTestScript" \
        --script-version "v1.0"
    ```

3.  **`check-script`**: Checks for a script version in the local DynamoDB.
    ```bash
    python test_cli.py check-script --script-id "S01" --script-version "v1.0"
    ```

4.  **`create-table`**: Creates a new custom table.
    ```bash
    python test_cli.py create-table \
        --table-name MyCustomTable \
        --key-schema '[{"AttributeName": "id", "KeyType": "HASH"}]' \
        --attribute-definitions '[{"AttributeName": "id", "AttributeType": "S"}]'
    ```
    (Optional: `--provisioned-throughput '{"ReadCapacityUnits": 1, "WriteCapacityUnits": 1}'`)

5.  **`put-item`**: Puts an item into a table.
    ```bash
    python test_cli.py put-item \
        --table-name MyCustomTable \
        --item-json '{"id": "item1", "data": "some value"}'
    ```

6.  **`get-item`**: Retrieves an item from a table.
    ```bash
    python test_cli.py get-item \
        --table-name MyCustomTable \
        --key-json '{"id": "item1"}'
    ```

7.  **`scan-table`**: Scans items from a table.
    ```bash
    python test_cli.py scan-table --table-name MyCustomTable
    ```

8.  **`delete-item`**: Deletes an item from a table.
    ```bash
    python test_cli.py delete-item \
        --table-name MyCustomTable \
        --key-json '{"id": "item1"}'
    ```
