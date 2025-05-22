#!/usr/bin/env python3
"""
Script to run data pipeline transforms on a remote EC2 instance.

Usage:
    python remote.py "transform [options]"

Example:
    python remote.py "window --session Hunter"

This script will:
1. Launch an EC2 instance
2. Copy over the necessary files from the local repository
3. Set up a virtual environment and install dependencies
4. Run the specified command with the given options
5. Optionally terminate the instance when done
"""

import argparse
import boto3
import os
import sys
import time
import paramiko
import subprocess
from pathlib import Path

# Hardcoded AWS credentials for testing
# These will be rotated soon, only for testing purposes
os.environ['AWS_ACCESS_KEY_ID'] = '???'
os.environ['AWS_SECRET_ACCESS_KEY'] = '?!?'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'  # Changed to us-east-2 (Ohio) to match S3 bucket region

# Configuration
DEFAULT_AMI_ID = "ami-06c8f2ec674c67112"  # Amazon Linux 2023 AMI in us-east-2 (Ohio)
DEFAULT_INSTANCE_TYPE = "r5.4xlarge"  # 16 vCPUs, 128GB RAM
DEFAULT_KEY_NAME = "data-pipeline-key"  # Your SSH key name
DEFAULT_SECURITY_GROUP = "data-pipeline-sg"  # Security group name
DEFAULT_REGION = "us-east-2"  # Explicitly set region to us-east-2 (Ohio) to match S3 bucket region
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_DIR = "/home/ec2-user/data-pipeline"

def parse_args():
    parser = argparse.ArgumentParser(description="Run data pipeline transforms on a remote EC2 instance")
    parser.add_argument("command", help="The full command to run (e.g., 'window --session Hunter')")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--ami-id", default=DEFAULT_AMI_ID, help=f"AMI ID to use (default: {DEFAULT_AMI_ID})")
    parser.add_argument("--key-name", default=None, help=f"SSH key name (default: first available key)")
    parser.add_argument("--security-group", default=DEFAULT_SECURITY_GROUP, help=f"Security group name (default: {DEFAULT_SECURITY_GROUP})")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    parser.add_argument("--keep-alive", action="store_true", help="Don't terminate the instance after completion")
    parser.add_argument("--no-force-copy", action="store_false", dest="force_copy",
                      help="Don't force copy files (by default, files are always force copied)")
    parser.set_defaults(force_copy=True)

    return parser.parse_args()

def create_ec2_instance(args):
    """Launch an EC2 instance and return its ID and public IP"""
    print(f"Launching EC2 instance ({args.instance_type})...")

    ec2 = boto3.resource('ec2', region_name=args.region)
    ec2_client = boto3.client('ec2', region_name=args.region)

    # Create a new key pair
    key_name = "data-pipeline-temp-key"
    print(f"Creating a new key pair: {key_name}")

    # Delete the key pair if it exists
    try:
        ec2_client.delete_key_pair(KeyName=key_name)
        print(f"Deleted existing key pair: {key_name}")
    except Exception:
        pass

    # Create a new key pair
    response = ec2_client.create_key_pair(KeyName=key_name)

    # Save the private key
    key_path = os.path.expanduser(f"~/.ssh/{key_name}.pem")
    with open(key_path, 'w') as f:
        f.write(response['KeyMaterial'])

    # Set proper permissions
    os.chmod(key_path, 0o600)
    print(f"Created new key pair and saved to {key_path}")

    # Create security group if it doesn't exist
    try:
        security_group_id = create_security_group(ec2, args.security_group)
    except Exception as e:
        print(f"Error creating security group: {e}")
        print("Using default security group instead")
        # Get default security group
        default_sg = list(ec2.security_groups.filter(
            Filters=[{'Name': 'group-name', 'Values': ['default']}]
        ))
        if not default_sg:
            print("Could not find default security group. Please check your AWS configuration.")
            sys.exit(1)
        security_group_id = default_sg[0].id

    # Create a 100GB EBS volume for the root device
    block_device_mappings = [
        {
            'DeviceName': '/dev/xvda',
            'Ebs': {
                'VolumeSize': 100,  # 100 GB
                'VolumeType': 'gp3',
                'DeleteOnTermination': True
            }
        }
    ]

    # Use the specified AMI (which is now the newer Amazon Linux 2023 AMI with Python 3.11 support)
    ami_id = args.ami_id
    print(f"Using Amazon Linux 2023 AMI: {ami_id}")

    # Create instance
    instances = ec2.create_instances(
        ImageId=ami_id,
        InstanceType=args.instance_type,
        KeyName=key_name,
        SecurityGroupIds=[security_group_id],
        BlockDeviceMappings=block_device_mappings,
        MinCount=1,
        MaxCount=1,
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [
                    {
                        'Key': 'Name',
                        'Value': 'data-pipeline-runner'
                    },
                ]
            }
        ]
    )

    instance = instances[0]
    print(f"Waiting for instance {instance.id} to start...")

    instance.wait_until_running()

    # Reload the instance to get the public IP
    instance.reload()
    public_ip = instance.public_ip_address

    print(f"Instance {instance.id} started at {public_ip}")

    # Wait a bit for SSH to be available
    print("Waiting for SSH to become available...")
    time.sleep(30)

    return instance.id, public_ip, key_name

def create_security_group(ec2, group_name):
    """Create a security group for SSH access if it doesn't exist"""
    # Check if security group already exists
    existing_groups = list(ec2.security_groups.filter(
        Filters=[{'Name': 'group-name', 'Values': [group_name]}]
    ))

    if existing_groups:
        return existing_groups[0].id

    # Create new security group
    vpc = list(ec2.vpcs.all())[0]  # Get default VPC
    security_group = ec2.create_security_group(
        GroupName=group_name,
        Description='Security group for data pipeline EC2 instances',
        VpcId=vpc.id
    )

    # Add SSH ingress rule
    security_group.authorize_ingress(
        IpProtocol='tcp',
        FromPort=22,
        ToPort=22,
        CidrIp='0.0.0.0/0'  # Note: In production, restrict this to your IP
    )

    print(f"Created security group: {security_group.id}")
    return security_group.id

def setup_ssh_connection(public_ip, key_name):
    """Set up an SSH connection to the instance"""
    # Use the newly created key
    key_path = os.path.expanduser(f"~/.ssh/{key_name}.pem")

    if not os.path.exists(key_path):
        print(f"Error: SSH key not found at {key_path}")
        sys.exit(1)

    print(f"Using SSH key: {key_path}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Try to connect with retries
    max_retries = 5
    for i in range(max_retries):
        try:
            ssh.connect(
                hostname=public_ip,
                username="ec2-user",
                key_filename=key_path,
                timeout=10
            )
            print("Connected successfully")
            break
        except Exception as e:
            if i < max_retries - 1:
                print(f"Connection attempt {i+1} failed, retrying in 10 seconds... ({e})")
                time.sleep(10)
            else:
                print(f"Failed to connect after {max_retries} attempts: {e}")
                sys.exit(1)

    return ssh

def run_remote_command(ssh, command, description=None):
    """Run a command on the remote instance"""
    if description:
        print(f"{description}...")

    stdin, stdout, stderr = ssh.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()

    if exit_code != 0:
        print(f"Command failed with exit code {exit_code}")
        print("STDOUT:")
        print(stdout.read().decode())
        print("STDERR:")
        print(stderr.read().decode())
        return False

    return True

def copy_files_to_instance(public_ip, key_name, force_copy=True):
    """Copy required files to the EC2 instance using scp"""
    # Try to find a usable SSH key
    possible_keys = [
        os.path.expanduser(f"~/.ssh/{key_name}.pem"),
        os.path.expanduser(f"~/.ssh/{key_name}"),
        os.path.expanduser("~/.ssh/id_ed25519"),
        os.path.expanduser("~/.ssh/id_rsa")
    ]

    key_path = None
    for path in possible_keys:
        if os.path.exists(path):
            key_path = path
            break

    if not key_path:
        print("Error: Could not find a suitable SSH key for SCP")
        sys.exit(1)

    # Create directories on remote instance using SSH instead of SCP
    ssh = setup_ssh_connection(public_ip, key_name)

    # If force_copy is True, remove existing directory first
    if force_copy:
        print("Force copy enabled - removing existing files...")
        run_remote_command(ssh, f"rm -rf {REMOTE_DIR}")

    run_remote_command(ssh, f"mkdir -p {REMOTE_DIR}/transforms {REMOTE_DIR}/utils")

    # Use SSH to transfer files instead of SCP
    print("Copying files to remote instance...")

    # Files to copy
    files_to_copy = [
        "cli.py",
        "base_transform.py",
        "requirements.txt"
    ]

    # Copy individual files using SFTP
    sftp = ssh.open_sftp()
    try:
        for file in files_to_copy:
            local_path = os.path.join(REPO_DIR, file)
            remote_path = f"{REMOTE_DIR}/{file}"

            print(f"Copying {file}...")
            try:
                sftp.put(local_path, remote_path)
            except Exception as e:
                print(f"Error copying {file}: {e}")
                ssh.close()
                sys.exit(1)

        # Copy directories
        for directory in ["transforms", "utils"]:
            local_dir = os.path.join(REPO_DIR, directory)

            # Ensure remote directory exists
            run_remote_command(ssh, f"mkdir -p {REMOTE_DIR}/{directory}")

            # Copy all files in the directory
            for root, dirs, files in os.walk(local_dir):
                # Create relative path
                rel_path = os.path.relpath(root, REPO_DIR)

                # Create remote directories
                for d in dirs:
                    remote_dir_path = f"{REMOTE_DIR}/{rel_path}/{d}"
                    run_remote_command(ssh, f"mkdir -p {remote_dir_path}")

                # Copy files
                for file in files:
                    local_file_path = os.path.join(root, file)
                    remote_file_path = f"{REMOTE_DIR}/{rel_path}/{file}"

                    try:
                        sftp.put(local_file_path, remote_file_path)
                    except Exception as e:
                        print(f"Error copying {local_file_path}: {e}")
    finally:
        sftp.close()
        ssh.close()

def setup_environment(ssh):
    """Set up Python environment on the remote instance"""
    # Set up AWS credentials with hardcoded values
    aws_access_key = '???'
    aws_secret_key = '?!?'
    aws_region = 'us-east-2'  # Set to match S3 bucket region

    # Create AWS credentials directory
    run_remote_command(ssh, "mkdir -p ~/.aws", "Creating AWS credentials directory")

    # Create credentials file
    credentials_content = f"""[default]
aws_access_key_id = {aws_access_key}
aws_secret_access_key = {aws_secret_key}
region = {aws_region}
"""

    # Write credentials file
    run_remote_command(ssh, f"echo '{credentials_content}' > ~/.aws/credentials", "Setting up AWS credentials")
    run_remote_command(ssh, "chmod 600 ~/.aws/credentials", "Setting permissions on AWS credentials")

    # Rest of environment setup
    commands = [
        # Update system
        "sudo dnf update -y",

        # Install Python 3.11 and development tools
        "sudo dnf install -y python3.11 python3.11-pip python3.11-devel gcc",

        # Create and activate virtual environment with Python 3.11
        f"cd {REMOTE_DIR} && python3.11 -m venv venv",
        f"cd {REMOTE_DIR} && source venv/bin/activate && pip install --upgrade pip",

        # Install requirements
        f"cd {REMOTE_DIR} && source venv/bin/activate && pip install -r requirements.txt"
    ]

    # Execute each command and check for success
    for cmd in commands:
        success = run_remote_command(ssh, cmd, f"Running: {cmd}")
        if not success:
            print(f"Command failed: {cmd}")
            # Try alternative approach if Python 3.11 installation fails
            if "python3.11" in cmd and not success:
                print("Attempting alternative Python 3.11 installation...")
                # Try alternative methods to install Python 3.11
                alt_commands = [
                    # Method 1: Try Amazon Linux extras
                    "sudo amazon-linux-extras install python3.11 -y",

                    # Method 2: Try EPEL repository
                    "sudo dnf install -y epel-release",
                    "sudo dnf install -y python3.11",

                    # Method 3: Install from source as last resort
                    "sudo dnf install -y wget gcc openssl-devel bzip2-devel libffi-devel zlib-devel",
                    "cd /tmp && wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz",
                    "cd /tmp && tar xzf Python-3.11.4.tgz",
                    "cd /tmp/Python-3.11.4 && ./configure --enable-optimizations",
                    "cd /tmp/Python-3.11.4 && sudo make altinstall",
                    "sudo ln -sf /usr/local/bin/python3.11 /usr/bin/python3.11"
                ]

                for alt_cmd in alt_commands:
                    alt_success = run_remote_command(ssh, alt_cmd, f"Trying alternative: {alt_cmd}")
                    if alt_success and "python3.11" in alt_cmd:
                        # If we successfully installed Python 3.11, try to continue with the setup
                        run_remote_command(ssh, f"cd {REMOTE_DIR} && python3.11 -m venv venv", "Creating virtual environment with Python 3.11")
                        run_remote_command(ssh, f"cd {REMOTE_DIR} && source venv/bin/activate && pip install --upgrade pip", "Upgrading pip")
                        run_remote_command(ssh, f"cd {REMOTE_DIR} && source venv/bin/activate && pip install -r requirements.txt", "Installing requirements")
                        return True

            return False

    # Verify Python and Zarr versions
    run_remote_command(ssh, f"cd {REMOTE_DIR} && source venv/bin/activate && python --version", "Checking Python version")
    run_remote_command(ssh, f"cd {REMOTE_DIR} && source venv/bin/activate && pip list | grep zarr", "Checking Zarr version")

    return True

def run_transform(ssh, command):
    """Run the specified command"""
    # Set AWS environment variables for the command
    env_vars = "export AWS_ACCESS_KEY_ID=??? && "
    env_vars += "export AWS_SECRET_ACCESS_KEY=?!? && "
    env_vars += "export AWS_DEFAULT_REGION=us-east-2 && "  # Set to match S3 bucket region

    # Add debug commands to verify which files are being used and Python/Zarr versions
    debug_commands = f"cd {REMOTE_DIR} && echo 'Current directory:' && pwd && "
    debug_commands += "echo 'Listing transform files:' && ls -la transforms/ && "
    debug_commands += "echo 'Listing neural_processing files:' && ls -la transforms/neural_processing/ && "
    debug_commands += "echo 'Python version:' && python --version && "
    debug_commands += "echo 'Zarr version:' && pip list | grep zarr && "

    full_command = f"cd {REMOTE_DIR} && {env_vars} {debug_commands} source venv/bin/activate && python cli.py {command}"

    print(f"Running command: python cli.py {command}")

    # Execute the command and stream output
    stdin, stdout, stderr = ssh.exec_command(full_command)

    # Stream output in real-time
    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            data = stdout.channel.recv(1024).decode('utf-8')
            print(data, end='')

    # Get any remaining output
    data = stdout.read().decode('utf-8')
    if data:
        print(data, end='')

    # Check for errors
    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        print(f"Command failed with exit code {exit_code}")
        error_data = stderr.read().decode('utf-8')
        if error_data:
            print(f"Error output:\n{error_data}")
        return False

    return True

def terminate_instance(instance_id, region):
    """Terminate the EC2 instance"""
    print(f"Terminating instance {instance_id}...")
    ec2 = boto3.resource('ec2', region_name=region)
    instance = ec2.Instance(instance_id)
    instance.terminate()
    print("Instance termination initiated")

def main():
    args = parse_args()

    try:
        # Launch EC2 instance
        instance_id, public_ip, key_name = create_ec2_instance(args)

        try:
            # Set up SSH connection
            ssh = setup_ssh_connection(public_ip, key_name)

            # Copy files to instance with force_copy option
            copy_files_to_instance(public_ip, key_name, args.force_copy)

            # Set up environment
            if not setup_environment(ssh):
                print("Failed to set up environment")
                sys.exit(1)

            # Run the command
            success = run_transform(ssh, args.command)

            # Close SSH connection
            ssh.close()

            if not success:
                print("Command execution failed")
                sys.exit(1)

            print("Command execution completed successfully")

        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)
        finally:
            # Terminate instance unless --keep-alive was specified
            if not args.keep_alive:
                terminate_instance(instance_id, args.region)
            else:
                print(f"Instance {instance_id} ({public_ip}) kept alive as requested")
                print(f"Connect with: ssh -i ~/.ssh/{key_name}.pem ec2-user@{public_ip}")
    except Exception as e:
        print(f"Failed to launch EC2 instance: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
