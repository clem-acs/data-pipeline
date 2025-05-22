#!/usr/bin/env python3
"""
Script to run data pipeline transforms in parallel on a remote EC2 instance.

Usage:
    python remote_parallel.py "transform --all" --parallel 4

This script will:
1. Launch an EC2 instance
2. Copy over the necessary files from the local repository
3. Set up a virtual environment and install dependencies
4. Partition sessions by data volume and run the command in parallel processes
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
import json
import math

# Configuration
DEFAULT_AMI_ID = "ami-06c8f2ec674c67112"  # Amazon Linux 2023 AMI in us-east-2 (Ohio) with Python 3.11 support
DEFAULT_INSTANCE_TYPE = "r5.4xlarge"  # 16 vCPUs, 128GB RAM
DEFAULT_KEY_NAME = "data-pipeline-key"  # Your SSH key name
DEFAULT_SECURITY_GROUP = "data-pipeline-sg"  # Security group name
DEFAULT_REGION = "us-east-2"  # Explicitly set region to us-east-2 (Ohio) to match S3 bucket region
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_DIR = "/home/ec2-user/data-pipeline"  # For Amazon Linux 2023

def parse_args():
    parser = argparse.ArgumentParser(description="Run data pipeline transforms in parallel on a remote EC2 instance")
    parser.add_argument("command", help="The base command to run (e.g., 'window --all')")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel processes to run")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--ami-id", default=DEFAULT_AMI_ID, help=f"AMI ID to use (default: {DEFAULT_AMI_ID})")
    parser.add_argument("--key-name", default=None, help=f"SSH key name (default: first available key)")
    parser.add_argument("--security-group", default=DEFAULT_SECURITY_GROUP, help=f"Security group name (default: {DEFAULT_SECURITY_GROUP})")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    parser.add_argument("--keep-alive", action="store_true", help="Don't terminate the instance after completion")
    parser.add_argument("--no-force-copy", action="store_false", dest="force_copy",
                      help="Don't force copy files (by default, files are always force copied)")
    parser.add_argument("--min-size", type=int, default=1, help="Minimum session size in MB (default: 1)")
    parser.add_argument("--max-size", type=int, default=5000, help="Maximum session size in MB (default: 5000)")
    parser.add_argument("--s3-bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--s3-prefix", default="curated-h5/", help="S3 prefix for sessions")
    parser.add_argument("--balance-method", choices=["size", "data"], default="data",
                      help="Method to balance partitions: 'size' for file size ranges, 'data' for data volume (default: data)")
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
            'DeviceName': '/dev/xvda',  # Amazon Linux 2023 uses /dev/xvda
            'Ebs': {
                'VolumeSize': 100,  # 100 GB
                'VolumeType': 'gp3',
                'DeleteOnTermination': True
            }
        }
    ]

    # Create instance
    instances = ec2.create_instances(
        ImageId=args.ami_id,
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
                        'Value': 'data-pipeline-parallel-runner'
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
    time.sleep(30)  # Wait time for Amazon Linux 2023 to initialize

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

def setup_ssh_connection(public_ip, key_name, ami_id=None):
    """Set up an SSH connection to the instance"""
    # Use the newly created key
    key_path = os.path.expanduser(f"~/.ssh/{key_name}.pem")

    if not os.path.exists(key_path):
        print(f"Error: SSH key not found at {key_path}")
        sys.exit(1)

    print(f"Using SSH key: {key_path}")

    # Determine the correct username based on the AMI
    # Amazon Linux 2023 uses ec2-user, Ubuntu uses ubuntu
    username = "ec2-user"  # Default for Amazon Linux

    print(f"Connecting with username: {username}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Try to connect with retries
    max_retries = 5
    for i in range(max_retries):
        try:
            ssh.connect(
                hostname=public_ip,
                username=username,
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

    stdout_content = stdout.read().decode()
    stderr_content = stderr.read().decode()

    if exit_code != 0:
        print(f"Command failed with exit code {exit_code}")
        print("STDOUT:")
        print(stdout_content)
        print("STDERR:")
        print(stderr_content)
        return False

    # For debugging, optionally print stdout even on success
    if "python" in command and "--version" in command:
        print(f"Command output: {stdout_content.strip()}")

    return True

def copy_files_to_instance(public_ip, key_name, force_copy=True):
    """Copy required files to the EC2 instance using scp"""
    # Use the newly created key
    key_path = os.path.expanduser(f"~/.ssh/{key_name}.pem")

    if not os.path.exists(key_path):
        print("Error: Could not find a suitable SSH key for SCP")
        sys.exit(1)

    # Create directories on remote instance using SSH
    ssh = setup_ssh_connection(public_ip, key_name)

    # If force_copy is True, remove existing directory first
    if force_copy:
        print("Force copy enabled - removing existing files...")
        run_remote_command(ssh, f"rm -rf {REMOTE_DIR}")

    run_remote_command(ssh, f"mkdir -p {REMOTE_DIR}/transforms {REMOTE_DIR}/utils")

    # Use SSH to transfer files
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
    # Create AWS credentials directory
    run_remote_command(ssh, "mkdir -p ~/.aws", "Creating AWS credentials directory")

    # Hardcode AWS credentials from .zshrc
    aws_access_key = "???"
    aws_secret_key = "?!?"
    aws_region = "us-east-1"  # Using us-east-1 as found in .zshrc

    # Create credentials file with hardcoded values
    credentials_content = f"""[default]
aws_access_key_id = {aws_access_key}
aws_secret_access_key = {aws_secret_key}
region = {aws_region}
"""

    # Write credentials file
    run_remote_command(ssh, f"echo '{credentials_content}' > ~/.aws/credentials", "Setting up AWS credentials")
    run_remote_command(ssh, "chmod 600 ~/.aws/credentials", "Setting permissions on AWS credentials")

    # Also set them as environment variables in the remote session
    run_remote_command(ssh, f"echo 'export AWS_ACCESS_KEY_ID={aws_access_key}' >> ~/.bashrc")
    run_remote_command(ssh, f"echo 'export AWS_SECRET_ACCESS_KEY={aws_secret_key}' >> ~/.bashrc")
    run_remote_command(ssh, f"echo 'export AWS_DEFAULT_REGION={aws_region}' >> ~/.bashrc")

    # Set them for the current session too
    run_remote_command(ssh, f"export AWS_ACCESS_KEY_ID={aws_access_key}")
    run_remote_command(ssh, f"export AWS_SECRET_ACCESS_KEY={aws_secret_key}")
    run_remote_command(ssh, f"export AWS_DEFAULT_REGION={aws_region}")

    print("Setting up Amazon Linux 2023 with Python 3.11")

    # First, update the system and install basic tools
    commands = [
        # Update system
        "sudo yum update -y",

        # Install development tools for compiling Python if needed
        "sudo yum groupinstall -y 'Development Tools'",
        "sudo yum install -y openssl-devel bzip2-devel libffi-devel",

        # Check if Python 3.11 is already installed
        "which python3.11 || echo 'Python 3.11 not found'",
    ]

    for cmd in commands:
        success = run_remote_command(ssh, cmd, f"Running: {cmd}")
        if not success:
            return False

    # Check if Python 3.11 is installed
    stdin, stdout, stderr = ssh.exec_command("which python3.11")
    python311_path = stdout.read().decode().strip()

    if not python311_path:
        print("Python 3.11 not found, installing it now...")

        # Install Python 3.11 using Amazon Linux extras
        install_commands = [
            # Try to install Python 3.11 from the Amazon Linux repos first
            "sudo dnf install -y python3.11",

            # Install pip for Python 3.11
            "sudo dnf install -y python3.11-pip",

            # Install venv for Python 3.11
            "sudo dnf install -y python3.11-devel",

            # Verify installation
            "python3.11 --version",
        ]

        for cmd in install_commands:
            success = run_remote_command(ssh, cmd, f"Running: {cmd}")
            if not success:
                print(f"Failed to run: {cmd}")
                return False

    # Now create the virtual environment with Python 3.11
    venv_commands = [
        # Remove any existing virtual environment
        f"rm -rf {REMOTE_DIR}/venv",

        # Create a new virtual environment with Python 3.11
        f"cd {REMOTE_DIR} && python3.11 -m venv venv",

        # Upgrade pip in the virtual environment
        f"cd {REMOTE_DIR} && source venv/bin/activate && pip install --upgrade pip",

        # Install AWS CLI in the virtual environment
        f"cd {REMOTE_DIR} && source venv/bin/activate && pip install awscli",

        # Install requirements
        f"cd {REMOTE_DIR} && source venv/bin/activate && pip install -r requirements.txt"
    ]

    for cmd in venv_commands:
        success = run_remote_command(ssh, cmd, f"Running: {cmd}")
        if not success:
            return False

    # Verify AWS CLI works and show configuration
    aws_check_commands = [
        f"cd {REMOTE_DIR} && source venv/bin/activate && aws --version",
        f"cd {REMOTE_DIR} && source venv/bin/activate && aws configure list"
    ]

    for cmd in aws_check_commands:
        success = run_remote_command(ssh, cmd, f"Running: {cmd}")
        if not success:
            print(f"Warning: AWS CLI check failed, but continuing anyway")

    return True

def get_session_sizes(ssh, bucket, prefix):
    """Get the sizes of all sessions in the S3 bucket"""
    print("Getting session sizes from S3...")

    # First, check AWS configuration
    debug_commands = [
        "aws configure list",
        "aws s3 ls",
        f"aws s3 ls s3://{bucket}/ --region us-east-1",
        f"aws s3 ls s3://{bucket}/{prefix} --region us-east-1 | head -5"
    ]

    for cmd in debug_commands:
        print(f"Running AWS debug command: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(f"cd {REMOTE_DIR} && source venv/bin/activate && {cmd}")
        print(stdout.read().decode())
        print(stderr.read().decode())

    # Run AWS CLI command to list objects and get their sizes with explicit region
    cmd = f"cd {REMOTE_DIR} && source venv/bin/activate && " \
          f"AWS_ACCESS_KEY_ID=??? " \
          f"AWS_SECRET_ACCESS_KEY=?!? " \
          f"AWS_DEFAULT_REGION=us-east-1 " \
          f"aws s3 ls s3://{bucket}/{prefix} --recursive --region us-east-1 | " \
          f"grep -E '\\.h5$' | awk '{{print $3, $4}}'"

    print(f"Running full S3 listing command...")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()

    if exit_code != 0:
        stderr_output = stderr.read().decode()
        print(f"Failed to get session sizes: {stderr_output}")
        return {}

    # Parse the output to get session sizes
    session_sizes = {}
    output = stdout.read().decode()

    # Print a sample of the output for debugging
    print(f"Sample of S3 listing output (first 5 lines):")
    for i, line in enumerate(output.splitlines()[:5]):
        print(f"  {line}")

    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            size = int(parts[0])  # Size in bytes
            path = ' '.join(parts[1:])  # Path might contain spaces

            # Extract session ID from path
            session_id = os.path.basename(path).replace('.h5', '')

            # Convert size to MB
            size_mb = size / (1024 * 1024)
            session_sizes[session_id] = size_mb

    print(f"Found {len(session_sizes)} sessions with sizes")
    return session_sizes

def partition_sessions_by_size(session_sizes, num_partitions, min_size=1, max_size=5000):
    """Partition sessions by size into num_partitions groups"""
    print(f"Partitioning {len(session_sizes)} sessions into {num_partitions} groups by size range...")

    # Calculate size ranges for each partition
    size_range = max_size - min_size
    partition_size = size_range / num_partitions

    partitions = []
    for i in range(num_partitions):
        partition_min = min_size + (i * partition_size)
        partition_max = min_size + ((i + 1) * partition_size)

        # For the last partition, ensure we include the max_size
        if i == num_partitions - 1:
            partition_max = max_size

        # Find sessions in this size range
        partition_sessions = [
            session_id for session_id, size in session_sizes.items()
            if partition_min <= size < partition_max or (i == num_partitions - 1 and size == max_size)
        ]

        # Calculate total data size in this partition
        total_data = sum(session_sizes[session_id] for session_id in partition_sessions)

        partitions.append({
            'min_size': partition_min,
            'max_size': partition_max,
            'sessions': partition_sessions,
            'total_data': total_data
        })

    # Print partition information with clear indication of max file sizes
    print("\n" + "="*80)
    print("PARTITION INFORMATION (SIZE RANGE BALANCED)")
    print("="*80)

    total_data_mb = sum(session_sizes.values())

    for i, partition in enumerate(partitions):
        print(f"Partition {i+1}:")
        print(f"  Sessions: {len(partition['sessions'])}")
        print(f"  Size range: {partition['min_size']:.2f} MB to {partition['max_size']:.2f} MB")
        print(f"  Total data: {partition['total_data']:.2f} MB ({(partition['total_data']/total_data_mb)*100:.1f}% of total)")
        print(f"  MAX FILE SIZE PROCESSED: {partition['max_size']:.2f} MB")
        print("-"*80)

    # Print the overall maximum file size
    overall_max = max(session_sizes.values())
    print(f"LARGEST SESSION IN DATASET: {overall_max:.2f} MB\n")

    return partitions

def partition_sessions_by_data_volume(session_sizes, num_partitions):
    """Partition sessions to balance total data volume across partitions"""
    print(f"Partitioning {len(session_sizes)} sessions into {num_partitions} groups by data volume...")

    # Calculate total data volume
    total_data_mb = sum(session_sizes.values())
    target_per_partition = total_data_mb / num_partitions

    print(f"Total data: {total_data_mb:.2f} MB")
    print(f"Target per partition: {target_per_partition:.2f} MB")

    # Sort sessions by size
    sorted_sessions = sorted(session_sizes.items(), key=lambda x: x[1])

    partitions = []
    current_partition = []
    current_size = 0

    for session_id, size in sorted_sessions:
        current_partition.append(session_id)
        current_size += size

        # When we reach the target size, start a new partition
        # But ensure we don't create more partitions than requested
        if current_size >= target_per_partition and len(partitions) < num_partitions - 1:
            min_size = sorted_sessions[0][1] if not partitions else partitions[-1]['max_size']
            max_size = size

            partitions.append({
                'min_size': min_size,
                'max_size': max_size,
                'sessions': current_partition,
                'total_data': current_size
            })

            current_partition = []
            current_size = 0

    # Add the last partition
    if current_partition:
        min_size = sorted_sessions[0][1] if not partitions else partitions[-1]['max_size']
        max_size = sorted_sessions[-1][1]

        partitions.append({
            'min_size': min_size,
            'max_size': max_size,
            'sessions': current_partition,
            'total_data': current_size
        })

    # Print partition information with clear indication of max file sizes
    print("\n" + "="*80)
    print("PARTITION INFORMATION (DATA VOLUME BALANCED)")
    print("="*80)

    for i, partition in enumerate(partitions):
        print(f"Partition {i+1}:")
        print(f"  Sessions: {len(partition['sessions'])}")
        print(f"  Size range: {partition['min_size']:.2f} MB to {partition['max_size']:.2f} MB")
        print(f"  Total data: {partition['total_data']:.2f} MB ({(partition['total_data']/total_data_mb)*100:.1f}% of total)")
        print(f"  MAX FILE SIZE PROCESSED: {partition['max_size']:.2f} MB")
        print("-"*80)

    # Print the overall maximum file size
    overall_max = max(session_sizes.values())
    print(f"LARGEST SESSION IN DATASET: {overall_max:.2f} MB\n")

    return partitions

def run_parallel_commands(ssh, base_command, partitions):
    """Run commands in parallel for each partition"""
    print(f"Running {len(partitions)} parallel commands...")

    # Create a script for each partition
    script_paths = []
    for i, partition in enumerate(partitions):
        # Add AWS environment variables to the script
        script_content = f"""#!/bin/bash
cd {REMOTE_DIR}
source venv/bin/activate

# Set AWS environment variables
export AWS_ACCESS_KEY_ID=???
export AWS_SECRET_ACCESS_KEY=?!?
export AWS_DEFAULT_REGION=us-east-1

# Run the command with size limits
python3 cli.py {base_command} --min-session-size {partition['min_size']:.2f} --max-session-size {partition['max_size']:.2f} 2>&1 | tee partition_{i+1}.log
"""

        # Write the script to a file on the remote instance
        script_path = f"{REMOTE_DIR}/run_partition_{i+1}.sh"
        stdin, stdout, stderr = ssh.exec_command(f"cat > {script_path} << 'EOL'\n{script_content}\nEOL")
        exit_code = stdout.channel.recv_exit_status()

        if exit_code != 0:
            print(f"Failed to create script for partition {i+1}: {stderr.read().decode()}")
            continue

        # Make the script executable
        run_remote_command(ssh, f"chmod +x {script_path}")
        script_paths.append(script_path)

    # Start all scripts in the background, but show partition 1 in real-time
    processes = []

    # Start partition 1 in the foreground to show real-time output
    if len(script_paths) > 0:
        print(f"Starting process for partition 1 (showing real-time output)...")
        channel = ssh.get_transport().open_session()
        channel.get_pty()  # Request a pseudo-terminal to get real-time output
        channel.exec_command(f"{script_paths[0]}")

        # Stream output in real-time
        while not channel.exit_status_ready():
            if channel.recv_ready():
                data = channel.recv(1024).decode('utf-8')
                print(f"Partition 1: {data}", end='')
            if channel.recv_stderr_ready():
                data = channel.recv_stderr(1024).decode('utf-8')
                print(f"Partition 1 (stderr): {data}", end='')
            time.sleep(0.1)

        # Get any remaining output
        while channel.recv_ready():
            data = channel.recv(1024).decode('utf-8')
            print(f"Partition 1: {data}", end='')

        exit_status = channel.recv_exit_status()
        print(f"Partition 1 completed with exit status: {exit_status}")

    # Start the rest of the partitions in the background
    for i, script_path in enumerate(script_paths[1:], 2):
        print(f"Starting process for partition {i}...")
        stdin, stdout, stderr = ssh.exec_command(f"nohup {script_path} &")
        processes.append((i, stdin, stdout, stderr))

    # Wait for all processes to complete
    print("All processes started. Monitoring progress...")

    # Check if processes are still running
    running = True
    while running:
        time.sleep(10)  # Check every 10 seconds

        # Check if any processes are still running
        stdin, stdout, stderr = ssh.exec_command("ps aux | grep 'python3 cli.py' | grep -v grep")
        output = stdout.read().decode()

        if not output.strip():
            running = False
            print("All processes have completed.")
        else:
            print(f"Still running: {output.count('python3 cli.py')} processes")

    # Collect logs from partitions 2 and onwards (partition 1 was shown in real-time)
    print("Collecting logs from all partitions...")
    for i in range(2, len(partitions) + 1):
        stdin, stdout, stderr = ssh.exec_command(f"cat {REMOTE_DIR}/partition_{i}.log")
        log_content = stdout.read().decode()

        print(f"\n--- Partition {i} Log ---")
        print(log_content[:1000] + ("..." if len(log_content) > 1000 else ""))
        print(f"--- End of Partition {i} Log ---\n")

    return True

def main():
    args = parse_args()

    try:
        # Launch EC2 instance
        instance_id, public_ip, key_name = create_ec2_instance(args)

        try:
            # Set up SSH connection
            ssh = setup_ssh_connection(public_ip, key_name, args.ami_id)

            # Copy files to instance
            copy_files_to_instance(public_ip, key_name, args.force_copy)

            # Set up environment
            if not setup_environment(ssh):
                print("Failed to set up environment")
                sys.exit(1)

            # Get session sizes
            session_sizes = get_session_sizes(ssh, args.s3_bucket, args.s3_prefix)

            if not session_sizes:
                print("No sessions found. Exiting.")
                sys.exit(1)

            # Partition sessions based on selected method
            if args.balance_method == "size":
                partitions = partition_sessions_by_size(
                    session_sizes,
                    args.parallel,
                    args.min_size,
                    args.max_size
                )
            else:  # data volume balancing
                partitions = partition_sessions_by_data_volume(
                    session_sizes,
                    args.parallel
                )

            # Run commands in parallel
            success = run_parallel_commands(ssh, args.command, partitions)

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

def terminate_instance(instance_id, region):
    """Terminate the EC2 instance"""
    print(f"Terminating instance {instance_id}...")
    ec2 = boto3.resource('ec2', region_name=region)
    instance = ec2.Instance(instance_id)
    instance.terminate()
    print("Instance termination initiated")

if __name__ == "__main__":
    main()
