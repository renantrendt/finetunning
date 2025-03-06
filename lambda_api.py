# Lambda Cloud API Client for Yanomami Translation Model Training
#
# This file provides functions to interact with Lambda Cloud's API
# for managing cloud instances for training the Yanomami translation model

import os
import sys
import json
import time
import requests
from pathlib import Path
import subprocess

# Lambda Cloud API base URL
LAMBDA_API_BASE_URL = "https://cloud.lambdalabs.com/api/v1"

class LambdaCloudAPI:
    def __init__(self, api_key=None):
        """Initialize the Lambda Cloud API client.
        
        Args:
            api_key (str, optional): Lambda Cloud API key. If not provided,
                                     will look for LAMBDA_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY")
        if not self.api_key:
            raise ValueError("Lambda Cloud API key is required. Set the LAMBDA_API_KEY environment variable or pass it to the constructor.")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _request(self, method, endpoint, data=None):
        """Make a request to the Lambda Cloud API.
        
        Args:
            method (str): HTTP method (GET, POST, DELETE)
            endpoint (str): API endpoint
            data (dict, optional): Request data
            
        Returns:
            dict: Response data
        """
        url = f"{LAMBDA_API_BASE_URL}/{endpoint}"
        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            json=data
        )
        
        if response.status_code >= 400:
            error_msg = f"Error {response.status_code}: {response.text}"
            raise Exception(error_msg)
        
        return response.json()
    
    def list_instance_types(self):
        """List available instance types.
        
        Returns:
            dict: Available instance types
        """
        # Make direct request to the correct endpoint
        url = f"{LAMBDA_API_BASE_URL}/instance-types"
        print(f"Making request to: {url}")
        
        response = requests.get(
            url=url,
            headers=self.headers
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code >= 400:
            error_msg = f"Error {response.status_code}: {response.text}"
            raise Exception(error_msg)
        
        # Print the raw response for debugging
        raw_response = response.json()
        print(f"Raw response type: {type(raw_response)}")
        print(f"Raw response structure: {raw_response[:500] if isinstance(raw_response, list) else list(raw_response.keys()) if isinstance(raw_response, dict) else 'Unknown type'}")
        
        return raw_response
    
    def list_instances(self):
        """List running instances.
        
        Returns:
            dict: Running instances
        """
        return self._request("GET", "instances")
    
    def launch_instance(self, instance_type, name, ssh_key_ids, region=None):
        """Launch a new instance.
        
        Args:
            instance_type (str): Instance type (e.g., "gpu_1x_a10")
            name (str): Instance name
            ssh_key_ids (list): List of SSH key IDs
            region (str, optional): Region to launch in
            
        Returns:
            dict: Instance information
        """
        # Print debug information
        print(f"Debug: Launching instance with type {instance_type}")
        print(f"Debug: Using SSH key IDs: {ssh_key_ids}")
        if region:
            print(f"Debug: In region {region}")
            
        # Prepare request data
        data = {
            "instance_type_id": instance_type,
            "name": name,
            "ssh_key_ids": ssh_key_ids
        }
        
        if region:
            data["region_id"] = region
            
        # Make direct request to avoid potential issues with the wrapper
        url = f"{LAMBDA_API_BASE_URL}/instance-operations/launch"
        response = requests.post(
            url=url,
            headers=self.headers,
            json=data
        )
        
        if response.status_code >= 400:
            error_msg = f"Error {response.status_code}: {response.text}"
            raise Exception(error_msg)
        
        return response.json()
    
    def terminate_instance(self, instance_id):
        """Terminate an instance.
        
        Args:
            instance_id (str): Instance ID
            
        Returns:
            dict: Response data
        """
        return self._request("DELETE", f"instances/{instance_id}")
    
    def list_ssh_keys(self):
        """List SSH keys.
        
        Returns:
            dict: SSH keys
        """
        # Make direct request to the correct endpoint
        url = f"{LAMBDA_API_BASE_URL}/ssh-keys"
        response = requests.get(
            url=url,
            headers=self.headers
        )
        
        if response.status_code >= 400:
            error_msg = f"Error {response.status_code}: {response.text}"
            raise Exception(error_msg)
        
        return response.json()
    
    def add_ssh_key(self, name, public_key):
        """Add an SSH key.
        
        Args:
            name (str): Key name
            public_key (str): Public key content
            
        Returns:
            dict: Response data
        """
        # Print debug information
        print(f"Debug: Adding SSH key with name: {name}")
        
        # Prepare request data
        data = {
            "name": name,
            "public_key": public_key
        }
        
        # Make direct request to the correct endpoint
        url = f"{LAMBDA_API_BASE_URL}/ssh-keys"
        response = requests.post(
            url=url,
            headers=self.headers,
            json=data
        )
        
        if response.status_code >= 400:
            error_msg = f"Error {response.status_code}: {response.text}"
            raise Exception(error_msg)
        
        return response.json()

# Helper functions
def get_or_create_ssh_key(api):
    """Get or create an SSH key for Lambda Cloud.
    
    Args:
        api (LambdaCloudAPI): Lambda Cloud API client
        
    Returns:
        str: SSH key ID
    """
    # Check if SSH key exists locally
    ssh_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
    if not os.path.exists(ssh_key_path):
        # Generate SSH key if it doesn't exist
        print("Generating SSH key...")
        os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "rsa", "-b", "4096", "-f", os.path.expanduser("~/.ssh/id_rsa"), "-N", ""], check=True)
    
    # Read public key
    with open(ssh_key_path, "r") as f:
        public_key = f.read().strip()
    
    print("Checking for existing SSH keys in Lambda Cloud...")
    try:
        # Check if key exists in Lambda Cloud
        response = api.list_ssh_keys()
        print(f"Found {len(response.get('data', []))} existing SSH keys")
        
        for key in response.get("data", []):
            if public_key in key.get("public_key", ""):
                print(f"Found existing SSH key: {key.get('name', 'Unknown')}")
                return key["id"]
        
        # Add key to Lambda Cloud
        print("Adding SSH key to Lambda Cloud...")
        hostname = os.uname().nodename
        key_name = f"yanomami-training-{hostname}-{int(time.time())}"
        print(f"Creating key with name: {key_name}")
        
        response = api.add_ssh_key(key_name, public_key)
        key_id = response.get("data", {}).get("id")
        
        if not key_id:
            raise ValueError("Failed to get SSH key ID from response")
            
        print(f"Successfully added SSH key with ID: {key_id}")
        return key_id
    
    except Exception as e:
        print(f"Error in SSH key management: {e}")
        raise

def wait_for_ssh(ip_address, max_attempts=10, delay=10):
    """Wait for SSH to become available.
    
    Args:
        ip_address (str): Instance IP address
        max_attempts (int): Maximum number of attempts
        delay (int): Delay between attempts in seconds
        
    Returns:
        bool: True if SSH is available, False otherwise
    """
    print(f"Waiting for SSH to become available on {ip_address}...")
    
    for attempt in range(max_attempts):
        try:
            subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip_address}", "echo 'SSH connection established'"],
                check=True,
                capture_output=True,
                text=True
            )
            print("SSH connection established!")
            return True
        except subprocess.CalledProcessError:
            print(f"Attempt {attempt + 1}/{max_attempts}: SSH not yet available, waiting {delay} seconds...")
            time.sleep(delay)
    
    print("Failed to establish SSH connection after multiple attempts.")
    return False

def transfer_files(ip_address, local_dir, remote_dir):
    """Transfer files to Lambda instance.
    
    Args:
        ip_address (str): Instance IP address
        local_dir (str): Local directory
        remote_dir (str): Remote directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Transferring files to {ip_address}:{remote_dir}...")
    
    # Create remote directory
    try:
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip_address}", f"mkdir -p {remote_dir}"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error creating remote directory: {e}")
        return False
    
    # Transfer files using rsync
    try:
        subprocess.run(
            ["rsync", "-avz", 
             "--exclude", ".git", 
             "--exclude", "__pycache__", 
             "--exclude", "*.pyc", 
             "--exclude", "*.pyo", 
             "--exclude", "*.pyd", 
             "--exclude", "logs", 
             "--exclude", "checkpoints", 
             "--exclude", "RAG", 
             "--exclude", "stash", 
             f"{local_dir}/", f"ubuntu@{ip_address}:{remote_dir}/"],
            check=True
        )
        print("Files transferred successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error transferring files: {e}")
        return False

def setup_environment(ip_address):
    """Set up the environment on the Lambda instance.
    
    Args:
        ip_address (str): Instance IP address
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("Setting up environment on Lambda instance...")
    
    setup_commands = [
        "cd ~/yanomami_project",
        "pip install -r requirements.txt",
        "pip install psutil matplotlib",
        "mkdir -p /lambda_storage/yanomami_dataset",
        "mkdir -p /lambda_storage/checkpoints",
        "mkdir -p /lambda_storage/logs",
        "mkdir -p /lambda_storage/enhanced_yanomami_translator",
        "mkdir -p /lambda_storage/visualization_results",
        "cp -r yanomami_dataset/* /lambda_storage/yanomami_dataset/",
        "cp -r yanomami_tokenizer /lambda_storage/"
    ]
    
    setup_script = "; ".join(setup_commands)
    
    try:
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip_address}", setup_script],
            check=True
        )
        print("Environment setup completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up environment: {e}")
        return False

def start_training(ip_address, resume=False):
    """Start training on the Lambda instance.
    
    Args:
        ip_address (str): Instance IP address
        resume (bool): Whether to resume training
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("Starting training on Lambda instance with 8x A100 GPUs...")
    
    # Setup distributed training environment variables
    setup_commands = [
        "cd ~/yanomami_project",
        "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "export WORLD_SIZE=8",
        "export MASTER_ADDR=localhost",
        "export MASTER_PORT=29500"
    ]
    
    # Base training command
    base_cmd = "python -m torch.distributed.launch --nproc_per_node=8 train_yanomami_lambda.py --distributed"
    if resume:
        base_cmd += " --resume"
    
    # Combine setup and training command
    training_cmd = " && ".join(setup_commands) + " && " + base_cmd
    
    # Create a script file for training
    script_creation = f"echo '{training_cmd}' > ~/yanomami_project/run_distributed_training.sh && chmod +x ~/yanomami_project/run_distributed_training.sh"
    
    # Setup tmux session with the training script
    tmux_commands = [
        "tmux new-session -d -s yanomami_training",
        "tmux send-keys -t yanomami_training 'cd ~/yanomami_project && ./run_distributed_training.sh' C-m"
    ]
    
    tmux_script = "; ".join(tmux_commands)
    
    try:
        # First create the training script
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip_address}", script_creation],
            check=True
        )
        
        # Then start the training in tmux
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip_address}", tmux_script],
            check=True
        )
        
        print("\nDistributed training started successfully across 8 GPUs in a tmux session")
        print("To view training progress, connect to the instance and run:")
        print("  tmux attach -t yanomami_training")
        print("\nTo detach from tmux without stopping training, press: Ctrl+B, then D")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error starting training: {e}")
        return False

def monitor_training(ip_address):
    """Monitor training progress.
    
    Args:
        ip_address (str): Instance IP address
    """
    print("\nMonitoring training progress (press Ctrl+C to stop monitoring)...")
    
    try:
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip_address}", "tail -f /lambda_storage/logs/*.log"],
            check=False  # Don't check return code since we expect Ctrl+C to interrupt
        )
    except KeyboardInterrupt:
        print("\nStopped monitoring. Training is still running on the instance.")
    except Exception as e:
        print(f"Error monitoring training: {e}")
