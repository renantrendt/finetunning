# Lambda Cloud Training Runner for Yanomami Translation Model
#
# This script handles the setup and execution of Yanomami translation model training
# on Lambda Cloud GPU instances. It manages data transfer, environment setup,
# and training job submission.

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from lambda_config import LambdaConfig

# Configure argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run Yanomami training on Lambda Cloud")
    parser.add_argument("--instance-type", type=str, default=None, 
                        help="Lambda instance type (e.g., gpu_1x_a10, gpu_1x_a100)")
    parser.add_argument("--region", type=str, default=None,
                        help="Lambda region (e.g., us-east-1)")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only setup Lambda instance without starting training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with verbose output")
    return parser.parse_args()

# Check if Lambda CLI is installed
def check_lambda_cli():
    try:
        subprocess.run(["lambda", "version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Install Lambda CLI if not present
def install_lambda_cli():
    print("Installing Lambda CLI...")
    subprocess.run(["pip", "install", "lambdacloud"], check=True)
    print("Lambda CLI installed successfully")

# Authenticate with Lambda Cloud
def authenticate_lambda():
    print("\nPlease authenticate with Lambda Cloud:")
    print("If you don't have an API key, create one at: https://cloud.lambdalabs.com/api-keys")
    subprocess.run(["lambda", "login"], check=True)

# Check available instance types
def check_available_instances():
    print("\nChecking available Lambda Cloud instances...")
    result = subprocess.run(["lambda", "instance-types", "list", "--region", "all", "--json"], 
                           check=True, capture_output=True, text=True)
    instances = json.loads(result.stdout)
    
    print("\nAvailable instance types:")
    for region, types in instances.items():
        print(f"\nRegion: {region}")
        for instance_type, details in types.items():
            availability = details.get("instance_type", {}).get("availability", "unknown")
            price = details.get("instance_type", {}).get("price_cents_per_hour", 0) / 100
            specs = details.get("instance_type", {}).get("specs", {})
            gpu_name = specs.get("gpu_name", "Unknown GPU")
            gpu_memory = specs.get("gpu_memory_gib", "Unknown")
            vcpus = specs.get("vcpus", "Unknown")
            
            status = "✅ Available" if availability == "available" else "❌ Unavailable"
            print(f"  {instance_type}: {gpu_name} ({gpu_memory} GB) - ${price}/hr - {status}")
    
    return instances

# Launch Lambda instance
def launch_instance(config):
    print(f"\nLaunching Lambda instance {config.instance_type} in {config.region}...")
    
    # Create launch command
    cmd = [
        "lambda", "instance", "launch",
        "--instance-type", config.instance_type,
        "--region", config.region,
        "--name", "yanomami-training",
        "--ssh",
        "--json"
    ]
    
    # Launch the instance
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    instance_info = json.loads(result.stdout)
    
    instance_id = instance_info.get("id")
    ip_address = instance_info.get("ip")
    ssh_command = f"ssh ubuntu@{ip_address}"
    
    print(f"\nInstance launched successfully!")
    print(f"Instance ID: {instance_id}")
    print(f"IP Address: {ip_address}")
    print(f"SSH Command: {ssh_command}")
    
    return instance_info

# Transfer files to Lambda instance
def transfer_files(instance_info, local_dir):
    ip_address = instance_info.get("ip")
    remote_user = "ubuntu"
    
    print(f"\nTransferring files to Lambda instance ({ip_address})...")
    
    # Create remote directories
    ssh_cmd = f"ssh {remote_user}@{ip_address} 'mkdir -p ~/yanomami_project'"
    subprocess.run(ssh_cmd, shell=True, check=True)
    
    # Use rsync to transfer files
    rsync_cmd = [
        "rsync", "-avz", "--exclude", ".git", "--exclude", "__pycache__",
        "--exclude", "*.pyc", "--exclude", "logs", "--exclude", "checkpoints",
        f"{local_dir}/", f"{remote_user}@{ip_address}:~/yanomami_project/"
    ]
    
    subprocess.run(rsync_cmd, check=True)
    print("Files transferred successfully")

# Setup environment on Lambda instance
def setup_environment(instance_info):
    ip_address = instance_info.get("ip")
    remote_user = "ubuntu"
    
    print(f"\nSetting up environment on Lambda instance...")
    
    # Install dependencies
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
    ssh_cmd = f"ssh {remote_user}@{ip_address} '{setup_script}'"
    
    subprocess.run(ssh_cmd, shell=True, check=True)
    print("Environment setup completed successfully")

# Start training on Lambda instance
def start_training(instance_info, resume=False):
    ip_address = instance_info.get("ip")
    remote_user = "ubuntu"
    
    print(f"\nStarting training on Lambda instance...")
    
    # Prepare training command
    training_cmd = "cd ~/yanomami_project && python train_yanomami.py"
    if resume:
        training_cmd += " --resume"
    
    # Run in tmux to keep it running after disconnection
    tmux_commands = [
        "tmux new-session -d -s yanomami_training",
        f"tmux send-keys -t yanomami_training '{training_cmd}' C-m"
    ]
    
    tmux_script = "; ".join(tmux_commands)
    ssh_cmd = f"ssh {remote_user}@{ip_address} '{tmux_script}'"
    
    subprocess.run(ssh_cmd, shell=True, check=True)
    
    print("\nTraining started successfully in a tmux session")
    print("To view training progress, connect to the instance and run:")
    print("  tmux attach -t yanomami_training")
    print("\nTo detach from tmux without stopping training, press: Ctrl+B, then D")

# Monitor training progress
def monitor_training(instance_info):
    ip_address = instance_info.get("ip")
    remote_user = "ubuntu"
    
    print(f"\nMonitoring training progress (press Ctrl+C to stop monitoring)...")
    
    # Stream logs from the instance
    ssh_cmd = f"ssh {remote_user}@{ip_address} 'tail -f /lambda_storage/logs/*.log'"
    
    try:
        subprocess.run(ssh_cmd, shell=True)
    except KeyboardInterrupt:
        print("\nStopped monitoring. Training is still running on the instance.")

# Main function
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = LambdaConfig()
    
    # Override config with command line arguments if provided
    if args.instance_type:
        config.instance_type = args.instance_type
    if args.region:
        config.region = args.region
    
    # Check and install Lambda CLI if needed
    if not check_lambda_cli():
        install_lambda_cli()
    
    # Authenticate with Lambda Cloud
    authenticate_lambda()
    
    # Check available instances
    available_instances = check_available_instances()
    
    # Verify selected instance type is available
    region_instances = available_instances.get(config.region, {})
    instance_details = region_instances.get(config.instance_type, {})
    instance_availability = instance_details.get("instance_type", {}).get("availability", "")
    
    if instance_availability != "available":
        print(f"\nError: Instance type {config.instance_type} is not available in region {config.region}")
        print("Please select an available instance type from the list above")
        return 1
    
    # Launch Lambda instance
    instance_info = launch_instance(config)
    
    # Wait for SSH to be available
    print("\nWaiting for SSH to become available...")
    ip_address = instance_info.get("ip")
    ssh_available = False
    
    for _ in range(10):
        try:
            subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                           f"ubuntu@{ip_address}", "echo 'SSH connection established'"], 
                          check=True, capture_output=True)
            ssh_available = True
            break
        except subprocess.CalledProcessError:
            print("SSH not yet available, waiting...")
            time.sleep(10)
    
    if not ssh_available:
        print("\nError: Could not establish SSH connection to the instance")
        return 1
    
    # Transfer files to instance
    local_dir = os.path.dirname(os.path.abspath(__file__))
    transfer_files(instance_info, local_dir)
    
    # Setup environment
    setup_environment(instance_info)
    
    # Exit if setup only
    if args.setup_only:
        print("\nSetup completed. Instance is ready for training.")
        print(f"Connect to the instance with: ssh ubuntu@{ip_address}")
        return 0
    
    # Start training
    start_training(instance_info, resume=args.resume)
    
    # Monitor training
    monitor_training(instance_info)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
