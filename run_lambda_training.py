# Lambda Cloud Training Runner for Yanomami Translation Model
#
# This script handles the setup and execution of Yanomami translation model training
# on Lambda Cloud GPU instances using the Lambda Cloud API.

import os
import sys
import argparse
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from lambda_api import LambdaCloudAPI, get_or_create_ssh_key, wait_for_ssh, transfer_files, setup_environment, start_training, monitor_training

# Load environment variables from .env file
load_dotenv()

# Configure argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run Yanomami training on Lambda Cloud")
    parser.add_argument("--api-key", type=str, help="Lambda Cloud API key")
    parser.add_argument("--instance-type", type=str, default="gpu_1x_a10", 
                        help="Lambda instance type (e.g., gpu_1x_a10, gpu_1x_a100)")
    parser.add_argument("--region", type=str, default=None,
                        help="Lambda region (e.g., us-east-1)")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only setup Lambda instance without starting training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--list-instances", action="store_true",
                        help="List running instances and exit")
    parser.add_argument("--list-instance-types", action="store_true",
                        help="List available instance types and exit")
    parser.add_argument("--terminate", type=str,
                        help="Terminate instance with the given ID and exit")
    parser.add_argument("--connect-ip", type=str,
                        help="Connect to an existing instance by IP address")
    return parser.parse_args()

# Display available instance types
def display_instance_types(instance_types):
    print("\nAvailable Lambda Cloud instance types:")
    print("-" * 100)
    print(f"{'Instance Type':<20} {'Region':<15} {'GPU':<15} {'Memory':<10} {'vCPUs':<8} {'Price/hr':<10} {'Status':<10}")
    print("-" * 100)
    
    # Print raw data for debugging
    print(f"Response structure: {type(instance_types)}")
    
    # Handle list response format (new API)
    if isinstance(instance_types, list):
        for instance in instance_types:
            name = instance.get("name", "N/A")
            description = instance.get("description", "")
            regions = instance.get("regions_with_capacity_available", [])
            regions_str = ", ".join(regions[:2]) if regions else "None"
            
            # Extract GPU info from name or description
            gpu_info = "N/A"
            if "a100" in name.lower() or "a100" in description.lower():
                gpu_info = "A100"
            elif "a10" in name.lower() or "a10" in description.lower():
                gpu_info = "A10"
            elif "v100" in name.lower() or "v100" in description.lower():
                gpu_info = "V100"
            
            # Extract memory and CPU info if available
            memory = "N/A"
            vcpus = "N/A"
            if description:
                # Try to extract memory info
                memory_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:GB|TB)\s*RAM", description, re.IGNORECASE)
                if memory_match:
                    memory = f"{memory_match.group(1)} GB"
                
                # Try to extract CPU info
                cpu_match = re.search(r"(\d+)\s*CPU\s*cores", description, re.IGNORECASE)
                if cpu_match:
                    vcpus = cpu_match.group(1)
            
            # Get price
            price = instance.get("price_cents_per_hour", 0) / 100
            
            # Determine status
            status = "✅ Available" if regions else "❌ Unavailable"
            
            # Print instance type information
            print(f"{name:<20} {regions_str[:14]:<15} {gpu_info:<15} {memory:<10} {vcpus:<8} ${price:<9.2f} {status:<10}")
            
            # Highlight 8x A100 instance if found
            if "8x" in name.lower() and "a100" in name.lower():
                print(f"  *** RECOMMENDED FOR TRAINING: {name} ***")
            
            # Print description if available
            if description:
                print(f"  Description: {description}")
    
    # Handle dictionary response format (old API)
    elif isinstance(instance_types, dict):
        data = instance_types.get("data", {})
        if not data:
            print("No instance types found in response")
            return
            
        for region_name, instance_types_in_region in data.items():
            for instance_type_name, instance_info in instance_types_in_region.items():
                # Get instance type details
                instance_type_data = instance_info.get("instance_type", {})
                specs = instance_type_data.get("specs", {})
                
                # Extract information
                gpu_name = specs.get("gpu_name", "N/A")
                memory = f"{specs.get('memory_gib', 'N/A')} GB"
                vcpus = specs.get("vcpus", "N/A")
                price = instance_type_data.get("price_cents_per_hour", 0) / 100
                availability = instance_type_data.get("availability", "")
                
                # Determine status
                status = "✅ Available" if availability == "available" else "❌ Unavailable"
                
                # Print instance type information
                print(f"{instance_type_name:<20} {region_name:<15} {gpu_name:<15} {memory:<10} {vcpus:<8} ${price:<9.2f} {status:<10}")
                
                # Print additional information if available
                description = specs.get("description", "")
                if description:
                    print(f"  Description: {description}")
    else:
        print(f"Unexpected response type: {type(instance_types)}")

# Display running instances
def display_instances(instances):
    if not instances.get("data"):
        print("\nNo running instances found.")
        return
    
    print("\nRunning Lambda Cloud instances:")
    print("-" * 100)
    print(f"{'Instance ID':<15} {'Name':<20} {'Type':<15} {'Region':<15} {'IP Address':<15} {'Status':<10}")
    print("-" * 100)
    
    for instance in instances.get("data", []):
        instance_id = instance.get("id", "N/A")
        name = instance.get("name", "N/A")
        instance_type = instance.get("instance_type", {}).get("name", "N/A")
        region = instance.get("region", {}).get("name", "N/A")
        ip = instance.get("ip", "N/A")
        status = instance.get("status", "N/A")
        
        print(f"{instance_id:<15} {name:<20} {instance_type:<15} {region:<15} {ip:<15} {status:<10}")

# Connect to existing instance
def connect_to_instance(ip_address):
    """Connect to an existing Lambda Cloud instance and prepare it for training.
    
    Args:
        ip_address (str): IP address of the instance
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print(f"Connecting to existing Lambda Cloud instance at {ip_address}...")
    
    # Wait for SSH to be available
    if not wait_for_ssh(ip_address):
        print("Failed to establish SSH connection. Please check your instance status.")
        return 1
    
    # Transfer files
    local_dir = os.path.dirname(os.path.abspath(__file__))
    if not transfer_files(ip_address, local_dir, "~/yanomami_project"):
        print("Failed to transfer files. Please check your SSH connection.")
        return 1
    
    # Setup environment
    if not setup_environment(ip_address):
        print("Failed to set up environment. Please check the instance logs.")
        return 1
    
    # Start training
    if not start_training(ip_address, resume=False):
        print("Failed to start training. Please check the instance logs.")
        return 1
    
    # Monitor training
    monitor_training(ip_address)
    
    print("\nTraining is running on the Lambda Cloud instance.")
    print(f"You can connect to the instance with: ssh ubuntu@{ip_address}")
    print("To view training progress: tmux attach -t yanomami_training")
    
    return 0

# Main function
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load Lambda configuration
    from lambda_config import LambdaConfig
    lambda_config = LambdaConfig()
    
    # Check if we need to connect to an existing instance
    if args.connect_ip:
        return connect_to_instance(args.connect_ip)
    
    # Override instance type from arguments if provided
    if args.instance_type:
        instance_type = args.instance_type
    else:
        instance_type = lambda_config.instance_type
        print(f"Using instance type from config: {instance_type}")
    
    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        print("Lambda Cloud API key is required. Set the LAMBDA_API_KEY environment variable or use --api-key.")
        print("You can create an API key at: https://cloud.lambdalabs.com/api-keys")
        return 1
    
    # Initialize API client
    try:
        api = LambdaCloudAPI(api_key)
    except Exception as e:
        print(f"Error initializing Lambda Cloud API client: {e}")
        return 1
    
    # List instances and exit if requested
    if args.list_instances:
        try:
            instances = api.list_instances()
            display_instances(instances)
            return 0
        except Exception as e:
            print(f"Error listing instances: {e}")
            return 1
    
    # List instance types and exit if requested
    if args.list_instance_types:
        try:
            instance_types = api.list_instance_types()
            display_instance_types(instance_types)
            return 0
        except Exception as e:
            print(f"Error listing instance types: {e}")
            return 1
    
    # Terminate instance and exit if requested
    if args.terminate:
        try:
            print(f"Terminating instance {args.terminate}...")
            api.terminate_instance(args.terminate)
            print("Instance terminated successfully.")
            return 0
        except Exception as e:
            print(f"Error terminating instance: {e}")
            return 1
    
    # Get or create SSH key
    try:
        ssh_key_id = get_or_create_ssh_key(api)
    except Exception as e:
        print(f"Error setting up SSH key: {e}")
        return 1
    
    # Launch instance
    try:
        print(f"\nLaunching Lambda Cloud instance ({instance_type})...")
        instance = api.launch_instance(
            instance_type=instance_type,
            name="yanomami-training-8xa100",
            ssh_key_ids=[ssh_key_id],
            region=args.region
        )
        
        instance_data = instance.get("data", {})
        instance_id = instance_data.get("id")
        ip_address = instance_data.get("ip")
        instance_type = instance_data.get("instance_type", {}).get("name")
        region = instance_data.get("region", {}).get("name")
        
        print("\nInstance launched successfully!")
        print(f"Instance ID: {instance_id}")
        print(f"Instance Type: {instance_type}")
        print(f"Region: {region}")
        print(f"IP Address: {ip_address}")
        print(f"SSH Command: ssh ubuntu@{ip_address}")
    except Exception as e:
        print(f"Error launching instance: {e}")
        return 1
    
    # Wait for SSH to be available
    if not wait_for_ssh(ip_address):
        print("Failed to establish SSH connection. Please check your instance status.")
        return 1
    
    # Transfer files
    local_dir = os.path.dirname(os.path.abspath(__file__))
    if not transfer_files(ip_address, local_dir, "~/yanomami_project"):
        print("Failed to transfer files. Please check your SSH connection.")
        return 1
    
    # Setup environment
    if not setup_environment(ip_address):
        print("Failed to set up environment. Please check the instance logs.")
        return 1
    
    # Exit if setup only
    if args.setup_only:
        print("\nSetup completed. Instance is ready for training.")
        print(f"Connect to the instance with: ssh ubuntu@{ip_address}")
        return 0
    
    # Start training
    if not start_training(ip_address, resume=args.resume):
        print("Failed to start training. Please check the instance logs.")
        return 1
    
    # Monitor training
    monitor_training(ip_address)
    
    print("\nTraining is running on the Lambda Cloud instance.")
    print(f"You can connect to the instance with: ssh ubuntu@{ip_address}")
    print("To view training progress: tmux attach -t yanomami_training")
    print("\nWhen you're done, you can terminate the instance with:")
    print(f"python run_lambda_training.py --terminate {instance_id}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
