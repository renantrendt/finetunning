# Lambda Cloud Configuration for Yanomami Translation Model Training
#
# This file contains configuration settings for running the Yanomami translation
# model training on Lambda Cloud (LambLabs) GPU instances

import os
from pathlib import Path

class LambdaConfig:
    def __init__(self):
        # Lambda instance settings
        # GPU configuration with fallback options
        self.gpu_configs = [
            "gpu_8x_a100",  # First choice: 8x A100 (40 GB SXM4)
            "gpu_1x_a100",  # Second choice: 1x A100
            "gpu_1x_mps"    # Third choice: MPS GPU
        ]
        self.region = "us-east-1"  # Region for deployment
        
        # Training configurations for different GPU types
        self.gpu_training_configs = {
            "gpu_8x_a100": {
                "batch_size": 32,
                "gradient_accumulation_steps": 1,
                "use_mixed_precision": True,
                "distributed_training": True,
                "num_epochs": 5
            },
            "gpu_1x_a100": {
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "use_mixed_precision": True,
                "distributed_training": False,
                "num_epochs": 8
            },
            "gpu_1x_mps": {
                "batch_size": 4,
                "gradient_accumulation_steps": 8,
                "use_mixed_precision": True,
                "distributed_training": False,
                "num_epochs": 10
            }
        }
        
        # Storage settings - Use dynamic paths that work both locally and on Lambda
        self.is_lambda = os.getenv('LAMBDA_TASK_ROOT') is not None
        self.storage_dir = '/yanomami_project' if self.is_lambda else os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.checkpoint_dir = os.path.join(self.storage_dir, "checkpoints")
        self.model_output_dir = os.path.join(self.storage_dir, "enhanced_yanomami_translator")
        self.log_dir = os.path.join(self.storage_dir, "logs")
        self.visualization_output_dir = os.path.join(self.storage_dir, "visualization_results")
        
        # Dataset settings
        self.dataset_dir = os.path.join(self.storage_dir, "yanomami_dataset")
        
        # Docker settings
        self.docker_image = "lambdal/lambda-stack:latest"  # Lambda's PyTorch image
        
        # Default to most conservative training settings (will be updated based on available GPU)
        self.batch_size = 4
        self.gradient_accumulation_steps = 8
        self.use_mixed_precision = True
        self.distributed_training = False
        
    def configure_for_gpu(self, available_gpu_type):
        """Configure training parameters based on available GPU type"""
        if available_gpu_type in self.gpu_training_configs:
            config = self.gpu_training_configs[available_gpu_type]
            self.batch_size = config["batch_size"]
            self.gradient_accumulation_steps = config["gradient_accumulation_steps"]
            self.use_mixed_precision = config["use_mixed_precision"]
            self.distributed_training = config["distributed_training"]
            return True
        return False
        
    # Create required directories
    def ensure_dirs_exist(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.visualization_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)
