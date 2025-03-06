# Lambda Cloud Configuration for Yanomami Translation Model Training
#
# This file contains configuration settings for running the Yanomami translation
# model training on Lambda Cloud (LambLabs) GPU instances

import os
from pathlib import Path

class LambdaConfig:
    def __init__(self):
        # Lambda instance settings
        self.instance_type = "gpu_8x_a100"  # 8x A100 (40 GB SXM4) for faster training
        self.region = "us-east-1"  # Region for deployment
        
        # Storage settings
        self.storage_dir = "/lambda_storage"  # Lambda persistent storage directory
        self.checkpoint_dir = os.path.join(self.storage_dir, "checkpoints")
        self.model_output_dir = os.path.join(self.storage_dir, "enhanced_yanomami_translator")
        self.log_dir = os.path.join(self.storage_dir, "logs")
        self.visualization_output_dir = os.path.join(self.storage_dir, "visualization_results")
        
        # Dataset settings
        self.dataset_dir = os.path.join(self.storage_dir, "yanomami_dataset")
        
        # Docker settings
        self.docker_image = "lambdal/lambda-stack:latest"  # Lambda's PyTorch image
        
        # Training optimization for 8x A100 GPUs
        self.batch_size = 32  # Significantly increased batch size for 8x A100 GPUs
        self.gradient_accumulation_steps = 1  # Reduced since we have more GPU memory
        self.use_mixed_precision = True  # Enable mixed precision training
        self.distributed_training = True  # Enable distributed training across multiple GPUs
        
        # Create required directories
        def ensure_dirs_exist(self):
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(self.model_output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            Path(self.visualization_output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)
