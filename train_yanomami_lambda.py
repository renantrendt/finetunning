# Yanomami Translation Model Training Script for Lambda Cloud
#
# This script serves as the entry point for training the Yanomami-English translation model
# on Lambda Cloud GPU instances. It integrates with Lambda-specific configurations
# and optimizations for better performance on cloud GPUs, including distributed training
# on multiple A100 GPUs.

import logging
import sys
import os
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
from yanomami_trainer.lambda_improvements_finetuning import main
from lambda_config import LambdaConfig

# Configure argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train Yanomami translation model on Lambda Cloud")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with verbose output")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training across multiple GPUs")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by torch.distributed.launch)")
    return parser.parse_args()

# Configure logging
def setup_logging(lambda_config):
    # Create log directory if it doesn't exist
    Path(lambda_config.log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = os.path.join(lambda_config.log_dir, "yanomami_training_lambda.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize distributed training
def init_distributed_training(args, logger):
    if args.distributed:
        # Set the device according to local_rank
        local_rank = args.local_rank
        if local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            
            # Initialize distributed process group
            dist.init_process_group(backend="nccl")
            
            logger.info(f"Initialized distributed training on rank {dist.get_rank()} of {dist.get_world_size()}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    return device

# Main function
def lambda_main():
    # Parse arguments
    args = parse_args()
    
    # Load Lambda configuration
    lambda_config = LambdaConfig()
    
    # Ensure required directories exist
    lambda_config.ensure_dirs_exist()
    
    # Setup logging
    logger = setup_logging(lambda_config)
    
    # Only log from main process in distributed training
    is_main_process = not args.distributed or (args.distributed and args.local_rank == 0)
    
    if is_main_process:
        logger.info("Starting Yanomami Translation Model Training on Lambda Cloud")
        logger.info(f"Using instance type: {lambda_config.instance_type}")
        logger.info(f"Using storage directory: {lambda_config.storage_dir}")
        if args.distributed:
            logger.info("Using distributed training across multiple GPUs")
    
    # Initialize distributed training if requested
    device = init_distributed_training(args, logger)
    
    if is_main_process:
        logger.info(f"Using device: {device}")
    
    # Override training configuration with Lambda-specific settings
    os.environ["YANOMAMI_LAMBDA_TRAINING"] = "1"
    os.environ["YANOMAMI_BATCH_SIZE"] = str(lambda_config.batch_size)
    os.environ["YANOMAMI_GRAD_ACCUM_STEPS"] = str(lambda_config.gradient_accumulation_steps)
    os.environ["YANOMAMI_MIXED_PRECISION"] = str(int(lambda_config.use_mixed_precision))
    os.environ["YANOMAMI_CHECKPOINT_DIR"] = lambda_config.checkpoint_dir
    os.environ["YANOMAMI_MODEL_OUTPUT_DIR"] = lambda_config.model_output_dir
    os.environ["YANOMAMI_LOG_DIR"] = lambda_config.log_dir
    os.environ["YANOMAMI_VISUALIZATION_DIR"] = lambda_config.visualization_output_dir
    os.environ["YANOMAMI_DATASET_DIR"] = lambda_config.dataset_dir
    
    # Set distributed training environment variables
    if args.distributed:
        os.environ["YANOMAMI_DISTRIBUTED_TRAINING"] = "1"
        os.environ["YANOMAMI_LOCAL_RANK"] = str(args.local_rank)
        os.environ["YANOMAMI_WORLD_SIZE"] = str(dist.get_world_size() if dist.is_initialized() else 1)
    
    if args.resume:
        os.environ["YANOMAMI_RESUME_TRAINING"] = "1"
    
    if args.debug:
        os.environ["YANOMAMI_DEBUG"] = "1"
    
    # Run the training
    main()
    
    # Clean up distributed training resources
    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()
    
    if is_main_process:
        logger.info("Training completed successfully")

if __name__ == "__main__":
    lambda_main()
