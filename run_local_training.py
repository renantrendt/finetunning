#!/usr/bin/env python3
# Yanomami Translation Model Local Training Script
#
# This script provides a convenient way to run the Yanomami translation model training
# on local hardware with optimized settings for your machine.

import os
import sys
import logging
import argparse
import torch
from pathlib import Path
from yanomami_trainer.improvements_finetuning import main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('yanomami_training.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Yanomami translation model locally")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with verbose output")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training (default: 4)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train (default: 5)")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Enable mixed precision training")
    return parser.parse_args()

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "yanomami_dataset",
        "checkpoints",
        "logs",
        "enhanced_yanomami_translator",
        "visualization_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def main_local():
    """Main function for local training"""
    # Parse arguments
    args = parse_args()
    
    # Ensure required directories exist
    ensure_directories()
    
    logger.info("Starting Yanomami Translation Model Training Locally")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set environment variables for training configuration
    os.environ["YANOMAMI_BATCH_SIZE"] = str(args.batch_size)
    os.environ["YANOMAMI_GRAD_ACCUM_STEPS"] = str(8)  # Default gradient accumulation steps
    os.environ["YANOMAMI_MIXED_PRECISION"] = "1" if args.mixed_precision else "0"
    
    # Set paths using relative directories
    current_dir = os.path.abspath(os.path.dirname(__file__))
    os.environ["YANOMAMI_DATASET_DIR"] = os.path.join(current_dir, "yanomami_dataset")
    os.environ["YANOMAMI_CHECKPOINT_DIR"] = os.path.join(current_dir, "checkpoints")
    os.environ["YANOMAMI_MODEL_OUTPUT_DIR"] = os.path.join(current_dir, "enhanced_yanomami_translator")
    os.environ["YANOMAMI_LOG_DIR"] = os.path.join(current_dir, "logs")
    os.environ["YANOMAMI_VISUALIZATION_DIR"] = os.path.join(current_dir, "visualization_results")
    
    if args.resume:
        os.environ["YANOMAMI_RESUME_TRAINING"] = "1"
    
    if args.debug:
        os.environ["YANOMAMI_DEBUG"] = "1"
    
    # Run the training
    main()
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main_local()
