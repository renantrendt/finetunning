# Enhanced GPT-2 Fine-tuning for Yanomami-English Translation - Lambda Cloud Version
# 
# This script implements an improved version of the Yanomami-English translator
# with optimizations for Lambda Cloud GPU instances.

import os
import json
import torch
import numpy as np
import time
import psutil
import platform
import gc
import shutil
import importlib.util
import unicodedata
from datetime import datetime
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    AdamW, 
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.model_selection import train_test_split
import logging
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Import from the same package
from yanomami_trainer.visualization_utils import TrainingVisualizer

# Import the tokenizer enhancement module
from yanomami_tokenizer.tokenizer_enhancement import (
    enhance_tokenizer,
    load_enhanced_tokenizer,
    SPECIAL_CHAR_WORDS,
    replace_special_chars
)

# Get logger
logger = logging.getLogger(__name__)

# Configuration class with Lambda Cloud optimizations
class TranslatorConfig:
    def __init__(self):
        # Check if running on Lambda Cloud
        self.is_lambda = os.environ.get("YANOMAMI_LAMBDA_TRAINING", "0") == "1"
        
        # Debug mode for testing
        self.debug_mode = os.environ.get("YANOMAMI_DEBUG", "0") == "1"
        
        # Data settings
        if self.is_lambda:
            # Use the current directory structure
            current_dir = os.getcwd()
            self.dataset_path = os.environ.get("YANOMAMI_DATASET_DIR", os.path.join(current_dir, "yanomami_dataset"))
            # Ensure path ends with a slash
            if not self.dataset_path.endswith('/'):
                self.dataset_path += '/'
            logger.info(f"Using dataset path: {self.dataset_path}")
        else:
            self.dataset_path = '/Users/renanserrano/CascadeProjects/Yanomami/finetunning/yanomami_dataset/'
        
        self.dataset_files = glob.glob(os.path.join(self.dataset_path, '*.jsonl'))
        
        # Model settings
        self.model_name = "gpt2"
        
        if self.is_lambda:
            # Use the current directory structure for all paths
            current_dir = os.getcwd()
            self.model_output_dir = os.environ.get("YANOMAMI_MODEL_OUTPUT_DIR", os.path.join(current_dir, "enhanced_yanomami_translator"))
            self.checkpoint_dir = os.environ.get("YANOMAMI_CHECKPOINT_DIR", os.path.join(current_dir, "checkpoints"))
            self.log_dir = os.environ.get("YANOMAMI_LOG_DIR", os.path.join(current_dir, "logs"))
            self.visualization_output_dir = os.environ.get("YANOMAMI_VISUALIZATION_DIR", os.path.join(current_dir, "visualization_results"))
            
            # Log the paths being used
            logger.info(f"Using model output directory: {self.model_output_dir}")
            logger.info(f"Using checkpoint directory: {self.checkpoint_dir}")
            logger.info(f"Using log directory: {self.log_dir}")
            logger.info(f"Using visualization directory: {self.visualization_output_dir}")
        else:
            self.model_output_dir = "./enhanced_yanomami_translator"
            self.checkpoint_dir = "./checkpoints"
            self.log_dir = "./logs"
            self.visualization_output_dir = "./visualization_results"
        
        # Training hyperparameters - optimized for Lambda Cloud GPUs
        if self.is_lambda:
            self.batch_size = int(os.environ.get("YANOMAMI_BATCH_SIZE", "8"))
            self.gradient_accumulation_steps = int(os.environ.get("YANOMAMI_GRAD_ACCUM_STEPS", "4"))
            self.use_mixed_precision = os.environ.get("YANOMAMI_MIXED_PRECISION", "1") == "1"
        else:
            self.batch_size = 4
            self.gradient_accumulation_steps = 8
            self.use_mixed_precision = True
        
        self.num_epochs = 5
        self.learning_rate = 5e-5
        self.warmup_ratio = 0.1
        self.max_grad_norm = 1.0
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        
        # Tokenizer settings
        self.max_length = 128
        self.padding = "max_length"
        self.truncation = True
        
        # Generation settings - Optimized to reduce repetition
        self.max_gen_length = 50
        self.temperature = 0.9
        self.top_p = 0.92
        self.top_k = 40
        self.num_beams = 5
        self.do_sample = True
        self.repetition_penalty = 1.5
        self.no_repeat_ngram_size = 3
        
        # Early stopping
        self.patience = 3
        self.min_delta = 0.005
        
        # Scheduler type: 'linear' or 'cosine'
        self.scheduler_type = 'cosine'
        
        # Evaluation frequency (in steps)
        self.eval_steps = 500
        
        # Save frequency (in steps)
        self.save_steps = 1000
        
        # HellaSwag evaluation settings
        self.enable_hellaswag_eval = True
        self.hellaswag_eval_epochs = [1, 3, 5]
        self.hellaswag_num_examples = 100
        self.hellaswag_compare_baselines = True
        
        # Visualization settings
        self.enable_visualizations = True
        self.plot_at_epochs = [1, 3, 5]
        self.plot_at_end = True
        self.plot_batch_interval = 500
        self.baseline_precision = {
            "GPT-2": 0.42,
            "GPT-3": 0.68
        }
        
        # Multi-phase training settings
        self.enable_multi_phase = True
        self.phases = [
            {
                'name': 'Phase 1: Basic vocabulary',
                'dataset_files': ['combined-ok-translations.jsonl'],
                'learning_rate': 5e-5,
                'num_epochs': 5
            },
            {
                'name': 'Phase 2: Grammar and structure',
                'dataset_files': ['grammar-plural.jsonl', 'grammar-verb.jsonl'],
                'learning_rate': 3e-5,
                'num_epochs': 8
            },
            {
                'name': 'Phase 3: Advanced phrases and usage',
                'dataset_files': ['combined-ok-phrases-english-to-yanomami.jsonl', 'combined-ok-phrases-yanomami-to-english.jsonl', 'combined-ok-how-to-p1.jsonl', 'combined-ok-how-to-p2.jsonl'],
                'learning_rate': 2e-5,
                'num_epochs': 5
            }
        ]
        
        # Resume training flag
        self.resume_training = os.environ.get("YANOMAMI_RESUME_TRAINING", "0") == "1"
        
        # Create required directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.visualization_output_dir).mkdir(parents=True, exist_ok=True)

# Import the original functions from improvements_finetuning.py
from yanomami_trainer.improvements_finetuning import (
    load_jsonl,
    prepare_data_for_training,
    tokenize_function,
    GPT2Dataset,
    EarlyStopping,
    plot_training_metrics,
    generate_translation,
    clean_translation_output,
    load_yanomami_translator,
    evaluate_model,
    run_hellaswag_evaluation,
    test_translations
)

# Main function with Lambda optimizations
def main():
    # Initialize configuration
    config = TranslatorConfig()
    
    # Log Lambda-specific information if running on Lambda
    if config.is_lambda:
        logger.info("Running on Lambda Cloud with optimized settings")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
        logger.info(f"Mixed precision: {config.use_mixed_precision}")
        logger.info(f"Model output directory: {config.model_output_dir}")
        logger.info(f"Checkpoint directory: {config.checkpoint_dir}")
        logger.info(f"Dataset path: {config.dataset_path}")
    
    # Log system information
    logger.info(f"{'='*30} YANOMAMI TRANSLATION MODEL TRAINING {'='*30}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"Available CPU cores: {psutil.cpu_count(logical=True)}")
    logger.info(f"System memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info(f"MPS (Apple Silicon) device available")
    else:
        logger.info(f"Running on CPU only")
    
    # Import the original main function and call it
    from yanomami_trainer.improvements_finetuning import main as original_main
    # The original main function doesn't accept a config parameter, it creates its own config
    original_main()

# If this script is run directly, call the main function
if __name__ == "__main__":
    main()
