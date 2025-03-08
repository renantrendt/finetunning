# Enhanced GPT-2 Fine-tuning for Yanomami-English Translation
# 
# This script implements an improved version of the Yanomami-English translator
# with focus on better performance, training efficiency, and offline functionality.

import os
import re
import json
import torch
import numpy as np
import time
import psutil
import platform
import shutil
import difflib
import unicodedata
from datetime import datetime
from transformers import (
    AutoModelForSeq2SeqLM, 
    AdamW, 
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AutoTokenizer,
    M2M100ForConditionalGeneration
)
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.model_selection import train_test_split
import logging
from torch.amp import autocast  # Updated import path
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from pathlib import Path
import glob
# Import from the same package
from yanomami_trainer.visualization_utils import TrainingVisualizer

# Import the tokenizer enhancement module
from yanomami_tokenizer.tokenizer_enhancement import (
    # enhance_tokenizer,
    load_enhanced_tokenizer
)

# Lambda Cloud environment detection
def is_running_on_lambda():
    """Detect if code is running on Lambda Cloud"""
    # Check for Lambda environment variable (existing method)
    if os.environ.get("YANOMAMI_LAMBDA_TRAINING", "0") == "1":
        return True
    
    # Check for Lambda-specific environment variables
    if 'LAMBDA_TASK_ROOT' in os.environ or 'LAMBDA_RUNTIME_DIR' in os.environ:
        return True
    
    # Check for Lambda-specific system characteristics
    try:
        with open('/etc/os-release', 'r') as f:
            os_info = f.read()
            if 'lambda' in os_info.lower():
                return True
    except:
        pass
    
    return False

# GH200 detection function
def is_gh200_gpu():
    """Detect if running on GH200 Superchip"""
    if not torch.cuda.is_available():
        return False
    
    # Check for high memory (GH200 has very large memory)
    try:
        if torch.cuda.get_device_properties(0).total_memory > 100e9:  # >100GB suggests GH200
            return True
        
        # Check device name for GH200 identifiers
        device_name = torch.cuda.get_device_name(0).lower()
        return 'gh200' in device_name or 'grace hopper' in device_name
    except:
        return False

# Configure enhanced logging with timestamps and detailed formatting
log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f"training_{log_timestamp}.log"
log_dir = "logs"

# Create logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = os.path.join(log_dir, log_file)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode='w')  # Use mode='w' to overwrite the file
    ]
)

# Force output to be displayed in the terminal
print(f"Starting Yanomami-English translation model fine-tuning... (Logs: {log_path})")

# Get logger
logger = logging.getLogger(__name__)

# Log system information at startup
logger.info(f"{'='*30} YANOMAMI TRANSLATION MODEL TRAINING {'='*30}")
logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Python version: {platform.python_version()}")
logger.info(f"System: {platform.system()} {platform.release()}")
logger.info(f"CPU: {platform.processor()}")
logger.info(f"Available CPU cores: {psutil.cpu_count(logical=True)}")
logger.info(f"System memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
logger.info(f"PyTorch version: {torch.__version__}")

def get_device(log_info=True):
    if torch.cuda.is_available():
        device = "cuda"
        if log_info:
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        if log_info:
            logger.info(f"MPS (Apple Silicon) device available")
    else:
        device = "cpu"
        if log_info:
            logger.info(f"Running on CPU only")
    
    return device

get_device(log_info=True)  # This will log device information

# Configuration class for easy parameter management
class TranslatorConfig:
    def __init__(self):
        # Debug mode for testing
        self.debug_mode = os.environ.get("YANOMAMI_DEBUG", "0") == "1"
        
        # Data settings
        # Check if running on Lambda Cloud
        self.is_lambda = is_running_on_lambda()
        
        # Use the current directory structure or environment variable for all environments
        current_dir = os.getcwd()
        self.dataset_path = os.environ.get("YANOMAMI_DATASET_DIR", os.path.join(current_dir, "yanomami_dataset"))
        # Ensure path ends with a slash
        if not self.dataset_path.endswith('/'):
            self.dataset_path += '/'
        logging.info(f"Using dataset path: {self.dataset_path}")
        
        # Logging configuration
        self.logging_steps = 1  
            
        self.dataset_files = glob.glob(os.path.join(self.dataset_path, '*.jsonl'))
        logging.info(f"Found {len(self.dataset_files)} dataset files in {self.dataset_path}")
        
        # Model settings
        self.model_name = "facebook/m2m100_418M"  # Using the distilled 600M parameter version of NLLB
        self.model_type = "m2m100"  # Add model type to distinguish between different architectures
        self.loss_type = 'CrossEntropyLoss'  # Appropriate loss for seq2seq models
        
        # Source and target language codes for NLLB
        # NLLB uses specific language codes - 'eng_Latn' for English
        # For Yanomami, we'll use a close proxy or add it as a new language
        self.source_lang = "eng_Latn"  # English in Latin script
        self.target_lang = "eng_Latn" 
        
        # Use the current directory structure for all paths in all environments
        current_dir = os.getcwd()
        self.model_output_dir = os.environ.get("YANOMAMI_MODEL_OUTPUT_DIR", os.path.join(current_dir, "enhanced_yanomami_translator"))
        self.checkpoint_dir = os.environ.get("YANOMAMI_CHECKPOINT_DIR", os.path.join(current_dir, "checkpoints"))
        self.log_dir = os.environ.get("YANOMAMI_LOG_DIR", os.path.join(current_dir, "logs"))
        self.visualization_output_dir = os.environ.get("YANOMAMI_VISUALIZATION_DIR", os.path.join(current_dir, "visualization_results"))
        
        # Log the paths being used
        logging.info(f"Using model output directory: {self.model_output_dir}")
        logging.info(f"Using checkpoint directory: {self.checkpoint_dir}")
        
        # Training hyperparameters
        self.num_epochs = 10
        self.batch_size = 32
        self.gradient_accumulation_steps = 4  # Effective batch size = batch_size * gradient_accumulation_steps
        self.learning_rate = 5e-5
        self.warmup_ratio = 0.1
        self.max_grad_norm = 1.0
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        
        # Tokenizer settings
        self.max_length = 128
        self.padding = "max_length"
        self.truncation = True
        
        # Generation settings - Optimized for initial outputs with more lenient parameters
        self.max_gen_length = 128  # Keep max length for comprehensive translations
        self.min_length = 1  # More lenient minimum length to ensure we get outputs
        self.temperature = 0.2  # Slightly higher temperature for more variety
        self.top_p = 0.95  # More lenient nucleus sampling
        self.top_k = 100  # Broader vocabulary selection
        self.num_beams = 1  # Reduced beams for faster generation while maintaining quality
        self.do_sample = True  # Keep sampling enabled
        self.repetition_penalty = 1.1  # Slightly reduced penalty
        self.no_repeat_ngram_size = 2  # More lenient repetition control
        self.length_penalty = 0.8  # Slight preference for shorter outputs initially
        self.early_stopping = True  # Keep early stopping
        
        # Early stopping
        self.patience = 3
        self.min_delta = 0.005
        
        # Mixed precision
        self.use_mixed_precision = False
        
        # Scheduler type: 'linear' or 'cosine'
        self.scheduler_type = 'cosine'
        
        # Evaluation frequency (in steps)
        self.eval_steps = 500
        
        # Save frequency (in steps)
        self.save_steps = 1000
        
        # HellaSwag evaluation settings
        self.enable_hellaswag_eval = True  # Enable HellaSwag evaluation
        self.hellaswag_eval_epochs = [1, 3, 5]  # Epochs to run HellaSwag evaluation
        self.hellaswag_num_examples = 10  # Number of examples to evaluate
        self.hellaswag_compare_baselines = True  # Compare with baseline models
        
        # Visualization settings
        self.enable_visualizations = True  # Enable training visualizations
        self.visualization_output_dir = "./visualization_results"  # Directory to save visualizations
        self.plot_at_epochs = [1, 3, 5]  # Epochs to generate plots
        self.plot_at_end = True  # Generate plots at the end of training
        self.plot_batch_interval = 500  # Plot every N batches (0 to disable)
        self.baseline_precision = {  # Baseline precision values for comparison
            "GPT-2": 0.42,
            "GPT-3": 0.68
        }
        
        # Multi-phase training settings
        self.enable_multi_phase = True
        self.phases = [
            {
                'name': 'Phase 1: Basic vocabulary',
                'complexity_min': 0,
                'complexity_max': 30,  # Lower complexity examples
                'learning_rate': 5e-5,
                'num_epochs': 8
            },
            {
                'name': 'Phase 2: Grammar and structure',
                'complexity_min': 30,
                'complexity_max': 60,  # Medium complexity examples
                'learning_rate': 3e-5,
                'num_epochs': 8
            },
            {
                'name': 'Phase 3: Advanced phrases and usage',
                'complexity_min': 60,
                'complexity_max': 1000,  # Higher complexity examples
                'learning_rate': 2e-5,
                'num_epochs': 8
            }
        ]
        
        # Add this line to define the attribute
        self.test_translations_every_epoch = True 

# Helper function
def load_jsonl(file_path):
    import unicodedata
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                # Normalize Unicode before parsing JSON
                normalized_line = unicodedata.normalize('NFC', line)
                example = json.loads(normalized_line)
                
                # Normalize all string values in the example recursively
                def normalize_strings(obj):
                    if isinstance(obj, str):
                        return unicodedata.normalize('NFC', obj)
                    elif isinstance(obj, dict):
                        return {k: normalize_strings(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [normalize_strings(item) for item in obj]
                    else:
                        return obj
                
                # Apply normalization to the entire example
                normalized_example = normalize_strings(example)
                data.append(normalized_example)
            except json.JSONDecodeError as e:
                logger.warning(f"Error decoding JSON in {file_path}: {str(e)}")
                continue
    return data

# Helper function for tokenization
def tokenize_text(text, tokenizer, config):
    try:
        tokens = tokenizer(
            text,
            padding=config.padding,
            truncation=config.truncation,
            max_length=config.max_length,
            return_tensors="pt"  # Return PyTorch tensors directly
        )
        return tokens
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return None

# Helper function to prepare tokenized datasets
def prepare_tokenized_dataset(examples, tokenizer, config):
    tokenized_data = []
    
    # Process each example individually to avoid batch errors
    for example in examples:
        try:
            # Combine input and output for training
            combined_text = f"{example['input']} {example['output']}"
            
            # Tokenize the text
            try:
                tokens = tokenize_text(combined_text, tokenizer, config)
                
                # Add to tokenized data
                tokenized_data.append({
                    "input_ids": tokens["input_ids"][0],  # Remove batch dimension
                    "attention_mask": tokens["attention_mask"][0]  # Remove batch dimension
                })
                
            except Exception as e:
                logger.warning(f"Error tokenizing text: {str(e)}. Using dummy tokens.")
                # Use dummy tokens if tokenization fails
                tokenized_data.append({
                    "input_ids": torch.tensor([0, 0]),
                    "attention_mask": torch.tensor([1, 1])
                })
                
        except Exception as e:
            logger.warning(f"Error processing example: {str(e)}. Skipping.")
    
    logger.info(f"Prepared {len(tokenized_data)} tokenized examples out of {len(examples)} total examples")
    return tokenized_data

# Helper functions for text extraction and processing
def extract_word_from_query(query):
    """Extract the main word from a query about usage, pluralization, or verb conjugation.
    The word is expected to be marked with <WORD> tags in the dataset.
    """
    # Look for word in <WORD> tags
    import re
    word_pattern = r'<WORD>([^<]+)</WORD>'
    matches = re.findall(word_pattern, query)
    
    if matches:
        # Return the first word found in <WORD> tags
        return matches[0].strip()
    
    # Fallback: try to find word after common phrases (for robustness)
    query_lower = query.lower()
    for phrase in ['use the word', 'how do I use', 'plural of', 'how is the plural of', 'the verb', 'how is the verb']:
        if phrase in query_lower:
            phrase_index = query_lower.find(phrase)
            remaining_text = query[phrase_index + len(phrase):].strip()
            if remaining_text:
                return remaining_text.split()[0].strip('?.,"\'')
    
    # Last resort fallback - return the last word
    words = query.split()
    if words:
        return words[-1].strip('?.,"\'')
    
    return "unknown"

def extract_words_from_comparison(query):
    """Extract the words being compared from a query.
    The words are expected to be marked with <WORD> tags in the dataset.
    """
    # Look for words in <WORD> tags
    import re
    word_pattern = r'<WORD>([^<]+)</WORD>'
    matches = re.findall(word_pattern, query)
    
    if len(matches) >= 2:
        # Return the first two words found in <WORD> tags
        return [matches[0].strip(), matches[1].strip()]
    
    # Fallback: try to extract from "difference between X and Y" pattern
    query_lower = query.lower()
    if 'difference between' in query_lower:
        after_phrase = query_lower.split('difference between')[1].strip()
        if ' and ' in after_phrase:
            parts = after_phrase.split(' and ')
            return [parts[0].strip(), parts[1].split('?')[0].strip()]
    
    return ["word1", "word2"]

def get_usage_example(word):
    """Get usage examples for a word"""
    return f"Here are some examples of how to use '{word}' in Yanomami."

def get_comparison_info(words):
    """Get comparison information for words"""
    return f"The difference between '{words[0]}' and '{words[1]}' in Yanomami is..."

def get_plural_form(word):
    """Get plural form of a word"""
    return f"The plural form of '{word}' in Yanomami is..."

def get_conjugation_info(verb):
    """Get conjugation information for a verb"""
    return f"The conjugation of the verb '{verb}' in Yanomami is..."

# Processes raw examples from the dataset and transforms them into structured input-output pairs
# for different types of linguistic queries including translations, definitions, usage examples,
# comparisons, pluralization, and verb conjugation
def prepare_data_for_training(examples):
    processed_data = []
    for example in examples:  # Only one loop needed
        # Check example structure
        if 'messages' in example and len(example['messages']) >= 2:
            user_message = example['messages'][0]
            assistant_message = example['messages'][1]

            if user_message['role'] == 'user' and assistant_message['role'] == 'assistant':
                input_text = user_message['content']
                output_text = assistant_message['content']

                # Extract XML-like query type if present
                query_type = None
                if '<QUERY>' in input_text:
                    import re
                    query_match = re.search(r'<QUERY>(.*?)</QUERY>', input_text)
                    if query_match:
                        input_text = query_match.group(1)

                # Determine query type and process accordingly
                input_lower = input_text.lower()
                
                if 'translate' in input_lower and 'yanomami' in input_lower:
                    processed_data.append({
                        "input": f"English: {input_text} => Yanomami:",
                        "output": output_text
                    })
                elif 'translate' in input_lower and 'english' in input_lower:
                    processed_data.append({
                        "input": f"Yanomami: {input_text} => English:",
                        "output": output_text
                    })
                elif 'mean' in input_lower and 'yanomami' in input_lower:
                    processed_data.append({
                        "input": f"English: {input_text} => Yanomami:",
                        "output": output_text
                    })
                elif 'how do i use' in input_lower or 'use the word' in input_lower:
                    # Handle usage example queries - extract word from <WORD> tags
                    word = extract_word_from_query(input_text)
                    # Preserve the structured format from the dataset
                    processed_data.append({
                        "input": f"Usage of <WORD>{word}</WORD>:",
                        "output": output_text  # Use actual response instead of placeholder
                    })
                elif 'difference between' in input_lower:
                    # Handle comparison queries - extract both words from <WORD> tags
                    words = extract_words_from_comparison(input_text)
                    processed_data.append({
                        "input": f"Compare <WORD>{words[0]}</WORD> and <WORD>{words[1]}</WORD>:",
                        "output": output_text  # Use actual response instead of placeholder
                    })
                elif 'how is the plural of' in input_lower:
                    # Handle pluralization queries
                    word = extract_word_from_query(input_text)
                    processed_data.append({
                        "input": f"Plural of <WORD>{word}</WORD>:",
                        "output": output_text  # Use actual response instead of placeholder
                    })
                elif 'how is the verb' in input_lower:
                    # Handle verb conjugation queries
                    verb = extract_word_from_query(input_text)
                    processed_data.append({
                        "input": f"Conjugate verb <WORD>{verb}</WORD>:",
                        "output": output_text  # Use actual response instead of placeholder
                    })
                else:
                    # Default case - preserve any XML tags in the input
                    processed_data.append({
                        "input": f"English: {input_text} => Yanomami:",
                        "output": output_text
                    })
    
    return processed_data

# Converts text examples into token IDs using the enhanced tokenizer, combining input and output texts
# for language modeling training with proper error handling for robust processing
def tokenize_function(examples, tokenizer, config):
    # Initialize lists to store processed data
    try:
        input_texts = examples['input']
        output_texts = examples['output']
        
        # For training, we need both input and output together
        combined_texts = [f"{input_text} {output_text}" for input_text, output_text in zip(input_texts, output_texts)]
        
        # Return empty dict with required columns if no texts to process
        if not combined_texts:
            logger.warning("No valid examples found in batch.")
            return {
                "input_ids": [[0]],  # Add a dummy token ID
                "attention_mask": [[0]]  # Add a dummy attention mask
            }
        
        # Tokenize combined texts with error handling
        try:
            # Use enhanced tokenization if available
            if hasattr(tokenizer, 'enhanced_encode'):
                input_ids = tokenizer.enhanced_encode(combined_texts, return_tensors="pt")
                inputs = {"input_ids": input_ids}
            else:
                inputs = tokenize_text(combined_texts, tokenizer, config)
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            # Return dummy tokenized data with the required columns
            return {
                "input_ids": [[0] for _ in range(len(combined_texts))],
                "attention_mask": [[0] for _ in range(len(combined_texts))]
            }
            
    except Exception as e:
        logger.error(f"Error processing examples: {str(e)}")
        # Return dummy data with required columns
        return {
            "input_ids": [[0]],
            "attention_mask": [[0]]
        }

# Custom PyTorch dataset class that handles the conversion of tokenized data to tensors
# and prepares the data format required by the GPT-2 model for training
class GPT2Dataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        # Get example from dataset
        logger.debug(f"Accessing dataset item at index: {idx}")
        example = self.encodings[idx]
        
        # Handle different types of encodings
        if isinstance(example, dict):
            logger.debug(f"Example at index {idx} is a dictionary")
            # Check if values are already tensors
            if isinstance(example.get("input_ids"), torch.Tensor):
                logger.debug(f"Example at index {idx} already contains tensor data")
                return {
                    "input_ids": example["input_ids"],
                    "attention_mask": example["attention_mask"],
                    "labels": example["input_ids"].clone()
                }
            else:
                logger.debug(f"Example at index {idx} contains non-tensor data, converting to tensors")
                return {
                    "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(example["input_ids"], dtype=torch.long)
                }
        else:
            logger.warning(f"Unexpected example format at index {idx}: {type(example)}")
            # Handle unexpected format
            raise ValueError(f"Unexpected example format: {type(example)}")

# Monitors validation loss during training to prevent overfitting by stopping training
# when the model performance stops improving and saving the best model checkpoint
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.005, checkpoint_dir='./checkpoints'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def __call__(self, val_loss, model, tokenizer, step):
        if val_loss < self.best_loss - self.min_delta:
            logger.info(f"Validation loss improved from {self.best_loss:.6f} to {val_loss:.6f}")
            logger.info(f"Resetting early stopping counter to 0 (was {self.counter})")
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, tokenizer, step, val_loss)
            return False
        else:
            self.counter += 1
            logger.info(f"Validation loss did not improve. Current: {val_loss:.6f}, Best: {self.best_loss:.6f}")
            logger.info(f"Early stopping counter increased to {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} evaluations without improvement")
                return True
            return False
    
    def save_checkpoint(self, model, tokenizer, step, val_loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint-{step}-loss-{val_loss:.4f}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"Model checkpoint saved at {checkpoint_path}")

# Creates and saves a visualization of training and validation losses over time
# to help monitor model convergence and detect potential overfitting
def plot_training_metrics(train_losses, val_losses, output_dir):
    plt.figure(figsize=(12, 6))
    
    # Plot both lines on the same chart with different colors
    plt.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    plt.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2, marker='s', markersize=4)
    
    # Add grid and styling
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Evaluation Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Yanomami Translation Model Training Progress', fontsize=14, fontweight='bold')
    
    # Add legend with better positioning
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    
    # Add minor ticks for more precise reading
    plt.minorticks_on()
    
    # Ensure y-axis starts from 0 or slightly below the minimum loss for better perspective
    min_loss = min(min(train_losses), min(val_losses)) if train_losses and val_losses else 0
    plt.ylim(max(0, min_loss * 0.9), None)
    
    plt.tight_layout()
    
    # Save the plot with higher DPI for better quality
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Enhanced training metrics plot saved to {plot_path}")

# Generates translations using the fine-tuned model with appropriate prefixes
# to control the direction of translation (English-to-Yanomami or Yanomami-to-English)
def generate_translation(text, model, tokenizer, config, prefix_type="english_to_yanomami"):
    # Ensure text is properly encoded
    try:
        # Normalize Unicode characters
        text = unicodedata.normalize('NFC', text)
    except Exception as e:
        logger.warning(f"Error normalizing input text: {str(e)}")
    
    # Enhanced prompt engineering with clear structure and context
    if prefix_type == "english_to_yanomami":
        prompt = (
            "You are a Yanomami language translator.\n"
            "Translate the following English text to Yanomami accurately:\n\n"
            f"English: {text}\n\n"
            "Yanomami: "
        )
    else:
        logger.info(f"Fallback. Using Yanomami-to-English translation direction for text: '{text[:30]}...'")
        prompt = (
            "You are a Yanomami language translator.\n"
            "Translate the following Yanomami text to English accurately:\n\n"
            f"Yanomami: {text}\n\n"
            "English: "
        )
    
    # Tokenize input with enhanced handling for special characters
    try:
        # Use enhanced tokenization if available
        if hasattr(tokenizer, 'enhanced_encode'):
            input_ids = tokenizer.enhanced_encode(prompt, return_tensors="pt")
            inputs = {"input_ids": input_ids}
        else:
            logger.warning("Enhanced tokenizer not available. Falling back to standard tokenization.")
            inputs = tokenize_text(prompt, tokenizer, config)
    except Exception as e:
        logger.warning(f"Error in enhanced tokenization: {str(e)}. Falling back to standard tokenization.")
        inputs = tokenize_text(prompt, tokenizer, config)
    
    # Determine device without logging details again
    device = get_device(log_info=False)
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    
    # Log input state
    logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
    logger.info(f"Input text tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    
    # Generate translation with simplified parameters for initial testing
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  
            min_length=1,   # Allow any length output
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id
        )
        logger.info(f"Generated output shape: {outputs.shape}")
        logger.info(f"Output tokens: {tokenizer.convert_ids_to_tokens(outputs[0])}")
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return ""  # Return empty string on error
    
    # Decode and return translation with enhanced handling for special characters
    try:
        # Use enhanced decoding if available
        if hasattr(tokenizer, 'enhanced_decode'):
            translation = tokenizer.enhanced_decode(outputs[0], skip_special_tokens=True)
        else:
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.warning(f"Error in enhanced decoding: {str(e)}. Falling back to standard decoding.")
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Log the raw model output for debugging
    logger.debug(f"Raw model output:\n{translation}")
    
    # Clean and extract the relevant part of the translation
    cleaned_translation = clean_translation_output(translation, prefix_type)
    
    return cleaned_translation

# Cleans and extracts the relevant part of the translation output with enhanced handling
def clean_translation_output(translation, prefix_type):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting translation cleaning. Raw output:\n{translation}")
    
    try:
        if not translation or translation.isspace():
            logger.warning("Received empty or whitespace-only translation")
            return ""
            
        # Normalize Unicode characters
        translation = unicodedata.normalize('NFC', translation)
        logger.debug(f"After Unicode normalization:\n{translation}")
        
        # Remove the context-setting prefix more leniently
        prefixes_to_remove = [
            "You are a Yanomami language translator",
            "Translate the following English text to Yanomami",
            "Translate the following Yanomami text to English",
            "accurately",
            "translation"
        ]
        for prefix in prefixes_to_remove:
            translation = re.sub(f"{prefix}.*?:", "", translation, flags=re.IGNORECASE).strip()
        
        logger.debug(f"After removing prefixes:\n{translation}")
        
        # Extract the actual translation based on the target language
        target_marker = "Yanomami:" if prefix_type == "english_to_yanomami" else "English:"
        if target_marker.lower() in translation.lower():
            parts = re.split(target_marker, translation, flags=re.IGNORECASE)
            translation = parts[-1].strip()  # Take the last occurrence
            logger.debug(f"Found target marker '{target_marker}'. Extracted:\n{translation}")
        
        # Clean up any remaining markers and artifacts
        markers = ["English:", "Yanomami:", "Translation:", "<start>", "<end>", "\n\n", ".:", ":", "\t"]
        for marker in markers:
            translation = translation.replace(marker, "").strip()
        
        # Remove multiple consecutive spaces and newlines
        translation = re.sub(r'\s+', ' ', translation).strip()
        logger.debug(f"After cleaning markers:\n{translation}")
        
        # Handle potential repetitions more intelligently
        if not translation:
            logger.warning("Translation became empty after cleaning")
            return ""
            
        # Split by common sentence delimiters
        sentences = re.split(r'[.!?]+', translation)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            logger.warning("No valid sentences found after splitting")
            return translation.strip()
            
        # Remove exact duplicates and near-duplicates
        unique_sentences = []
        for sentence in sentences:
            # Check if this sentence is too similar to any existing one
            is_duplicate = False
            for existing in unique_sentences:
                similarity = difflib.SequenceMatcher(None, sentence.lower(), existing.lower()).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_sentences.append(sentence)
        
        # Combine sentences
        final_translation = ' '.join(unique_sentences)
        logger.info(f"Final cleaned translation:\n{final_translation}")
        
        return final_translation
    except Exception as e:
        logger.error(f"Error in clean_translation_output: {str(e)}")

       # Fallback cleaning in case of error
        try:
            # Keep only non-empty, unique lines
            unique_lines = []
            for line in translation.split('\n'):
                line = line.strip()
                if line and line not in unique_lines:
                    unique_lines.append(line)
            cleaned_translation = ' '.join(unique_lines)
            
            # Ensure proper Unicode normalization
            cleaned_translation = unicodedata.normalize('NFC', cleaned_translation)
            return cleaned_translation
        except Exception as e2:
            logger.warning(f"Error in fallback cleaning: {e2}")
            return translation.strip()  # Return stripped original as last resort

# Calculate complexity of translation examples based on multiple factors
def calculate_complexity(example):
    # 1. Length of text
    input_length = len(example['input'])
    output_length = len(example['output'])
    
    # 2. Count of special characters (like ɨ)
    special_chars = ['ɨ', 'ë', 'ã', 'õ', 'ñ', 'ï']
    special_char_count = sum(example['input'].count(c) + example['output'].count(c) for c in special_chars)
    
    # 3. Word count
    input_word_count = len(example['input'].split())
    output_word_count = len(example['output'].split())
    
    # Calculate overall complexity score
    # We normalize by dividing by 10 to keep scores manageable
    complexity = (input_length + output_length + special_char_count*3 + input_word_count + output_word_count) / 10
    
    return complexity

# Load the model and tokenizer for Yanomami-English translation
def load_yanomami_translator(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory {model_path} does not exist.")
    
    # Load tokenizer and model
    tokenizer = load_enhanced_tokenizer(model_path)
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Configure device for inference
    device = get_device(log_info=False)
    
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    return model, tokenizer

# Main function to run
def main():
    # Initialize variables
    writer = None
    start_time = time.time()
    global_step = 0
    
    # Initialize configuration
    config = TranslatorConfig()
    
    # Check for GH200 GPU when running on Lambda
    if config.is_lambda:
        logger.info("Running on Lambda Cloud with optimized settings")
        
        # Check for GH200 GPU
        if is_gh200_gpu():
            logger.info("Detected GH200 Superchip - applying specialized optimizations")
            
            # Optimize configuration for GH200 on Lambda
            config.use_mixed_precision = True
            logger.info("Enabled mixed precision training for GH200")
            
            # Lambda GH200 instances have high memory - optimize batch size
            if hasattr(config, 'batch_size') and config.batch_size < 32:
                original_batch_size = config.batch_size
                config.batch_size = min(32, config.batch_size * 2)  # Double batch size but cap at 32
                logger.info(f"GH200: Increased batch size from {original_batch_size} to {config.batch_size}")
            
            # Optimize gradient accumulation for high memory
            if hasattr(config, 'gradient_accumulation_steps') and config.gradient_accumulation_steps > 1:
                original_grad_accum = config.gradient_accumulation_steps
                config.gradient_accumulation_steps = 1  # GH200 has enough memory to avoid gradient accumulation
                logger.info(f"GH200: Reduced gradient accumulation steps from {original_grad_accum} to 1")
            
            # Optimize memory allocation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache for optimal memory usage")
            
            # Log GH200-specific GPU info
            logger.info(f"GH200 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directories
    os.makedirs(config.model_output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    logger.info(f"Created output directories: {config.model_output_dir} and {config.checkpoint_dir}")
    
    # Load and prepare data
    logger.info("Loading dataset files...")
    all_data_by_file = {}  # Dictionary to store data by filename
    
    for file in config.dataset_files:
        try:
            file_path = os.path.join(config.dataset_path, file)
            logger.info(f"Attempting to load file: {file_path}")
            file_data = load_jsonl(file_path)
            
            # Store data by filename
            filename = os.path.basename(file_path)
            all_data_by_file[filename] = file_data
            
            logger.info(f"Successfully loaded {len(file_data)} examples from {filename}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    # Calculate total examples
    total_examples = sum(len(data) for data in all_data_by_file.values())
    logger.info(f"Total examples loaded: {total_examples}")
    
    # Process data by file
    all_processed_data = {}
    for filename, file_data in all_data_by_file.items():
        processed_examples = prepare_data_for_training(file_data)
        all_processed_data[filename] = processed_examples
        logger.info(f"Processed {len(processed_examples)} examples from {filename}")
    
    # Calculate total processed examples
    total_processed = sum(len(data) for data in all_processed_data.values())
    logger.info(f"Total processed examples: {total_processed}")
    
    # Implement curriculum learning with file-based or complexity-based dataset selection
    def get_phase_dataset(phase, all_examples):
        phase_examples = []
        
        # Debug: Print all available example keys
        logger.info(f"Available dataset files: {list(all_examples.keys())}")
        
        # If phase specifies complexity thresholds, use complexity-based selection
        if 'complexity_min' in phase and 'complexity_max' in phase:
            logger.info(f"Using complexity-based selection for {phase['name']}")
            # Flatten all examples into a single list
            all_flattened = []
            for examples in all_examples.values():
                all_flattened.extend(examples)
            
            # Filter examples by complexity
            for example in all_flattened:
                if 'complexity' in example and phase['complexity_min'] <= example['complexity'] <= phase['complexity_max']:
                    phase_examples.append(example)
            
            logger.info(f"Selected {len(phase_examples)} examples with complexity between {phase['complexity_min']} and {phase['complexity_max']} for {phase['name']}")
        
        # If phase specifies dataset files, use file-based selection
        elif 'dataset_files' in phase:
            for filename in phase['dataset_files']:
                # Try to find the file by exact name or as part of a path
                found = False
                
                # Check for exact match
                if filename in all_examples:
                    phase_examples.extend(all_examples[filename])
                    logger.info(f"Added {len(all_examples[filename])} examples from {filename} to {phase['name']}")
                    found = True
                else:
                    # Check if the filename is part of any key in all_examples
                    for key in all_examples.keys():
                        if filename in key:
                            phase_examples.extend(all_examples[key])
                            logger.info(f"Added {len(all_examples[key])} examples from {key} (matched {filename}) to {phase['name']}")
                            found = True
                            break
                
                if not found:
                    logger.warning(f"Could not find dataset file {filename} for {phase['name']}")
        
        # If neither complexity nor dataset files are specified, use all data
        else:
            logger.warning(f"No dataset files or complexity thresholds specified for {phase['name']}, using all data")
            for examples in all_examples.values():
                phase_examples.extend(examples)
        
        logger.info(f"Phase {phase['name']}: {len(phase_examples)} examples selected")
        return phase_examples
        
    # Calculate complexity for all examples (for potential use in validation)
    flattened_data = []
    for examples in all_processed_data.values():
        for example in examples:
            example['complexity'] = calculate_complexity(example)
            flattened_data.append(example)
    
    # Create a validation set from all data
    # We'll use 10% of all data for validation
    val_data = train_test_split(flattened_data, test_size=0.1, random_state=42)[1]  # Only keep the validation split
    logger.info(f"Total training examples (all phases): {len(flattened_data) - len(val_data)}")
    logger.info(f"Validation examples: {len(val_data)}")
    
    # First, load the Yanomami-specific tokenizer
    logger.info("Loading tokenizer")
    current_dir = os.getcwd()
    yanomami_tokenizer_path = os.path.join(current_dir, 'yanomami_tokenizer/complete_yanomami_tokenizer')
    
    # Check if we should use NLLB tokenizer or custom Yanomami tokenizer
    if config.model_type == "nllb":
        logger.info(f"Using NLLB tokenizer for {config.model_name}")
        # Load NLLB tokenizer using AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Set source and target language
        tokenizer.src_lang = config.source_lang
        tokenizer.tgt_lang = config.target_lang
        
        logger.info(f"NLLB tokenizer vocabulary size: {len(tokenizer)}")
        logger.info(f"Source language: {config.source_lang}, Target language: {config.target_lang}")
    else:
        # Original code for custom Yanomami tokenizer
        if os.path.exists(yanomami_tokenizer_path):
            logger.info(f"Loading Yanomami-specific tokenizer from {yanomami_tokenizer_path}")
            tokenizer = load_enhanced_tokenizer(yanomami_tokenizer_path)
        else:
            # Halt execution if tokenizer is not available
            error_msg = f"ERROR: Yanomami-specific tokenizer not found at {yanomami_tokenizer_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    # Now load or initialize the model
    logger.info("Loading or initializing model")
    if config.model_type == "nllb":
        logger.info(f"Using NLLB model: {config.model_name}")
        model = M2M100ForConditionalGeneration.from_pretrained(config.model_name)
        
        # For NLLB, we don't need to resize token embeddings as it already has a comprehensive vocabulary
        logger.info(f"NLLB model loaded with {model.get_input_embeddings().num_embeddings} embeddings")
    else:
        # Original code for loading other models
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        
        # Resize token embeddings to match the tokenizer vocabulary size
        logger.info(f"Resizing token embeddings from {model.get_input_embeddings().num_embeddings} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Set padding token if needed
    if config.model_type == "nllb":
        logger.info("NLLB tokenizer already has padding token configured")
    else:
        # For other models
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set padding token to EOS token")
    
    # GH200 Superchip Optimizations
    # Enable mixed precision training to utilize Tensor Cores efficiently
    config.use_mixed_precision = True
    logger.info(f"Mixed precision training enabled: {config.use_mixed_precision}")
    
    # Check for multi-GPU setup
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for distributed training")
        model = torch.nn.DataParallel(model)
    
    # Configure device
    device = get_device(log_info=True)
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # GH200-specific memory optimizations
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 40e9:  # Check if we have a high-memory GPU (>40GB)
        logger.info("Detected high-memory GPU, optimizing memory usage for GH200")
        # Increase batch size if we have more memory available
        if hasattr(config, 'batch_size') and config.batch_size < 16:
            original_batch_size = config.batch_size
            config.batch_size = min(16, config.batch_size * 2)  # Double batch size but cap at 16
            logger.info(f"Increased batch size from {original_batch_size} to {config.batch_size} for high-memory GPU")
        
        # Reduce gradient accumulation steps if we have a larger batch size
        if hasattr(config, 'gradient_accumulation_steps') and config.gradient_accumulation_steps > 4:
            original_grad_accum = config.gradient_accumulation_steps
            config.gradient_accumulation_steps = max(4, config.gradient_accumulation_steps // 2)
            logger.info(f"Reduced gradient accumulation steps from {original_grad_accum} to {config.gradient_accumulation_steps}")
    
    # Store original processed data for phases
    all_train_data = all_processed_data
    
    # Prepare validation dataset once (used across all phases)
    logger.info("Preparing validation dataset...")

    # Create a new dataset with the required columns for validation
    tokenized_val_data = prepare_tokenized_dataset(val_data, tokenizer, config)

    # Create a new dataset from the tokenized validation data
    if tokenized_val_data:
        tokenized_val = Dataset.from_list(tokenized_val_data)
        logger.info(f"Created validation dataset with {len(tokenized_val_data)} tokenized examples")
    else:
        # Create a dummy dataset if no examples were processed successfully
        logger.warning("No validation examples were tokenized successfully. Creating dummy dataset.")
        tokenized_val = Dataset.from_dict({
            "input_ids": [torch.tensor([0, 0])],
            "attention_mask": [torch.tensor([1, 1])]
        })
    
    # Create the validation dataset
    val_dataset = GPT2Dataset(tokenized_val)
    eval_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size
    )
    
    # Initialize visualization utilities if enabled
    visualizer = None
    if config.enable_visualizations:
        logger.info("Initializing visualization utilities...")
        visualizer = TrainingVisualizer(output_dir=config.visualization_output_dir)
        
        # Set baseline precision values for comparison
        for model_name, precision in config.baseline_precision.items():
            visualizer.set_baseline_precision(model_name, precision)
            logger.info(f"Set baseline precision for {model_name}: {precision}")
    
    # Multi-phase training
    try:
        if config.enable_multi_phase:
            logger.info("Starting multi-phase training...")
            overall_start_time = time.time()
        
            for phase_idx, phase in enumerate(config.phases):
                logger.info(f"\n{'='*50}\n{phase['name']}\n{'='*50}")
                
                # Get data for this phase based on dataset files
                phase_train_data = get_phase_dataset(phase, all_train_data)
                logger.info(f"Phase {phase_idx+1} ({phase['name']}) training on {len(phase_train_data)} examples")
                
                # Tokenize datasets for this phase
                logger.info(f"Tokenizing phase {phase_idx+1} dataset...")
                
                # Create a new dataset with the required columns
                tokenized_data = prepare_tokenized_dataset(phase_train_data, tokenizer, config)
                
                # Create a new dataset from the tokenized data
                if tokenized_data:
                    tokenized_train = Dataset.from_list(tokenized_data)
                    logger.info(f"Created dataset with {len(tokenized_data)} tokenized examples")
                else:
                    # Create a dummy dataset if no examples were processed successfully
                    logger.warning("No examples were tokenized successfully. Creating dummy dataset.")
                    tokenized_train = Dataset.from_dict({
                        "input_ids": [torch.tensor([0, 0])],
                        "attention_mask": [torch.tensor([1, 1])]
                    })
                
                # Create PyTorch datasets
                train_dataset = GPT2Dataset(tokenized_train)
                
                # Create data loader for this phase
                train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=config.batch_size, 
                    shuffle=True
                )
                
                # Calculate training steps for this phase
                total_steps = max(1, len(train_dataloader) * phase['num_epochs'] // config.gradient_accumulation_steps)
                warmup_steps = max(1, int(total_steps * config.warmup_ratio))
                
                # Configure optimizer with phase-specific learning rate
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": config.weight_decay,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
                
                optimizer = AdamW(
                    optimizer_grouped_parameters, 
                    lr=phase['learning_rate'], 
                    eps=config.adam_epsilon
                )
                logger.info(f"Configured optimizer with learning rate: {phase['learning_rate']} for {total_steps} total training steps.")
                
                # Configure scheduler
                # Ensure optimizer has non-zero learning rate before creating scheduler
                for param_group in optimizer.param_groups:
                    if param_group['lr'] == 0:
                        logger.warning(f"Initial learning rate was 0, setting to {phase['learning_rate']}")
                        param_group['lr'] = phase['learning_rate']
                
                if config.scheduler_type == 'linear':
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer, 
                        num_warmup_steps=warmup_steps, 
                        num_training_steps=total_steps
                    )
                else:  # cosine
                    scheduler = get_cosine_schedule_with_warmup(
                        optimizer, 
                        num_warmup_steps=warmup_steps, 
                        num_training_steps=total_steps
                    )
                
                # Log initial learning rate for this phase
                initial_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Initial learning rate for phase {phase_idx+1} ({phase['name']}): {initial_lr:.8f}")
                if initial_lr == 0:
                    logger.warning(f"WARNING: Initial learning rate is 0! This will prevent model training. Setting to {phase['learning_rate']}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = phase['learning_rate']
                    logger.info(f"Updated initial learning rate to: {optimizer.param_groups[0]['lr']:.8f}")
                
                # Initialize early stopping for this phase
                phase_checkpoint_dir = os.path.join(config.checkpoint_dir, f"phase_{phase_idx+1}")
                os.makedirs(phase_checkpoint_dir, exist_ok=True)
                
                early_stopping = EarlyStopping(
                    patience=config.patience, 
                    min_delta=config.min_delta,
                    checkpoint_dir=phase_checkpoint_dir
                )
                
                # Initialize mixed precision scaler
                scaler = GradScaler() if config.use_mixed_precision and torch.cuda.is_available() else None
                
                # Training metrics for this phase
                train_losses = []
                val_losses = []
                global_step = 0
                
                # Training loop for this phase
                logger.info(f"Starting training for phase {phase_idx+1}...")
                start_time = time.time()
                
                try:
                    for epoch in range(phase['num_epochs']):
                        logger.info(f"Starting epoch {epoch + 1}/{phase['num_epochs']} of phase {phase_idx + 1}")
                        
                        # Log training mode at the start of the epoch
                        if config.use_mixed_precision and torch.cuda.is_available():
                            logger.info("Using mixed precision training.")
                        else:
                            logger.info("Using standard training.")
                            
                        # Log GPU memory usage at the start of each epoch (useful for GH200)
                        if torch.cuda.is_available():
                            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                        
                        epoch_loss = 0
                        model.train()
                
                        # Track batch progress
                        total_batches = len(train_dataloader)
                        
                        # Training
                        for batch_idx, batch in enumerate(train_dataloader):
                            # Move batch to device
                            batch = {k: v.to(device) for k, v in batch.items()}
                            
                            # Log training examples in this batch (first example only to avoid excessive logging)
                            if batch_idx % config.logging_steps == 0:
                                try:
                                    # Decode the first example in the batch with improved Unicode handling
                                    input_ids = batch['input_ids'][0].cpu().numpy()
                                    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                                    
                                    # Normalize Unicode for consistent representation
                                    import unicodedata
                                    normalized_text = unicodedata.normalize('NFC', decoded_text)
                                    
                                    # Log both the raw and normalized text for comparison
                                    logger.info(f"Batch {batch_idx} training example (normalized): {normalized_text}")
                                    
                                    # Log character codes for debugging encoding issues
                                    char_codes = ', '.join([f"{c}({ord(c):04x})" for c in normalized_text[:20]])
                                    logger.info(f"Character codes (first 20): {char_codes}")
                                    
                                    # Find and log special characters that might cause issues
                                    special_chars = []
                                    for i, c in enumerate(normalized_text):
                                        # Check for non-ASCII characters or characters that might be problematic
                                        if ord(c) > 127 or c in '<>[](){}\"\'':
                                            context = normalized_text[max(0, i-10):min(len(normalized_text), i+10)]
                                            special_chars.append(f"{c}({ord(c):04x}) at position {i} in context: ...{context}...")
                                    
                                    if special_chars:
                                        logger.info(f"Special characters found: {len(special_chars)}")
                                        for i, char_info in enumerate(special_chars):  # Limit to first 5 to avoid excessive logging
                                            logger.info(f"Special char {i+1}: {char_info}")
                                except Exception as e:
                                    logger.warning(f"Could not decode training example: {str(e)}")

                
                            # Mixed precision training
                            if config.use_mixed_precision and torch.cuda.is_available():
                                with autocast(device_type='cuda'):
                                    outputs = model(**batch)
                                    loss = outputs.loss / config.gradient_accumulation_steps
                            
                                # Scale loss and compute gradients
                                scaler.scale(loss).backward()
                            
                                # Accumulate gradients
                                if (batch_idx + 1) % config.gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                                    # Clip gradients
                                    scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                                
                                    # Update weights
                                    scaler.step(optimizer)
                                    scaler.update()
                                    
                                    # Debug learning rate before and after scheduler step
                                    lr_before = optimizer.param_groups[0]['lr']
                                    scheduler.step()
                                    lr_after = optimizer.param_groups[0]['lr']
                                    
                                    # Log if learning rate changed
                                    if abs(lr_before - lr_after) > 1e-8:
                                        logger.info(f"Learning rate change (mixed precision): {lr_before:.8f} -> {lr_after:.8f} (diff: {lr_after - lr_before:.8f})")
                                    
                                    optimizer.zero_grad()
                                    global_step += 1
                            else:
                                # Standard training
                                outputs = model(**batch)
                                loss = outputs.loss / config.gradient_accumulation_steps
                                loss.backward()
                            
                                # Accumulate gradients
                                if (batch_idx + 1) % config.gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                                    # Clip gradients
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                                    
                                    # Log gradient information for a few parameters (every 10 batches)
                                    if batch_idx % 10 == 0:
                                        grad_norms = {}
                                        for name, param in model.named_parameters():
                                            if param.grad is not None:
                                                # Only log a subset of parameters to avoid excessive output
                                                if 'layer.0' in name or 'layer.11' in name or 'layer.23' in name:
                                                    grad_norms[name] = torch.norm(param.grad).item()
                                        
                                        # Log the top 5 largest gradients
                                        sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
                                        logger.info(f"Top 5 gradient norms at batch {batch_idx}:")
                                        for param_name, norm in sorted_grads:
                                            logger.info(f"  {param_name}: {norm:.6f}")
                                
                                    # Update weights
                                    optimizer.step()
                                    
                                    # Debug learning rate before and after scheduler step
                                    lr_before = optimizer.param_groups[0]['lr']
                                    scheduler.step()
                                    lr_after = optimizer.param_groups[0]['lr']
                                    
                                    # Log if learning rate changed
                                    if abs(lr_before - lr_after) > 1e-8:
                                        logger.info(f"Learning rate change: {lr_before:.8f} -> {lr_after:.8f} (diff: {lr_after - lr_before:.8f})")
                                    
                                    optimizer.zero_grad()
                                    global_step += 1
                
                                # Accumulate loss
                                epoch_loss += loss.item() * config.gradient_accumulation_steps
                            
                            # Calculate progress metrics
                            progress_percent = (batch_idx + 1) / total_batches * 100
                            remaining_batches = total_batches - batch_idx - 1
                            elapsed_time = time.time() - start_time
                            
                            # Initialize time_per_batch outside the conditional block
                            time_per_batch = 0
                                
                            # Calculate estimated time remaining
                            if batch_idx > 0:
                                time_per_batch = elapsed_time / (batch_idx + 1)
                                estimated_time_remaining = time_per_batch * remaining_batches
                                eta_str = f", ETA: {estimated_time_remaining:.2f}s"
                            else:
                                eta_str = ""

                            # Get current learning rate
                            current_lr = optimizer.param_groups[0]['lr']

                            # Get memory usage if CUDA is available
                            mem_str = ""
                            if torch.cuda.is_available():
                                mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                                mem_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
                                mem_str = f", GPU: {mem_allocated:.0f}MB/{mem_reserved:.0f}MB"
                            
                            # Calculate progress percentage
                            progress_percent = 100.0 * (batch_idx + 1) / total_batches

                            # Log detailed batch information (every batch)
                            logger.info(
                                f"Phase {phase_idx+1}, Epoch {epoch+1}/{phase['num_epochs']}, "
                                f"Batch {batch_idx+1}/{total_batches} ({progress_percent:.1f}%) | "
                                f"Loss: {loss.item()*config.gradient_accumulation_steps:.4f} | "
                                f"LR: {current_lr:.2e} | "
                                f"Time/batch: {time_per_batch:.2f}s{eta_str}{mem_str}"
                            )

                            if config.enable_visualizations and visualizer:
                                batch_loss = loss.item() * config.gradient_accumulation_steps
                                visualizer.record_training_metrics(
                                    phase=phase_idx+1,
                                    epoch=epoch+1,
                                    batch=batch_idx+1,
                                    loss=batch_loss,
                                    learning_rate=current_lr
                                )
                            
                            # Generate plots at specified batch intervals if configured
                            if config.plot_batch_interval > 0 and batch_idx % config.plot_batch_interval == 0 and batch_idx > 0:
                                batch_suffix = f" (Phase {phase_idx+1}, Epoch {epoch+1}, Batch {batch_idx+1})"
                                visualizer.plot_loss_and_lr(title_suffix=batch_suffix)
                                
                                # Run translation tests at batch intervals
                                # Only do this occasionally to avoid slowing down training too much
                                if batch_idx % (config.plot_batch_interval * 2) == 0:
                                    logger.info(f"Running quick translation test at batch {batch_idx+1}...")
                                    test_translations(
                                        model=model,
                                        tokenizer=tokenizer,
                                        config=config,
                                        phase=phase_idx+1,
                                        epoch=epoch+1,
                                        batch=batch_idx+1,
                                        save_results=True
                                    )
                                    
                            # Log sample tokens from batch (first example only) if in debug mode
                            if config.debug_mode and batch_idx % 50 == 0:
                                try:
                                    # Get the first example from the batch
                                    input_ids = batch['input_ids'][0].cpu().tolist()
                                    input_text = tokenizer.decode(input_ids)
                                    logger.info(f"Sample input: {input_text[:100]}...")
                                except Exception as e:
                                    logger.debug(f"Could not decode sample input: {e}")
                    
                        # Calculate and log average loss for the epoch
                        avg_epoch_loss = epoch_loss / total_batches
                        logger.info(f"\n{'-'*20} Epoch {epoch+1}/{phase['num_epochs']} of phase {phase_idx+1} ({phase['name']}) completed {'-'*20}")
                        logger.info(f"Average training loss: {avg_epoch_loss:.4f}")
                        
                        # Log to tensorboard if available
                        if writer:
                            writer.add_scalar(f"Loss/train/phase_{phase_idx+1}", avg_epoch_loss, epoch)
                        
                        # Evaluate on validation set at the end of each epoch
                        logger.info(f"Evaluating on validation set at the end of epoch {epoch+1}...")
                        epoch_eval_loss = evaluate_model(model, eval_dataloader, device, tokenizer, config)
                        logger.info(f"Validation loss: {epoch_eval_loss:.4f}")
                        logger.info(f"{'-'*80}\n")
                        
                        # Test translations at the end of each epoch
                        if config.test_translations_every_epoch:
                            logger.info(f"Running translation tests at the end of epoch {epoch+1}...")
                            test_translations(
                                model=model,
                                tokenizer=tokenizer,
                                config=config,
                                phase=phase_idx+1,
                                epoch=epoch+1,
                                save_results=True
                            )
                        
                        # Evaluate and save checkpoint
                        if global_step % config.eval_steps == 0:
                            # Evaluate with enhanced logging
                            logger.info(f"\n{'-'*20} Evaluation at Step {global_step} {'-'*20}")
                            eval_loss = evaluate_model(model, eval_dataloader, device, tokenizer, config)
                            val_losses.append(eval_loss)
                            train_losses.append(avg_epoch_loss)
                            
                            # Calculate and log training/validation loss difference
                            train_loss = avg_epoch_loss
                            loss_diff = train_loss - eval_loss
                            logger.info(f"Step {global_step}: Train Loss: {train_loss:.4f}, Val Loss: {eval_loss:.4f}, Diff: {loss_diff:.4f}")
                            logger.info(f"{'-'*65}\n")
                            
                            # Record evaluation metrics for visualization
                            if config.enable_visualizations and visualizer:
                                # Calculate precision (using 1 - eval_loss as a proxy for precision)
                                # This is a simplified metric - in a real scenario, you might want to use
                                # a more sophisticated precision calculation
                                precision_proxy = max(0, min(1, 1 - eval_loss))
                                
                                visualizer.record_training_metrics(
                                    phase=phase_idx+1,
                                    epoch=epoch+1,
                                    loss=train_loss,
                                    precision=precision_proxy
                                )
                            
                            # Check early stopping
                            if early_stopping(eval_loss, model, tokenizer, global_step):
                                logger.info("Early stopping triggered")
                                break
                        
                        # Save regular checkpoint
                        if global_step % config.save_steps == 0:
                            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint-{global_step}")
                            os.makedirs(checkpoint_path, exist_ok=True)
                            model.save_pretrained(checkpoint_path)
                            tokenizer.save_pretrained(checkpoint_path)
                            logger.info(f"Regular checkpoint saved at {checkpoint_path}")
            
                    # Calculate and log detailed epoch statistics
                    avg_epoch_loss = epoch_loss / len(train_dataloader)
                    epoch_time = time.time() - start_time
                    examples_per_second = len(train_dataset) / epoch_time
                    
                    # Generate plots at specified epochs if configured
                    if config.enable_visualizations and visualizer and epoch+1 in config.plot_at_epochs:
                        epoch_suffix = f" (Phase {phase_idx+1}, Epoch {epoch+1})"
                        logger.info(f"Generating plots at the end of epoch {epoch+1}...")
                        visualizer.plot_loss_and_lr(title_suffix=epoch_suffix)
                        visualizer.plot_precision_comparison(title_suffix=epoch_suffix)
                        
                        # Run translation tests at the same points as plot generation
                        logger.info(f"Running translation tests at the end of epoch {epoch+1}...")
                        test_translations(
                            model=model,
                            tokenizer=tokenizer,
                            config=config,
                            phase=phase_idx+1,
                            epoch=epoch+1,
                            save_results=True
                        )
                    
                    logger.info(f"{'='*30} EPOCH SUMMARY {'='*30}")
                    logger.info(f"Epoch {epoch+1}/{phase['num_epochs']} of phase {phase_idx+1} ({phase['name']}) completed")
                    logger.info(f"Average loss: {avg_epoch_loss:.4f}")
                    logger.info(f"Processing speed: {examples_per_second:.2f} examples/second")
                    logger.info(f"Epoch duration: {epoch_time:.2f}s")
                    logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
                    logger.info(f"{'='*75}")
                    
                    # Evaluate at end of epoch with enhanced logging
                    logger.info(f"\n{'-'*20} End of Epoch {epoch+1} Evaluation {'-'*20}")
                    eval_loss = evaluate_model(model, eval_dataloader, device, tokenizer, config)
                    val_losses.append(eval_loss)
                    train_losses.append(avg_epoch_loss)
                    
                    # Calculate and log training/validation loss difference
                    loss_diff = avg_epoch_loss - eval_loss
                    logger.info(f"End of epoch {epoch+1}/{phase['num_epochs']} of phase {phase_idx+1}:")
                    logger.info(f"Train Loss: {avg_epoch_loss:.4f}, Val Loss: {eval_loss:.4f}, Diff: {loss_diff:.4f}")
                    logger.info(f"{'-'*65}\n")
                    
                    # Check early stopping
                    if early_stopping(eval_loss, model, tokenizer, global_step):
                        logger.info("Early stopping triggered")
                        break
                    
                    # Run HellaSwag evaluation if configured
                    if config.enable_hellaswag_eval and (epoch+1) in config.hellaswag_eval_epochs:
                        logger.info(f"\n{'*'*30} Running HellaSwag Evaluation {'*'*30}")
                        run_hellaswag_evaluation(
                            model=model,
                            tokenizer=tokenizer,
                            config=config,
                            epoch=epoch+1,
                            phase=phase_idx+1,
                            phase_name=phase['name']
                        )
                        logger.info(f"{'*'*80}\n")
                        
                        # Run translation tests at the same points as HellaSwag evaluation
                        logger.info(f"Running translation tests after HellaSwag evaluation...")
                        test_translations(
                            model=model,
                            tokenizer=tokenizer,
                            config=config,
                            phase=phase_idx+1,
                            epoch=epoch+1,
                            save_results=True
                        )
        
                except Exception as e:
                    logger.error(f"Error during phase {phase_idx+1} training: {e}")
                    raise
                    
                # Phase completed - log comprehensive phase summary
                phase_time = time.time() - start_time
                logger.info(f"\n{'#'*30} PHASE {phase_idx+1} SUMMARY {'#'*30}")
                logger.info(f"Phase: {phase['name']}")
                logger.info(f"Training examples: {len(phase_train_data)}")
                logger.info(f"Learning rate: {phase['learning_rate']}")
                logger.info(f"Epochs completed: {epoch+1}")
                logger.info(f"Total phase duration: {phase_time:.2f}s")
                logger.info(f"Average examples/second: {len(phase_train_data)*epoch/(phase_time):.2f}")
                
                # Log validation metrics
                if val_losses:
                    logger.info(f"Initial validation loss: {val_losses[0]:.4f}")
                    logger.info(f"Final validation loss: {val_losses[-1]:.4f}")
                    logger.info(f"Validation loss improvement: {val_losses[0] - val_losses[-1]:.4f} ({(val_losses[0] - val_losses[-1])/val_losses[0]*100:.2f}%)")
                
                logger.info(f"{'#'*80}\n")
                logger.info(f"Phase {phase_idx+1} training completed in {phase_time:.2f} seconds")
                
                # Save phase checkpoint
                phase_checkpoint_path = os.path.join(config.checkpoint_dir, f"phase-{phase_idx+1}-final")
                os.makedirs(phase_checkpoint_path, exist_ok=True)
                model.save_pretrained(phase_checkpoint_path)
                tokenizer.save_pretrained(phase_checkpoint_path)
                logger.info(f"Phase {phase_idx+1} model saved to {phase_checkpoint_path}")
            
        # All phases completed
        total_time = time.time() - overall_start_time
        logger.info(f"All training phases completed in {total_time:.2f} seconds")
        
        # Generate final visualization plots if enabled
        if config.enable_visualizations and visualizer and config.plot_at_end:
            logger.info("Generating final visualization plots...")
            
            # Generate comprehensive plots with all training data
            final_loss_chart = visualizer.plot_loss_and_lr(title_suffix=" (Final)")
            final_precision_chart = visualizer.plot_precision_comparison(title_suffix=" (Final)")
            
            # Save all visualization data to JSON for future reference
            loss_history_file, precision_history_file = visualizer.save_history_data()
            
            logger.info(f"Final loss and learning rate chart saved to {final_loss_chart}")
            logger.info(f"Final precision comparison chart saved to {final_precision_chart}")
            logger.info(f"Training history data saved to {loss_history_file} and {precision_history_file}")
            
            # Run final translation tests and save results
            logger.info("Running final translation tests...")
            final_translations = test_translations(
                model=model,
                tokenizer=tokenizer,
                config=config,
                save_results=True
            )
            
            # Generate comprehensive training report with all metrics and visualizations
            logger.info("Generating comprehensive training report...")
            
            # Get final metrics from the last evaluation
            final_metrics = {
                "Final Training Loss": train_losses[-1] if train_losses else None,
                "Final Validation Loss": val_losses[-1] if val_losses else None,
                "Training Time (seconds)": total_time,
                "Total Phases": len(config.phases) if config.enable_multi_phase else 1
            }
            
            # Generate the report with translation results
            report_file = visualizer.generate_training_report(
                final_metrics=final_metrics,
                translation_results=final_translations
            )
            logger.info(f"Comprehensive training report generated at {report_file}")
        
        # Plot legacy training metrics
        plot_training_metrics(train_losses, val_losses, config.model_output_dir)
        
        # Save the trained model and tokenizer
        output_dir = 'yanomami_translator_model'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logging.info(f'Model and tokenizer saved to {output_dir}')
        
        # Save training configuration
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(config.__dict__, f, indent=4)
        logging.info('Training configuration saved')
        
        # Save final model
        model.save_pretrained(config.model_output_dir)
        tokenizer.save_pretrained(config.model_output_dir)
        logger.info(f"Final model saved to {config.model_output_dir}")
        
        # Test translations
        test_translations(model, tokenizer, config)
        
        # Run final HellaSwag evaluation
        if config.enable_hellaswag_eval:
            logger.info(f"\n{'*'*30} Running Final HellaSwag Evaluation {'*'*30}")
            run_hellaswag_evaluation(
                model=model,
                tokenizer=tokenizer,
                config=config
            )
            logger.info(f"{'*'*80}\n")
    except Exception as e:
        logger.error(f"Error during overall training process: {e}")
        raise

# Evaluate model
def evaluate_model(model, eval_dataloader, device, tokenizer=None, config=None):
    model.eval()
    total_eval_loss = 0
    eval_steps = 0
    start_time = time.time()
    
    # Initialize metrics collection
    losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            # Collect metrics
            losses.append(loss.item())
            total_eval_loss += loss.item()
            eval_steps += 1
            
            # Log sample evaluation if debug mode and tokenizer provided
            if config and config.debug_mode and tokenizer and batch_idx == 0:
                try:
                    # Get a sample from the batch
                    input_ids = batch['input_ids'][0].cpu().tolist()
                    input_text = tokenizer.decode(input_ids)
                    logger.info(f"Eval sample input: {input_text[:100]}...")
                except Exception as e:
                    logger.debug(f"Could not decode eval sample: {e}")
    
    # Calculate metrics
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    eval_time = time.time() - start_time
    examples_per_second = len(eval_dataloader) * eval_dataloader.batch_size / eval_time if hasattr(eval_dataloader, 'batch_size') else 0
    
    # Log detailed evaluation metrics
    logger.info(f"Evaluation completed in {eval_time:.2f}s")
    logger.info(f"Average evaluation loss: {avg_eval_loss:.4f}")
    logger.info(f"Evaluation speed: {examples_per_second:.2f} examples/second")
    
    if losses:
        # Calculate loss statistics
        min_loss = min(losses)
        max_loss = max(losses)
        std_loss = np.std(losses) if len(losses) > 1 else 0
        logger.info(f"Loss range: {min_loss:.4f} - {max_loss:.4f}, StdDev: {std_loss:.4f}")
    
    model.train()  # Set model back to training mode
    return avg_eval_loss

def run_hellaswag_evaluation(model, tokenizer, config, epoch=None, phase=None, phase_name=None):
    # Path to the hellaswag_evaluation module in the new location
    hellaswag_path = os.path.join(os.path.dirname(__file__), "evaluation")
    
    # Add the hellaswag_path to sys.path if it's not already there
    import sys
    if hellaswag_path not in sys.path:
        sys.path.append(hellaswag_path)
    
    # Check if hellaswag_evaluation module is available in the new location
    hellaswag_file = os.path.join(hellaswag_path, "hellaswag_evaluation.py")
    if not os.path.exists(hellaswag_file):
        logger.warning(f"HellaSwag evaluation module not found at {hellaswag_file}. Skipping evaluation.")
        return None
    
    try:
        # Import the HellaSwag evaluator from the new location
        sys.path.insert(0, hellaswag_path)  # Prioritize the new location
        from hellaswag_evaluation import HellaSwagEvaluator
        
        # Create a unique model identifier
        model_identifier = f"yanomami_phase{phase}_epoch{epoch}" if epoch and phase else "yanomami"
        
        # Log evaluation start
        phase_info = f"epoch {epoch} of phase {phase} ({phase_name})" if epoch and phase and phase_name else "current state"
        logger.info(f"\n{'='*40}\nStarting HellaSwag evaluation for model at {phase_info}\n{'='*40}")
        
        # Create a temporary directory to save the current model state
        temp_model_dir = os.path.join("./temp_models", model_identifier)
        os.makedirs(temp_model_dir, exist_ok=True)
        
        # Save the current model state
        logger.info(f"Saving current model state to {temp_model_dir}")
        model.save_pretrained(temp_model_dir)
        tokenizer.save_pretrained(temp_model_dir)
        
        # Initialize the evaluator
        evaluator = HellaSwagEvaluator(
            model_path=temp_model_dir,
            model_type="yanomami",
            device=str(next(model.parameters()).device)
        )
        
        # Run evaluation
        logger.info(f"Evaluating on {config.hellaswag_num_examples} HellaSwag examples")
        metrics = evaluator.evaluate(num_examples=config.hellaswag_num_examples)
        
        # Create results directory
        results_dir = os.path.join(config.model_output_dir, "hellaswag_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Compare with baselines if configured and generate comparison chart
        comparison = None
        if config.hellaswag_compare_baselines:
            logger.info("Comparing with baseline models...")
            comparison = evaluator.compare_with_baselines()
            
            # Generate and save comparison chart
            chart_path = evaluator.plot_comparison_chart(
                comparison=comparison,
                output_dir=results_dir,
                phase=phase,
                epoch=epoch
            )
            logger.info(f"Comparison chart saved to: {chart_path}")
        
        # Save results with phase and epoch information
        results_file = evaluator.save_results(
            output_dir=results_dir,
            phase=phase,
            epoch=epoch
        )
        
        # Log summary with visual separator
        logger.info(f"\n{'*'*40}\nHellaSwag evaluation completed for {model_identifier}\n"
                   f"Accuracy: {metrics['accuracy']:.4f}, Perplexity: {metrics['perplexity']:.4f}\n{'*'*40}")
        
        # Clean up temporary model directory if not in debug mode
        if not config.debug_mode and os.path.exists(temp_model_dir):
            try:
                shutil.rmtree(temp_model_dir)
                logger.debug(f"Removed temporary model directory: {temp_model_dir}")
            except Exception as e:
                logger.debug(f"Failed to remove temporary model directory: {e}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during HellaSwag evaluation: {e}")
        return None


def test_translations(model, tokenizer, config, phase=None, epoch=None, batch=None, save_results=False):
    logger = logging.getLogger(__name__)
    
    # Create a descriptive header
    test_header = "\n" + "="*50 + "\nTESTING TRANSLATIONS\n" + "="*50
    if phase is not None:
        test_header += f"\nPhase: {phase}, Epoch: {epoch}"
    logger.info(test_header)
    
    # Test cases for both directions
    test_cases = [
        {"text": "Hello", "direction": "english_to_yanomami"},
        {"text": "How are you?", "direction": "english_to_yanomami"},
        {"text": "Hapai kua?", "direction": "yanomami_to_english"}
    ]
    
    logger.info("\n" + "="*50)
    logger.info("STARTING TRANSLATION TESTS")
    logger.info("="*50)
    
    # Log model and tokenizer state
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    all_results = {"translations": []}
    
    try:
        for test_case in test_cases:
            test_phrase = test_case["text"]
            direction = test_case["direction"]
            
            logger.info(f"\n{'-'*30}\nTesting: {test_phrase} ({direction})\n{'-'*30}")
            
            # Get device
            device = next(model.parameters()).device
            
            # Handle differently based on model type
            if config.model_type == "nllb":
                # For NLLB models, we use language codes instead of prompts
                if direction == "english_to_yanomami":
                    # Set source language to English and target to proxy language
                    src_lang = config.source_lang  # English
                    tgt_lang = config.target_lang  # Proxy for Yanomami
                    
                    # Log what we're doing
                    logger.info(f"Using NLLB model for English-to-Yanomami translation")
                    logger.info(f"Source language: {src_lang}, Target language: {tgt_lang}")
                    
                    # No prompt needed, just the text
                    prompt = test_phrase
                else:  # yanomami_to_english
                    # Swap source and target languages
                    src_lang = config.target_lang  # Proxy for Yanomami
                    tgt_lang = config.source_lang  # English
                    
                    logger.info(f"Using NLLB model for Yanomami-to-English translation")
                    logger.info(f"Source language: {src_lang}, Target language: {tgt_lang}")
                    
                    # No prompt needed, just the text
                    prompt = test_phrase
                
                # Tokenize for NLLB
                logger.info(f"Tokenizing with source language: {src_lang}")
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, src_lang=src_lang)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate translation with NLLB
                model.eval()
                with torch.no_grad():
                    # Check if the tokenizer has the necessary attributes
                    if hasattr(tokenizer, 'lang_code_to_id'):
                        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
                        logger.info(f"Using forced_bos_token_id: {forced_bos_token_id} for language {tgt_lang}")
                    else:
                        # Some models use a different approach
                        forced_bos_token_id = None
                        logger.info("No lang_code_to_id found in tokenizer, not using forced_bos_token_id")
                    
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_new_tokens=300,
                        min_length=1,
                        num_return_sequences=1,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
            else:
                # Original GPT-2 approach with instruction format
                if direction == "english_to_yanomami":
                    prompt = (
                        "You are a Yanomami language translator.\n"
                        "Translate the following English text to Yanomami accurately:\n\n"
                        f"English: {test_phrase}\n\n"
                        "Yanomami: "
                    )
                else:  # yanomami_to_english
                    logger.info(f"Using Yanomami-to-English translation direction for text: '{test_phrase[:30]}...'")
                    prompt = (
                        "You are a Yanomami language translator.\n"
                        "Translate the following Yanomami text to English accurately:\n\n"
                        f"Yanomami: {test_phrase}\n\n"
                        "English: "
                    )
                
                # Log the prompt
                logger.info(f"\nUsing prompt:\n{prompt}")
                
                # Tokenize and log input
                inputs = tokenize_text(prompt, tokenizer, config)
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate with simplified parameters
                model.eval()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        min_length=1,
                        num_return_sequences=1,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id
                    )
            
            # Decode and log output based on model type
            if config.model_type == "nllb":
                # For NLLB, we need to specify the target language when decoding
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Log raw input and output
                logger.info(f"\nRAW INPUT (with special tokens):\n{tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}")
                logger.info(f"\nRAW MODEL OUTPUT (with special tokens):\n{decoded_output}")
                
                # Clean output
                cleaned_output = decoded_output.strip()
                logger.info(f"\nCleaned output:\n{cleaned_output}")
            else:
                # Original decoding for other models
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Log raw input and output
                logger.info(f"\nRAW INPUT (with special tokens):\n{tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}")
                logger.info(f"\nRAW MODEL OUTPUT (with special tokens):\n{decoded_output}")
                
                # Clean output
                cleaned_output = decoded_output.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "").strip()
                logger.info(f"\nCleaned output:\n{cleaned_output}")
            
            # Store result for this test case
            result = {
                "input": test_phrase,
                "direction": direction,
                "output": cleaned_output
            }
            
            # Add to all results
            all_results["translations"].append(result)
        
        # Add metadata to results
        all_results["timestamp"] = datetime.now().isoformat()
        all_results["phase"] = phase
        all_results["epoch"] = epoch
        all_results["model_device"] = str(device)
        all_results["tokenizer_size"] = len(tokenizer)
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"translation_results_{timestamp}.json"
            output_path = os.path.join(config.visualization_output_dir, filename)
            os.makedirs(config.visualization_output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"\nSaved results to: {output_path}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"\nError in test_translations: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return {"translations": [], "error": str(e)}