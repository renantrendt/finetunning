# Enhanced GPT-2 Fine-tuning for Yanomami-English Translation
# 
# This script implements an improved version of the Yanomami-English translator
# with focus on better performance, training efficiency, and offline functionality.

import os
import json
import torch
import numpy as np
import time
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode='w')  # Use mode='w' to overwrite the file
    ]
)

# Force output to be displayed in the terminal
print("Starting Yanomami-English translation model fine-tuning...")

# Create a console handler with a higher log level
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Add the handler to the logger
logger = logging.getLogger(__name__)
logger.addHandler(console)

# Configuration class for easy parameter management
class TranslatorConfig:
    def __init__(self):
        # Debug mode for testing
        self.debug_mode = True  # Set to True for verbose output
        
        # Data settings
        self.dataset_path = '/Users/renanserrano/CascadeProjects/Yanomami/finetunning/yanomami_dataset/'
        self.dataset_files = glob.glob(os.path.join(self.dataset_path, '*.jsonl'))
        
        # Model settings
        self.model_name = "gpt2"
        self.model_output_dir = "./enhanced_yanomami_translator"
        self.checkpoint_dir = "./checkpoints"
        
        # Training hyperparameters
        self.num_epochs = 5
        self.batch_size = 8
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
        
        # Generation settings
        self.max_gen_length = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
        self.num_beams = 4
        self.do_sample = True
        
        # Early stopping
        self.patience = 3
        self.min_delta = 0.005
        
        # Mixed precision
        self.use_mixed_precision = True
        
        # Scheduler type: 'linear' or 'cosine'
        self.scheduler_type = 'cosine'
        
        # Evaluation frequency (in steps)
        self.eval_steps = 500
        
        # Save frequency (in steps)
        self.save_steps = 1000
        
        # Multi-phase training settings
        self.enable_multi_phase = True
        self.phases = [
            {
                'name': 'Phase 1: Basic vocabulary',
                'complexity_threshold': 60,  # Low complexity examples (approx. 25th percentile)
                'learning_rate': 5e-5,
                'num_epochs': 3
            },
            {
                'name': 'Phase 2: Grammar and structure',
                'complexity_threshold': 90,  # Medium complexity examples (approx. 75th percentile)
                'learning_rate': 3e-5,
                'num_epochs': 2
            },
            {
                'name': 'Phase 3: Full translation',
                'complexity_threshold': float('inf'),  # Train on all examples
                'learning_rate': 2e-5,
                'num_epochs': 2
            }
        ]

# Helper functions
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def prepare_data_for_training(examples):
    processed_data = []
    for example in examples:
        # Check example structure
        if 'messages' in example and len(example['messages']) >= 2:
            user_message = example['messages'][0]
            assistant_message = example['messages'][1]
            
            if user_message['role'] == 'user' and assistant_message['role'] == 'assistant':
                input_text = user_message['content']
                output_text = assistant_message['content']
                
                # Determine if this is English-to-Yanomami or Yanomami-to-English
                if 'translate' in input_text.lower() and 'yanomami' in input_text.lower():
                    # This is English-to-Yanomami
                    processed_data.append({"input": f"English: {input_text} => Yanomami:", "output": output_text})
                elif 'translate' in input_text.lower() and 'english' in input_text.lower():
                    # This is Yanomami-to-English
                    processed_data.append({"input": f"Yanomami: {input_text} => English:", "output": output_text})
                elif 'mean' in input_text.lower() and 'yanomami' in input_text.lower():
                    # This is a definition query
                    processed_data.append({"input": f"English: {input_text} => Yanomami:", "output": output_text})
                else:
                    # Default case - assume English-to-Yanomami
                    processed_data.append({"input": f"English: {input_text} => Yanomami:", "output": output_text})
    
    return processed_data

def tokenize_function(examples, tokenizer, config):
    # Initialize lists to store processed data
    input_texts = examples['input']
    output_texts = examples['output']
    
    # For training, we need both input and output together
    combined_texts = [f"{input_text} {output_text}" for input_text, output_text in zip(input_texts, output_texts)]
    
    # Return empty dict if no texts to process
    if not combined_texts:
        logger.warning("No valid examples found in batch.")
        return {"input_ids": [], "attention_mask": []}
    
    # Tokenize combined texts
    tokenized = tokenizer(
        combined_texts, 
        padding=config.padding, 
        truncation=config.truncation, 
        max_length=config.max_length
    )
    
    return tokenized

class GPT2Dataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        # Get example from dataset
        example = self.encodings[idx]
        
        # Handle different types of encodings
        if isinstance(example, dict):
            # Check if values are already tensors
            if isinstance(example.get("input_ids"), torch.Tensor):
                return {
                    "input_ids": example["input_ids"],
                    "attention_mask": example["attention_mask"],
                    "labels": example["input_ids"].clone()
                }
            else:
                # Convert to tensors if they're not already
                return {
                    "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(example["input_ids"], dtype=torch.long)
                }
        else:
            # Handle unexpected format
            raise ValueError(f"Unexpected example format: {type(example)}")

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
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, tokenizer, step, val_loss)
            return False
        else:
            self.counter += 1
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

def plot_training_metrics(train_losses, val_losses, output_dir):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    logger.info(f"Training metrics plot saved to {plot_path}")

def generate_translation(text, model, tokenizer, config, prefix_type="english_to_yanomami"):
    """
    Generate translation using the fine-tuned model.
    
    Args:
        text (str): Text to translate
        model: The fine-tuned model
        tokenizer: The tokenizer
        config: Configuration object
        prefix_type (str): Type of translation ('english_to_yanomami' or 'yanomami_to_english')
        
    Returns:
        str: The generated translation
    """
    # Add appropriate prefix based on translation direction
    if prefix_type == "english_to_yanomami":
        if "translate" in text.lower():
            prompt = f"English: {text} => Yanomami:"
        else:
            prompt = f"English: Translate this to Yanomami: {text} => Yanomami:"
    else:
        if "translate" in text.lower():
            prompt = f"Yanomami: {text} => English:"
        else:
            prompt = f"Yanomami: Translate this to English: {text} => English:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    
    # Generate translation
    outputs = model.generate(
        **inputs,
        max_length=config.max_gen_length,
        num_return_sequences=1,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        num_beams=config.num_beams,
        do_sample=config.do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the actual translation part (after the prompt)
    if "=>" in translation:
        translation = translation.split("=>")[1].strip()
    
    return translation

def load_yanomami_translator(model_path):
    """
    Load the model and tokenizer for Yanomami-English translation.
    
    Args:
        model_path (str): Path to directory containing model and tokenizer
        
    Returns:
        tuple: (model, tokenizer) loaded and ready for use
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory {model_path} does not exist.")
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Configure device for inference
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    return model, tokenizer

def main():
    # Initialize configuration
    config = TranslatorConfig()
    
    print("\n=== STARTING QUICK TEST OF MULTI-PHASE TRAINING ===")
    print("This will run a small test to verify the code works correctly")
    
    # Modify config for quick test
    config.num_epochs = 1
    config.batch_size = 2
    config.gradient_accumulation_steps = 2
    config.eval_steps = 10
    config.save_steps = 20
    config.dataset_files = [
        'grammar-plural.jsonl',  # Small dataset for testing
    ]
    print(f"Using dataset file: {config.dataset_files[0]}")
    
    # Set debug mode to True for verbose output
    config.debug_mode = True
    print(f"Debug mode: {config.debug_mode}")
    
    # Print device information
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Configure multi-phase training with smaller thresholds for testing
    config.enable_multi_phase = True
    config.phases = [
        {
            'name': 'Phase 1: Basic vocabulary',
            'complexity_threshold': 5,  # Very low for testing
            'learning_rate': 5e-5,
            'num_epochs': 1
        },
        {
            'name': 'Phase 2: Grammar and structure',
            'complexity_threshold': 10,  # Low for testing
            'learning_rate': 3e-5,
            'num_epochs': 1
        },
        {
            'name': 'Phase 3: Full translation',
            'complexity_threshold': float('inf'),  # All examples
            'learning_rate': 2e-5,
            'num_epochs': 1
        }
    ]
    
    # Set output directories for test
    config.model_output_dir = "./test_yanomami_translator"
    config.checkpoint_dir = "./test_checkpoints"
    
    # Create output directories
    os.makedirs(config.model_output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print(f"Created output directories: {config.model_output_dir} and {config.checkpoint_dir}")
    
    # Load and prepare data
    logger.info("Loading dataset files...")
    print("Loading dataset files...")
    all_data = []
    
    for file in config.dataset_files:
        try:
            file_path = os.path.join(config.dataset_path, file)
            print(f"Attempting to load file: {file_path}")
            file_data = load_jsonl(file_path)
            all_data.extend(file_data)
            print(f"Successfully loaded {len(file_data)} examples from {file}")
            logger.info(f"Loaded {len(file_data)} examples from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            logger.error(f"Error loading {file}: {e}")
    
    logger.info(f"Total examples loaded: {len(all_data)}")
    
    # Process data
    processed_data = prepare_data_for_training(all_data)
    logger.info(f"Total processed examples: {len(processed_data)}")
    
    # Implement curriculum learning by sorting examples by complexity
    def calculate_complexity(example):
        # Calculate complexity based on multiple factors
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
    
    # Sort data by complexity
    for example in processed_data:
        example['complexity'] = calculate_complexity(example)
    
    sorted_data = sorted(processed_data, key=lambda x: x['complexity'])
    logger.info(f"Data sorted by complexity for curriculum learning")
    
    # Split data
    train_data, val_data = train_test_split(sorted_data, test_size=0.1, random_state=42)
    logger.info(f"Training examples: {len(train_data)}")
    logger.info(f"Validation examples: {len(val_data)}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Load model and tokenizer
    if os.path.exists(config.model_output_dir) and os.path.isdir(config.model_output_dir):
        logger.info(f"Loading model and tokenizer from {config.model_output_dir}")
        tokenizer = GPT2Tokenizer.from_pretrained(config.model_output_dir)
        model = GPT2LMHeadModel.from_pretrained(config.model_output_dir)
    else:
        logger.info(f"Loading pre-trained model and tokenizer")
        
        # Check if we have a custom Yanomami tokenizer
        yanomami_tokenizer_path = './yanomami_tokenizer/complete_yanomami_tokenizer'
        if os.path.exists(yanomami_tokenizer_path):
            logger.info(f"Loading Yanomami-specific tokenizer from {yanomami_tokenizer_path}")
            tokenizer = GPT2Tokenizer.from_pretrained(yanomami_tokenizer_path)
        else:
            logger.info(f"Loading default tokenizer from {config.model_name}")
            tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
            
        model = GPT2LMHeadModel.from_pretrained(config.model_name)
        
        # Resize token embeddings to match the tokenizer vocabulary size
        logger.info(f"Resizing token embeddings from {model.get_input_embeddings().num_embeddings} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Store original training data for phases
    all_train_data = train_data.copy()
    
    # Prepare validation dataset once (used across all phases)
    logger.info("Preparing validation dataset...")
    val_dataset_obj = Dataset.from_list(val_data)
    tokenized_val = val_dataset_obj.map(
        lambda examples: tokenize_function(examples, tokenizer, config), 
        batched=True
    )
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask"])
    val_dataset = GPT2Dataset(tokenized_val)
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size
    )
    
    # Multi-phase training
    try:
        if config.enable_multi_phase:
            logger.info("Starting multi-phase training...")
            overall_start_time = time.time()
        
            for phase_idx, phase in enumerate(config.phases):
                logger.info(f"\n{'='*50}\n{phase['name']}\n{'='*50}")
                
                # Filter data based on complexity threshold for this phase
                phase_train_data = [ex for ex in all_train_data if ex['complexity'] <= phase['complexity_threshold']]
                logger.info(f"Phase {phase_idx+1} training on {len(phase_train_data)} examples (complexity <= {phase['complexity_threshold']})")
                
                # Create dataset for this phase
                phase_train_dataset = Dataset.from_list(phase_train_data)
                
                # Tokenize datasets for this phase
                logger.info(f"Tokenizing phase {phase_idx+1} dataset...")
                tokenized_train = phase_train_dataset.map(
                    lambda examples: tokenize_function(examples, tokenizer, config), 
                    batched=True
                )
                
                # Set format for PyTorch
                tokenized_train.set_format("torch", columns=["input_ids", "attention_mask"])
                
                # Create PyTorch datasets
                train_dataset = GPT2Dataset(tokenized_train)
                
                # Create data loader for this phase
                train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=config.batch_size, 
                    shuffle=True
                )
                
                # Calculate training steps for this phase
                total_steps = len(train_dataloader) * phase['num_epochs'] // config.gradient_accumulation_steps
                warmup_steps = int(total_steps * config.warmup_ratio)
                
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
                
                # Configure scheduler
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
                        logger.info(f"Starting epoch {epoch+1}/{phase['num_epochs']} of phase {phase_idx+1}")
                        epoch_loss = 0
                        model.train()
                
                        # Track batch progress
                        total_batches = len(train_dataloader)
                        
                        # Training
                        for batch_idx, batch in enumerate(train_dataloader):
                            # Move batch to device
                            batch = {k: v.to(device) for k, v in batch.items()}
                
                            # Mixed precision training
                            if config.use_mixed_precision and torch.cuda.is_available():
                                with autocast():
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
                                    scheduler.step()
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
                                
                                    # Update weights
                                    optimizer.step()
                                    scheduler.step()
                                    optimizer.zero_grad()
                                    global_step += 1
                
                        # Accumulate loss
                        epoch_loss += loss.item() * config.gradient_accumulation_steps
                        
                        # Log progress
                        if batch_idx % 10 == 0:
                            progress_percent = (batch_idx + 1) / total_batches * 100
                            remaining_batches = total_batches - batch_idx - 1
                            elapsed_time = time.time() - start_time
                            
                            logger.info(
                                f"Epoch {epoch+1}/{phase['num_epochs']} of phase {phase_idx+1}, "
                                f"Batch {batch_idx+1}/{total_batches} ({progress_percent:.2f}%), "
                                f"Loss: {loss.item()*config.gradient_accumulation_steps:.4f}, "
                                f"Elapsed: {elapsed_time:.2f}s"
                            )
                
                        # Evaluate and save checkpoint
                        if global_step % config.eval_steps == 0:
                            # Evaluate
                            eval_loss = evaluate_model(model, eval_dataloader, device)
                            val_losses.append(eval_loss)
                            train_losses.append(epoch_loss / (batch_idx + 1))
                            
                            logger.info(f"Step {global_step}: Validation Loss: {eval_loss:.4f}")
                            
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
            
                    # Calculate average epoch loss
                    avg_epoch_loss = epoch_loss / len(train_dataloader)
                    logger.info(f"Epoch {epoch+1}/{phase['num_epochs']} of phase {phase_idx+1} completed. Average loss: {avg_epoch_loss:.4f}")
                    
                    # Evaluate at end of epoch
                    eval_loss = evaluate_model(model, eval_dataloader, device)
                    val_losses.append(eval_loss)
                    train_losses.append(avg_epoch_loss)
                    
                    logger.info(f"End of epoch {epoch+1}/{phase['num_epochs']} of phase {phase_idx+1}: Validation Loss: {eval_loss:.4f}")
                    
                    # Check early stopping
                    if early_stopping(eval_loss, model, tokenizer, global_step):
                        logger.info("Early stopping triggered")
                        break
        
                except Exception as e:
                    logger.error(f"Error during phase {phase_idx+1} training: {e}")
                    raise
                    
                # Phase completed
                phase_time = time.time() - start_time
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
        
        # Plot training metrics
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
    except Exception as e:
        logger.error(f"Error during overall training process: {e}")
        raise

def evaluate_model(model, eval_dataloader, device):
    """Evaluate the model on validation data"""
    model.eval()
    total_eval_loss = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.item()
    
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    model.train()
    return avg_eval_loss

def test_translations(model, tokenizer, config):
    """Test the model with sample translations"""
    logger.info("\n===== TESTING TRANSLATIONS =====")
    
    test_phrases = [
        ("What does 'aheprariyo' mean in Yanomami?", "english_to_yanomami"),
        ("Hello, how are you?", "english_to_yanomami"),
        ("I am learning Yanomami", "english_to_yanomami"),
        ("What is your name?", "english_to_yanomami"),
        ("Thank you for your help", "english_to_yanomami"),
        ("aheprariyo", "yanomami_to_english"),
        ("Kami yanomae thë ã", "yanomami_to_english"),
        ("thë aheai", "yanomami_to_english"),
        ("Weti tha?", "yanomami_to_english"),
        ("Kami yai huë", "yanomami_to_english")
    ]
    
    for phrase, direction in test_phrases:
        logger.info(f"\nInput ({direction}): {phrase}")
        translation = generate_translation(phrase, model, tokenizer, config, direction)
        logger.info(f"Translation: {translation}")