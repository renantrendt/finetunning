# Comprehensive Testing Script for Multi-Phase Training
# This script validates the performance of the Yanomami-English translation model
# through detailed complexity analysis, performance metrics tracking, and enhanced error handling.

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comprehensive_testing.log", mode='w')
    ]
)

# Function to process Yanomami text with special character handling
def process_yanomami_text(text):
    # Handle special character ɨ (U+0268)
    text = text.replace('ɨ', 'i')
    return text

# Function to load and process dataset
def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            if 'messages' in example and len(example['messages']) >= 2:
                user_message = example['messages'][0]
                assistant_message = example['messages'][1]
                
                if user_message['role'] == 'user' and assistant_message['role'] == 'assistant':
                    input_text = process_yanomami_text(user_message['content'])
                    output_text = process_yanomami_text(assistant_message['content'])
                    
                    # Format for training
                    data.append({
                        'input': f"Translate: {input_text}",
                        'output': output_text
                    })
    return data

# Function to visualize complexity statistics
def visualize_complexity_statistics(complexity_stats):
    plt.figure(figsize=(10, 6))
    plt.hist(complexity_stats, bins=30, color='blue', alpha=0.7)
    plt.title('Complexity Distribution of Training Examples')
    plt.xlabel('Complexity Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('complexity_distribution.png')
    plt.show()

# Function to run multi-phase training
def run_multi_phase_training(config, dataset):
    print(f"Starting multi-phase training with {len(dataset)} examples")
    logging.info(f"Starting multi-phase training with {len(dataset)} examples")
    def collate_fn(batch):
        input_texts = [item['input'] for item in batch]
        output_texts = [item['output'] for item in batch]
        return {
            'input': input_texts,
            'output': output_texts
        }
    
    # Split dataset into training and validation
    train_data, val_data = train_test_split(dataset, test_size=0.1)
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config['device'] = device

    # Load and configure tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and configure model
    model = GPT2LMHeadModel.from_pretrained(config['model_name']).to(device)
    model.config.pad_token_id = model.config.eos_token_id

    # Initialize performance metrics
    train_losses = []
    val_losses = []

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    for phase in config['phases']:
        logging.info(f'Starting {phase["name"]}')
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            try:
                # Prepare input text
                input_text = batch['input']
                output_text = batch['output']
                
                # Tokenize input and output with consistent padding
                inputs = tokenizer(
                    input_text,
                    padding='max_length',
                    truncation=True,
                    max_length=config['max_length'],
                    return_tensors='pt'
                )
                labels = tokenizer(
                    output_text,
                    padding='max_length',
                    truncation=True,
                    max_length=config['max_length'],
                    return_tensors='pt'
                )
                
                # Move tensors to device
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                label_ids = labels['input_ids'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids
                )
                loss = outputs.loss
                
                # Log progress
                total_loss += loss.item()
                train_losses.append(loss.item())
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Log batch progress
                if len(train_losses) % 5 == 0:  # Log every 5 batches
                    logging.info(f'Batch {len(train_losses)}: Loss = {loss.item():.4f}')
                    
            except Exception as e:
                logging.error(f'Error during training: {str(e)}')
                logging.error(f'Failed batch input: {input_text}')
                logging.error(f'Failed batch output: {output_text}')

        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f'Average training loss for {phase["name"]}: {avg_train_loss}')

        # Validation phase
        model.eval()
        total_val_loss = 0
        batch_val_losses = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_text = batch['input']
                output_text = batch['output']
                
                # Tokenize validation data with consistent padding
                inputs = tokenizer(
                    input_text,
                    padding='max_length',
                    truncation=True,
                    max_length=config['max_length'],
                    return_tensors='pt'
                )
                labels = tokenizer(
                    output_text,
                    padding='max_length',
                    truncation=True,
                    max_length=config['max_length'],
                    return_tensors='pt'
                )
                
                # Move tensors to device
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                label_ids = labels['input_ids'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids
                )
                val_loss = outputs.loss.item()
                total_val_loss += val_loss
                batch_val_losses.append(val_loss)
                
                # Log validation batch progress
                if len(batch_val_losses) % 2 == 0:  # Log every 2 validation batches
                    logging.info(f'Validation Batch {len(batch_val_losses)}: Loss = {val_loss:.4f}')

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.extend(batch_val_losses)  # Add all validation batch losses
        logging.info(f'Average validation loss for {phase["name"]}: {avg_val_loss}')

    # Visualize training and validation losses
    plt.figure(figsize=(12, 6))
    
    # Plot training losses
    plt.plot(range(len(train_losses)), train_losses, 'b-', label='Training Loss', alpha=0.6)
    
    # Plot validation losses with proper x-axis alignment
    val_steps = range(0, len(train_losses), len(train_losses)//len(val_losses))
    val_losses_interp = np.interp(range(len(train_losses)), val_steps, val_losses)
    plt.plot(range(len(train_losses)), val_losses_interp, 'r-', label='Validation Loss', alpha=0.6)
    
    plt.title('Training and Validation Losses Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_validation_losses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the trained model and tokenizer
    output_dir = 'yanomami_translator_model'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f'Model and tokenizer saved to {output_dir}')
    
    # Save training configuration
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    logging.info('Training configuration saved')


def main():
    print("Starting comprehensive testing script...")
    logging.info("Starting comprehensive testing script...")
    
    # Load dataset
    dataset_path = '/Users/renanserrano/CascadeProjects/Yanomami/finetunning/yanomami_dataset/grammar-plural.jsonl'
    print(f"Loading dataset from {dataset_path}")
    logging.info(f"Loading dataset from {dataset_path}")
    
    try:
        dataset = load_dataset(dataset_path)
        print(f"Successfully loaded {len(dataset)} examples")
        logging.info(f"Successfully loaded {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        logging.error(f"Error loading dataset: {str(e)}")
        return
    
    # Configuration
    config = {
        'model_name': 'gpt2',
        'batch_size': 4,
        'learning_rate': 5e-5,
        'max_length': 128,  # Maximum sequence length for input and output
        'phases': [
            {
                'name': 'Phase 1: Basic vocabulary',
                'complexity_threshold': 60,
                'learning_rate': 5e-5
            },
            {
                'name': 'Phase 2: Grammar and structure',
                'complexity_threshold': 90,
                'learning_rate': 3e-5
            },
            {
                'name': 'Phase 3: Full translation',
                'complexity_threshold': float('inf'),
                'learning_rate': 2e-5
            }
        ]
    }
    
    # Run training
    try:
        run_multi_phase_training(config, dataset)
        print("Training completed successfully!")
        logging.info("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        logging.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()
