#!/usr/bin/env python3
"""
Validate Tokenizer Alignment with Yanomami Dataset
--------------------------------------------------
This script analyzes how well the GPT-2 tokenizer handles Yanomami text
by examining token distributions, unknown tokens, and tokenization patterns.
"""

import os
import json
import re
import torch
from transformers import GPT2Tokenizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = '/Users/renanserrano/CascadeProjects/Yanomami/finetunning/yanomami_dataset/'
MODEL_PATH = './gpt2_yanomami_translator'
DATASET_FILES = [
    'translations.jsonl',
    'yanomami-to-english.jsonl',
    'phrases.jsonl',
    'grammar.jsonl',
    'comparison.jsonl',
    'how-to.jsonl'
]

def load_jsonl(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_yanomami_text(examples):
    """Extract Yanomami text from dataset examples."""
    yanomami_texts = []
    english_texts = []
    
    # Yanomami characters and patterns
    yanomami_chars = {'ë', 'ã', 'ï', 'ö', 'ñ', 'ü', 'ê', 'õ', 'ô', 'á', 'é', 'í', 'ó', 'ú', 'à', 'è', 'ì', 'ò', 'ù'}
    
    def is_likely_yanomami(text):
        # Check if text has Yanomami-specific characters or patterns
        if any(char in text for char in yanomami_chars):
            return True
        # Check for common Yanomami word patterns (ending with -ma, -pë, etc.)
        if any(text.endswith(suffix) for suffix in ['ma', 'pë', 'ri', 'ha', 'he', 'ki', 'ko', 'na', 'po', 'ra', 'ro', 'ru', 'wa', 'we', 'wi', 'ya', 'yo', 'yu']):
            return True
        return False
    
    for example in examples:
        if 'messages' in example and len(example['messages']) >= 2:
            user_message = example['messages'][0]['content'].lower()
            assistant_message = example['messages'][1]['content']
            
            # Extract text from user message
            if 'yanomami' in user_message:
                # Try to find phrases between quotes
                quotes = re.findall(r'["\']([^"\']*)["\']', user_message)
                for quote in quotes:
                    if len(quote) > 0:
                        if is_likely_yanomami(quote):
                            yanomami_texts.append(quote)
                        else:
                            english_texts.append(quote)
            
            # Extract from assistant message if user asked about Yanomami
            if 'yanomami' in user_message:
                # Extract words that look like Yanomami from assistant response
                quotes = re.findall(r'["\']([^"\']*)["\']', assistant_message)
                for quote in quotes:
                    if len(quote) > 0 and is_likely_yanomami(quote):
                        yanomami_texts.append(quote)
    
    # Remove duplicates while preserving order
    unique_yanomami = []
    for text in yanomami_texts:
        if text not in unique_yanomami:
            unique_yanomami.append(text)
    
    return unique_yanomami

def analyze_tokenization(tokenizer, texts):
    """Analyze how the tokenizer handles the given texts."""
    token_counts = Counter()
    token_lengths = []
    unknown_tokens = Counter()
    character_coverage = {}
    token_to_texts = {}
    
    # Analyze character distribution in the dataset
    all_chars = ''.join(texts)
    char_counts = Counter(all_chars)
    total_chars = len(all_chars)
    
    for text in texts:
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        token_counts.update(tokens)
        token_lengths.append(len(tokens))
        
        # Track which texts each token appears in
        for token in set(tokens):
            if token not in token_to_texts:
                token_to_texts[token] = []
            token_to_texts[token].append(text)
        
        # Check for unknown tokens
        for token in tokens:
            if token == tokenizer.unk_token:
                unknown_tokens.update([text])
    
    # Calculate character coverage
    for char, count in char_counts.items():
        character_coverage[char] = {
            'count': count,
            'percentage': (count / total_chars) * 100
        }
    
    return {
        'token_counts': token_counts,
        'token_lengths': token_lengths,
        'unknown_tokens': unknown_tokens,
        'character_coverage': character_coverage,
        'token_to_texts': token_to_texts
    }

def plot_token_distribution(token_lengths, output_path='token_distribution.png'):
    """Plot the distribution of token lengths."""
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title('Distribution of Token Lengths for Yanomami Texts')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    logger.info(f"Token distribution plot saved to {output_path}")

def plot_character_distribution(character_coverage, output_path='character_distribution.png'):
    """Plot the distribution of character frequencies."""
    # Sort characters by frequency
    sorted_chars = sorted(character_coverage.items(), key=lambda x: x[1]['count'], reverse=True)
    chars = [char for char, _ in sorted_chars[:30]]  # Top 30 characters
    counts = [data['count'] for _, data in sorted_chars[:30]]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(chars)), counts, color='green', alpha=0.7)
    plt.xticks(range(len(chars)), chars, rotation=45)
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.title('Distribution of Character Frequencies in Yanomami Texts')
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Character distribution plot saved to {output_path}")

def main():
    # Load tokenizer
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading tokenizer from {MODEL_PATH}")
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    else:
        logger.info("Loading default GPT-2 tokenizer")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load dataset
    all_data = []
    for file in DATASET_FILES:
        try:
            file_path = os.path.join(DATASET_PATH, file)
            file_data = load_jsonl(file_path)
            all_data.extend(file_data)
            logger.info(f"Loaded {len(file_data)} examples from {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    logger.info(f"Total examples loaded: {len(all_data)}")
    
    # Extract Yanomami text
    yanomami_texts = extract_yanomami_text(all_data)
    logger.info(f"Extracted {len(yanomami_texts)} Yanomami texts")
    
    if not yanomami_texts:
        logger.warning("No Yanomami texts found in the dataset!")
        return
    
    # Sample some texts
    logger.info("Sample Yanomami texts:")
    for i, text in enumerate(yanomami_texts[:5]):
        logger.info(f"  {i+1}. '{text}'")
    
    # Analyze tokenization
    analysis = analyze_tokenization(tokenizer, yanomami_texts)
    
    # Report statistics
    logger.info("\n===== TOKENIZATION ANALYSIS =====")
    logger.info(f"Total unique tokens: {len(analysis['token_counts'])}")
    
    avg_tokens = np.mean(analysis['token_lengths'])
    median_tokens = np.median(analysis['token_lengths'])
    max_tokens = max(analysis['token_lengths'])
    min_tokens = min(analysis['token_lengths'])
    
    logger.info(f"Average tokens per text: {avg_tokens:.2f}")
    logger.info(f"Median tokens per text: {median_tokens}")
    logger.info(f"Min tokens per text: {min_tokens}")
    logger.info(f"Max tokens per text: {max_tokens}")
    
    # Report most common tokens
    logger.info("\nMost common tokens:")
    for token, count in analysis['token_counts'].most_common(20):
        logger.info(f"  '{token}': {count}")
    
    # Report unknown tokens
    if analysis['unknown_tokens']:
        logger.info("\nTexts with unknown tokens:")
        for text, count in analysis['unknown_tokens'].most_common(10):
            logger.info(f"  '{text}': {count} unknown tokens")
        logger.warning(f"Total texts with unknown tokens: {len(analysis['unknown_tokens'])}")
    else:
        logger.info("\nNo unknown tokens found! The tokenizer handles all Yanomami texts.")
    
    # Report character coverage
    logger.info("\n===== CHARACTER ANALYSIS =====")
    # Sort by frequency
    sorted_chars = sorted(analysis['character_coverage'].items(), 
                         key=lambda x: x[1]['count'], 
                         reverse=True)
    
    logger.info(f"Total unique characters: {len(sorted_chars)}")
    logger.info("\nMost common characters:")
    for char, data in sorted_chars[:20]:
        logger.info(f"  '{char}': {data['count']} ({data['percentage']:.2f}%)")
    
    # Plot distributions
    plot_token_distribution(analysis['token_lengths'])
    plot_character_distribution(analysis['character_coverage'], 'character_distribution.png')

if __name__ == "__main__":
    main()
