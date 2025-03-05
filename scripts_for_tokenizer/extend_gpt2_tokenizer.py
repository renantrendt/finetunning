#!/usr/bin/env python3
"""
Script to extend GPT-2's tokenizer with Yanomami-specific tokens.

This script:
1. Loads the standard GPT-2 tokenizer
2. Extracts Yanomami words and special tokens from your dataset
3. Adds these tokens to the tokenizer
4. Saves the extended tokenizer
5. Tests the tokenization on sample Yanomami text
"""

import os
import json
import re
import argparse
from collections import Counter
from typing import List, Dict, Set, Tuple
import time

# Import transformers
try:
    from transformers import AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizerFast
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with 'pip install transformers torch'")


def extract_special_tokens(jsonl_file: str) -> Set[str]:
    """Extract all special XML-like tokens from the dataset."""
    special_tokens = set()
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                for message in data.get('messages', []):
                    content = message.get('content', '')
                    # Find all XML-like tags
                    tags = re.findall(r'<([A-Z_]+)>', content)
                    for tag in tags:
                        special_tokens.add(f"<{tag}>")
                        special_tokens.add(f"</{tag}>")
            except json.JSONDecodeError:
                continue
    return special_tokens


def extract_yanomami_words(jsonl_file: str, min_frequency: int = 5) -> List[str]:
    """Extract frequent Yanomami words from the dataset."""
    word_counter = Counter()
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Extract words between <WORD> tags
                for message in data.get('messages', []):
                    content = message.get('content', '')
                    word_matches = re.findall(r'<WORD>([^<]+)</WORD>', content)
                    for word in word_matches:
                        word_counter[word] += 1
                    
                    # Also extract Yanomami examples
                    example_matches = re.findall(r'<EXAMPLE_YANOMAMI>([^<]+)</EXAMPLE_YANOMAMI>', content)
                    for example in example_matches:
                        if example.strip():  # Skip empty examples
                            # Split by spaces to get individual words
                            example_words = re.findall(r'\b\w+\b', example)
                            for word in example_words:
                                word_counter[word] += 1
            except json.JSONDecodeError:
                continue
    
    # Filter by frequency
    frequent_words = [word for word, count in word_counter.most_common() 
                     if count >= min_frequency]
    
    return frequent_words


def test_tokenization(tokenizer, test_texts: List[str]) -> None:
    """Test the tokenizer on sample texts and print results."""
    print("\nTesting tokenization on sample texts:")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded back: {tokenizer.decode(token_ids)}")
        print(f"Number of tokens: {len(tokens)}")


def extend_gpt2_tokenizer(jsonl_file: str, output_dir: str, 
                         max_yanomami_words: int = 1000,
                         min_word_frequency: int = 5) -> None:
    """Extend GPT-2's tokenizer with Yanomami-specific tokens."""
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library is required for this script.")
        return
    
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading base GPT-2 tokenizer...")
    # Load the base GPT-2 tokenizer
    base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Extract special tokens from the dataset
    print(f"Extracting special tokens from {jsonl_file}...")
    special_tokens = extract_special_tokens(jsonl_file)
    print(f"Found {len(special_tokens)} special tokens")
    
    # Extract frequent Yanomami words
    print(f"Extracting Yanomami words with frequency >= {min_word_frequency}...")
    yanomami_words = extract_yanomami_words(jsonl_file, min_frequency=min_word_frequency)
    print(f"Found {len(yanomami_words)} frequent Yanomami words")
    
    # Limit the number of words to add
    if max_yanomami_words > 0 and len(yanomami_words) > max_yanomami_words:
        print(f"Limiting to top {max_yanomami_words} most frequent words")
        yanomami_words = yanomami_words[:max_yanomami_words]
    
    # Combine special tokens and Yanomami words
    all_new_tokens = list(special_tokens) + yanomami_words
    
    # Add the new tokens to the tokenizer
    print(f"Adding {len(all_new_tokens)} new tokens to the tokenizer...")
    num_added = base_tokenizer.add_tokens(all_new_tokens)
    print(f"Successfully added {num_added} new tokens")
    
    # Save the extended tokenizer
    tokenizer_path = os.path.join(output_dir, 'yanomami_gpt2_tokenizer')
    base_tokenizer.save_pretrained(tokenizer_path)
    print(f"Saved extended tokenizer to {tokenizer_path}")
    
    # Create a vocabulary map file for reference
    vocab_map = {}
    for token, token_id in base_tokenizer.get_vocab().items():
        vocab_map[token] = token_id
    
    vocab_map_path = os.path.join(output_dir, 'yanomami_vocab_map.json')
    with open(vocab_map_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_map, f, indent=2, ensure_ascii=False)
    print(f"Saved vocabulary map to {vocab_map_path}")
    
    # Test the tokenizer on some sample texts
    print("\nLoading the extended tokenizer for testing...")
    extended_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    
    # Create some test texts using Yanomami words and special tokens
    test_texts = [
        # Test with a special token and a Yanomami word
        f"<WORD>{yanomami_words[0]}</WORD> means something in Yanomami.",
        
        # Test with multiple special tokens
        f"<WORD>{yanomami_words[1]}</WORD> <POS>Noun</POS> <DEFINITION>a meaning</DEFINITION>",
        
        # Test with an example
        f"<EXAMPLE_YANOMAMI>{' '.join(yanomami_words[2:5])}</EXAMPLE_YANOMAMI>"
    ]
    
    test_tokenization(extended_tokenizer, test_texts)
    
    # Compare with the original tokenizer
    print("\nComparing with the original GPT-2 tokenizer:")
    original_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    for text in test_texts:
        original_tokens = original_tokenizer.tokenize(text)
        extended_tokens = extended_tokenizer.tokenize(text)
        print(f"\nText: {text}")
        print(f"Original tokenizer: {len(original_tokens)} tokens")
        print(f"Extended tokenizer: {len(extended_tokens)} tokens")
        print(f"Token reduction: {len(original_tokens) - len(extended_tokens)} tokens ({(1 - len(extended_tokens)/len(original_tokens))*100:.1f}%)")
    
    # Print instructions for using the tokenizer with a model
    print("\n" + "=" * 80)
    print("NEXT STEPS FOR USING THE EXTENDED TOKENIZER WITH GPT-2:")
    print("=" * 80)
    print("1. Load the model and tokenizer:")
    print("   ```python")
    print("   from transformers import GPT2LMHeadModel, GPT2Tokenizer")
    print(f"   tokenizer = GPT2Tokenizer.from_pretrained('{tokenizer_path}')")
    print("   model = GPT2LMHeadModel.from_pretrained('gpt2')")
    print("   ```")
    print("\n2. Resize the model's token embeddings to match the new tokenizer:")
    print("   ```python")
    print("   model.resize_token_embeddings(len(tokenizer))")
    print("   ```")
    print("\n3. Now you can fine-tune the model with your Yanomami dataset")
    print("\n4. When saving the fine-tuned model, make sure to save both the model and tokenizer:")
    print("   ```python")
    print("   model.save_pretrained('yanomami_gpt2_model')")
    print("   tokenizer.save_pretrained('yanomami_gpt2_model')")
    print("   ```")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Extend GPT-2 tokenizer with Yanomami-specific tokens')
    parser.add_argument('jsonl_file', help='Path to the JSONL dataset file')
    parser.add_argument('--output_dir', default='./yanomami_tokenizer', help='Directory to save the extended tokenizer')
    parser.add_argument('--max_words', type=int, default=1000, help='Maximum number of Yanomami words to add')
    parser.add_argument('--min_frequency', type=int, default=5, help='Minimum frequency for Yanomami words to be included')
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_file):
        print(f"Error: File {args.jsonl_file} does not exist")
        return
    
    extend_gpt2_tokenizer(
        args.jsonl_file, 
        args.output_dir,
        max_yanomami_words=args.max_words,
        min_word_frequency=args.min_frequency
    )


if __name__ == "__main__":
    main()
