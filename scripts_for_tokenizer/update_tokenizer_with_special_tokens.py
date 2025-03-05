#!/usr/bin/env python3
"""
Update Tokenizer with Special Tokens
---------------------------------
This script updates the GPT-2 tokenizer with special tokens for the Yanomami language project.
It adds the special tokens to the tokenizer vocabulary and saves the updated tokenizer.
"""

import os
import json
from transformers import GPT2Tokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Special tokens from the translations_phrases_special_tokens.py script
SPECIAL_TOKENS = [
    '<WORD>', '</WORD>', 
    '<POS>', '</POS>', 
    '<DEFINITION>', '</DEFINITION>', 
    '<EXAMPLES>', '</EXAMPLES>', 
    '<EXAMPLE_YANOMAMI>', '</EXAMPLE_YANOMAMI>',
    '<EXAMPLE_TRANSLATION>', '</EXAMPLE_TRANSLATION>',
    '<QUERY>', '</QUERY>',
    '<YANOMAMI>', '</YANOMAMI>', 
    '<TRANSLATION>', '</TRANSLATION>', 
    '<LITERAL>', '</LITERAL>',
    '<RELATED_FORMS>', '</RELATED_FORMS>',
    '<USAGE>', '</USAGE>',
    '<GRAMMATICAL>', '</GRAMMATICAL>'
]

def update_tokenizer(model_path):
    """
    Update the tokenizer with special tokens and save it back to the model directory.
    
    Args:
        model_path (str): Path to the model directory containing the tokenizer files
    """
    try:
        # Load the tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Add special tokens
        logger.info(f"Adding {len(SPECIAL_TOKENS)} special tokens to the tokenizer")
        special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}
        num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_tokens} tokens to the tokenizer vocabulary")
        
        # Update the special_tokens_map.json file
        special_tokens_map_path = os.path.join(model_path, 'special_tokens_map.json')
        if os.path.exists(special_tokens_map_path):
            with open(special_tokens_map_path, 'r', encoding='utf-8') as f:
                special_tokens_map = json.load(f)
            
            # Add additional_special_tokens to the map if not already present
            if 'additional_special_tokens' not in special_tokens_map:
                special_tokens_map['additional_special_tokens'] = SPECIAL_TOKENS
            else:
                # Update existing additional_special_tokens
                current_tokens = special_tokens_map['additional_special_tokens']
                if isinstance(current_tokens, list):
                    # Add any tokens that aren't already in the list
                    for token in SPECIAL_TOKENS:
                        if token not in current_tokens:
                            current_tokens.append(token)
                else:
                    # If it's not a list, replace it with our list
                    special_tokens_map['additional_special_tokens'] = SPECIAL_TOKENS
            
            # Write the updated map back to the file
            with open(special_tokens_map_path, 'w', encoding='utf-8') as f:
                json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated special_tokens_map.json with additional special tokens")
        
        # Save the updated tokenizer
        logger.info(f"Saving updated tokenizer to {model_path}")
        tokenizer.save_pretrained(model_path)
        logger.info(f"Tokenizer successfully updated and saved")
        
        return tokenizer
    
    except Exception as e:
        logger.error(f"Error updating tokenizer: {e}")
        raise

def main():
    # Path to the model directory
    model_path = "./gpt2_yanomami_translator"
    
    # Update the tokenizer
    updated_tokenizer = update_tokenizer(model_path)
    
    # Print some information about the updated tokenizer
    logger.info(f"Tokenizer vocabulary size: {len(updated_tokenizer)}")
    logger.info(f"Tokenizer has the following special tokens:")
    for token_name, token_value in updated_tokenizer.special_tokens_map.items():
        if isinstance(token_value, list):
            logger.info(f"  {token_name}: {len(token_value)} tokens")
        else:
            logger.info(f"  {token_name}: {token_value}")

if __name__ == "__main__":
    main()
