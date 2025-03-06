# Tokenizer Enhancement Module
# This module provides functions to enhance the tokenizer's handling of special Yanomami characters,
# particularly the character 'ɨ' (U+0268)

import re
import os
from transformers import AutoTokenizer

# List of common Yanomami words containing the special character 'ɨ'
# These words will be added to the tokenizer vocabulary to ensure they are tokenized as single tokens
SPECIAL_CHAR_WORDS = [
    "kɨ", "Ɨhɨki", "Ɨhɨ", "kɨa", "pɨ", "ɨpɨtɨ", "ɨ", "rɨ", "kɨrɨ", "hɨ",
    "Ɨpɨtɨ", "wakɨ", "pɨtaha", "pɨtɨ", "ɨpɨ", "ɨhɨ", "ɨhɨki", "ɨwɨ",
    "hɨɨkɨrɨ", "hɨɨkɨ", "kɨri", "ɨpɨhɨ", "ɨpɨtɨwɨ", "sɨ", "yɨwɨ",
    "Wakɨ", "ɨpɨtɨhɨ", "pɨɨ", "ɨkɨɨ", "ɨkɨma", "ɨkɨmai", "ɨkɨmarei",
    "ɨkɨoprou", "ɨkɨrayou", "ɨkɨhiwë", "ɨkɨ", "pehihiprɨ", "ɨpɨtɨawɨ",
    "kamɨ", "hɨɨrɨ", "Kɨ", "ãpɨtɨ", "Yiiyɨ", "bɨtɨ", "ɨhurëpɨ",
    "Ũrihipɨ", "ɨtɨ", "hɨkɨri", "hɨtɨmɨ", "ãhiãmɨ"
]

# Character replacement mapping for fallback tokenization
CHAR_REPLACEMENTS = {
    'ɨ': 'i',  # Replace ɨ (U+0268) with i
    'ë': 'e',  # Replace ë with e
    'ã': 'a',  # Replace ã with a
    'ũ': 'u',  # Replace ũ with u
}

def enhance_tokenizer(tokenizer):
    """
    Enhance a tokenizer with special handling for Yanomami characters.
    
    Args:
        tokenizer: The tokenizer to enhance
        
    Returns:
        The enhanced tokenizer
    """
    # Add special words to the tokenizer vocabulary
    num_added = tokenizer.add_tokens(SPECIAL_CHAR_WORDS)
    print(f"Added {num_added} special Yanomami words to tokenizer vocabulary")
    
    # Store the original tokenizer methods
    original_tokenize = tokenizer.tokenize
    original_encode = tokenizer.encode
    
    # Override the tokenize method to handle special characters
    def enhanced_tokenize(text, *args, **kwargs):
        # First try with the original tokenizer
        tokens = original_tokenize(text, *args, **kwargs)
        
        # Check if any tokens contain the special character
        # If they do, it means the tokenizer didn't recognize them as whole tokens
        special_char_pattern = re.compile(f"[{''.join(CHAR_REPLACEMENTS.keys())}]")
        
        # If we find special characters in the tokens, try the fallback approach
        if any(special_char_pattern.search(token) for token in tokens if isinstance(token, str)):
            # Apply character replacements for fallback tokenization
            replaced_text = replace_special_chars(text)
            tokens = original_tokenize(replaced_text, *args, **kwargs)
            print(f"Used fallback tokenization for text containing special characters: {text}")
        
        return tokens
    
    # Override the encode method
    def enhanced_encode(text, *args, **kwargs):
        # First try with the original encoder
        try:
            return original_encode(text, *args, **kwargs)
        except Exception as e:
            # If encoding fails, try with character replacement
            replaced_text = replace_special_chars(text)
            return original_encode(replaced_text, *args, **kwargs)
    
    # Apply the enhanced methods
    tokenizer.tokenize = enhanced_tokenize
    tokenizer.encode = enhanced_encode
    
    return tokenizer

def replace_special_chars(text):
    """
    Replace special Yanomami characters with ASCII equivalents for fallback tokenization.
    
    Args:
        text: The text to process
        
    Returns:
        Text with special characters replaced
    """
    for char, replacement in CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    
    return text

def load_enhanced_tokenizer(model_name_or_path):
    """
    Load a tokenizer and enhance it with Yanomami special character handling.
    
    Args:
        model_name_or_path: The model name or path to load the tokenizer from
        
    Returns:
        An enhanced tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return enhance_tokenizer(tokenizer)

def test_tokenizer_enhancement(text_samples=None):
    """
    Test the tokenizer enhancement with sample Yanomami texts.
    
    Args:
        text_samples: Optional list of text samples to test
        
    Returns:
        None, prints results to console
    """
    if text_samples is None:
        text_samples = [
            "Ɨhɨ heri ka wakɨ",  # I walk on the path
            "pei yoka ha a ahetou tëhë a husi koikoimoma",  # he whistled when he was near the entrance
            "ɨpɨtɨ ɨhɨki kɨa",  # Sample with multiple special characters
            "Yanomami ɨhɨ rɨ kɨ",  # Another sample
        ]
    
    # Load and enhance tokenizer
    tokenizer = load_enhanced_tokenizer("gpt2")
    
    print("\nTesting tokenizer enhancement with Yanomami text samples:")
    print("-" * 60)
    
    for text in text_samples:
        print(f"\nOriginal text: {text}")
        
        # Tokenize and show tokens
        tokens = tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        
        # Encode and decode to verify roundtrip
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"Decoded: {decoded}")
        print(f"Roundtrip successful: {text == decoded.strip()}")

if __name__ == "__main__":
    # Run a test of the tokenizer enhancement
    test_tokenizer_enhancement()
