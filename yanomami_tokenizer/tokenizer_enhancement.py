# Tokenizer Enhancement Module
# This module provides functions to enhance the tokenizer's handling of special Yanomami characters,
# particularly the character 'ɨ' (U+0268)

import re
import os
import unicodedata
from transformers import AutoTokenizer

# List of common Yanomami words containing special characters (ɨ, ë, ã, ũ)
# These words will be added to the tokenizer vocabulary to ensure they are tokenized as single tokens
SPECIAL_CHAR_WORDS = [
    # Common grammatical examples and test words
    "grano", "granoyë", "granopë", "granopɨ", "granopɨyë", "granopɨpë",
    "aheprariyo", "yanomami", "yanomamɨ", "thëpë", "totihi", "heri", "weti",
    "tha", "kami", "yai", "huë", "hãrãrema", "thë", "aheai", "wak", "ka",
    
    # Words with ɨ
    "kɨ", "Ɨhɨki", "Ɨhɨ", "kɨa", "pɨ", "ɨpɨtɨ", "ɨ", "rɨ", "kɨrɨ", "hɨ",
    "Ɨpɨtɨ", "wakɨ", "pɨtaha", "pɨtɨ", "ɨpɨ", "ɨhɨ", "ɨhɨki", "ɨwɨ",
    "hɨɨkɨrɨ", "hɨɨkɨ", "kɨri", "ɨpɨhɨ", "ɨpɨtɨwɨ", "sɨ", "yɨwɨ",
    "Wakɨ", "ɨpɨtɨhɨ", "pɨɨ", "ɨkɨɨ", "ɨkɨma", "ɨkɨmai", "ɨkɨmarei",
    "ɨkɨoprou", "ɨkɨrayou", "ɨkɨhiwë", "ɨkɨ", "pehihiprɨ", "ɨpɨtɨawɨ",
    "kamɨ", "hɨɨrɨ", "Kɨ", "ãpɨtɨ", "Yiiyɨ", "bɨtɨ", "ɨhurëpɨ",
    "Ũrihipɨ", "ɨtɨ", "hɨkɨri", "hɨtɨmɨ", "ãhiãmɨ",
    
    # Words with ë
    "ë", "yë", "hëri", "hëa", "hëtɨ", "hëmɨ", "hëyë", "hëtëhë",
    "ɨkɨhiwë", "ɨhurëpɨ", "iyë", "yëtɨ", "yëpɨ", "yëhë", "yëtëhë", "yëyë",
    "yëkɨ", "yëpë", "yëpɨa", "yëpɨrɨ", "yëpɨtɨ", "yëpɨwɨ", "yëpɨyë",
    
    # Words with ã
    "ã", "ãpɨtɨ", "ãhiãmɨ", "ãhã", "ãhɨ", "ãhɨã", "ãhɨãmɨ", "ãhɨãpɨ",
    "ãhɨãwɨ", "ãhɨãyë", "ãhɨãyɨ", "ãhɨãyɨpɨ", "ãhɨãyɨwɨ", "ãhɨãyɨyë",
    
    # Words with ũ
    "ũ", "Ũrihipɨ", "ũhũ", "ũhũã", "ũhũãmɨ", "ũhũãpɨ", "ũhũãwɨ", "ũhũãyë",
    "ũhũãyɨ", "ũhũãyɨpɨ", "ũhũãyɨwɨ", "ũhũãyɨyë"
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
    
    # Add common prefixes and suffixes for grammatical forms
    grammatical_forms = []
    
    # Add common plural forms and other grammatical variations
    for base_word in ['grano', 'yanomami', 'aheprariyo', 'heri', 'wak', 'tha', 'kami']:
        # Add plural forms with special characters
        grammatical_forms.extend([
            f"{base_word}pë",  # Common plural suffix with ë
            f"{base_word}pɨ",  # Common plural suffix with ɨ
            f"{base_word}yë",  # Common suffix with ë
            f"{base_word}yɨ",  # Common suffix with ɨ
            f"{base_word}thë",  # Common suffix with ë
        ])
    
    # Add these grammatical forms to the tokenizer
    num_added_grammar = tokenizer.add_tokens(grammatical_forms)
    print(f"Added {num_added_grammar} grammatical forms to tokenizer vocabulary")
    
    # Store the original tokenizer methods
    original_tokenize = tokenizer.tokenize
    original_encode = tokenizer.encode
    
    # Override the tokenize method to handle special characters
    def enhanced_tokenize(text, *args, **kwargs):
        try:
            # First, normalize the Unicode characters
            try:
                normalized_text = unicodedata.normalize('NFC', text)
            except Exception as e:
                print(f"Warning: Unicode normalization failed: {str(e)}. Using original text.")
                normalized_text = text
            
            # Check if text contains any special characters
            special_char_pattern = re.compile(f"[{''.join(CHAR_REPLACEMENTS.keys())}]")
            
            # If text contains special characters, handle them specially
            if special_char_pattern.search(normalized_text):
                print(f"Processing text with Yanomami special characters: {normalized_text[:50]}...")
                
                # Extract all words with special characters from the text
                words_with_special_chars = set()
                for word in re.findall(r'\w+', normalized_text):
                    if special_char_pattern.search(word):
                        words_with_special_chars.add(word)
                        # Also add versions with common prefixes/suffixes
                        words_with_special_chars.add(f"{word}pë")  # Common plural form
                        words_with_special_chars.add(f"{word}yë")  # Common suffix
                
                # If we found any words with special characters, add them to the tokenizer
                if words_with_special_chars:
                    num_added = tokenizer.add_tokens(list(words_with_special_chars))
                    if num_added > 0:
                        print(f"Added {num_added} new words with special characters to tokenizer vocabulary")
                        
                # For words that might still cause issues, create a fallback version
                # with special characters replaced
                fallback_text = replace_special_chars(normalized_text)
                if fallback_text != normalized_text:
                    # Try tokenizing both versions and use the one with fewer tokens
                    original_tokens = original_tokenize(normalized_text, *args, **kwargs)
                    fallback_tokens = original_tokenize(fallback_text, *args, **kwargs)
                    
                    if len(fallback_tokens) < len(original_tokens):
                        print(f"Using fallback tokenization for better handling of special characters")
                        return fallback_tokens
            
            # Use the original tokenizer with normalized text
            tokens = original_tokenize(normalized_text, *args, **kwargs)
            return tokens
        except Exception as e:
            print(f"Error in enhanced tokenization: {str(e)}. Falling back to original method.")
            # If anything fails, fall back to the original tokenizer with the original text
            return original_tokenize(text, *args, **kwargs)
    
    # Override the encode method
    def enhanced_encode(text, *args, **kwargs):
        # Check if text contains any special characters
        special_char_pattern = re.compile(f"[{''.join(CHAR_REPLACEMENTS.keys())}]")
        
        # If text contains special characters, log a warning but continue
        if special_char_pattern.search(text):
            print(f"Warning: Text contains Yanomami special characters in encode method: {text[:50]}...")
            
            # Extract all words with special characters from the text
            words_with_special_chars = set()
            for word in re.findall(r'\w+', text):
                if special_char_pattern.search(word):
                    words_with_special_chars.add(word)
            
            # If we found any words with special characters, add them to the tokenizer
            if words_with_special_chars:
                num_added = tokenizer.add_tokens(list(words_with_special_chars))
                if num_added > 0:
                    print(f"Added {num_added} new words with special characters to tokenizer vocabulary")
        
        # Use the original encoder, but catch exceptions
        try:
            return original_encode(text, *args, **kwargs)
        except Exception as e:
            print(f"Warning: Encoding failed for text: {text[:50]}... Error: {str(e)}")
            # Use a fallback approach - replace special characters with ASCII equivalents
            fallback_text = replace_special_chars(text)
            print(f"Using fallback encoding with replaced special characters")
            return original_encode(fallback_text, *args, **kwargs)
    
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
