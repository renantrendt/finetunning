from transformers import GPT2Tokenizer
import os
import unicodedata
import re

def remove_diacritics(text):
    """Remove diacritical marks from text while preserving base characters."""
    # Normalize to decomposed form (separate base characters from diacritics)
    normalized = unicodedata.normalize('NFD', text)
    # Remove diacritical marks
    result = ''.join(c for c in normalized if not unicodedata.combining(c))
    # Normalize back to composed form
    return unicodedata.normalize('NFC', result)

def replace_special_chars(text):
    """Replace special characters with their closest ASCII equivalents."""
    # Map of special characters to their ASCII equivalents
    char_map = {
        '\u0268': 'i',  # Latin small letter i with stroke (U+0268)
    }
    
    # Replace each special character
    for special_char, ascii_char in char_map.items():
        text = text.replace(special_char, ascii_char)
    
    return text

def main():
    # Load the base GPT-2 tokenizer
    base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"Loaded base GPT-2 tokenizer with {len(base_tokenizer)} tokens")
    
    # Load the Yanomami words
    words_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'yanomami_tokenizer', 'yanomami_words_for_tokenizer.txt')
    
    with open(words_file, 'r', encoding='utf-8') as f:
        yanomami_words = [line.strip() for line in f if line.strip()]
    
    # Load additional words with special characters
    special_chars_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'yanomami_tokenizer', 'words_with_special_chars.txt')
    
    if os.path.exists(special_chars_file):
        with open(special_chars_file, 'r', encoding='utf-8') as f:
            special_words = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Loaded {len(special_words)} additional words with special characters")
        yanomami_words.extend(special_words)
    
    print(f"Loaded {len(yanomami_words)} Yanomami words to add")
    
    # Add special tokens first
    special_tokens = [
        '<WORD>', '</WORD>',
        '<POS>', '</POS>',
        '<DEFINITION>', '</DEFINITION>',
        '<QUERY>', '</QUERY>',
        '<EXAMPLES>', '</EXAMPLES>',
        '<EXAMPLE_YANOMAMI>', '</EXAMPLE_YANOMAMI>',
        '<EXAMPLE_TRANSLATION>', '</EXAMPLE_TRANSLATION>',
        '<RELATED_FORMS>', '</RELATED_FORMS>',
        '<YANOMAMI>', '</YANOMAMI>',
        '<TRANSLATION>', '</TRANSLATION>',
        '<LITERAL>', '</LITERAL>',
        '<USAGE>', '</USAGE>'
    ]
    
    # Create a new tokenizer with all Yanomami words
    num_added = 0
    
    # Add special tokens first
    for token in special_tokens:
        if token not in base_tokenizer.get_vocab():
            base_tokenizer.add_tokens([token])
            num_added += 1
    
    print(f"Added {num_added} special tokens")
    
    # Prepare to add both versions of each word
    all_word_forms = set()
    
    # Process each word to get all versions (original, without diacritics, with special chars replaced)
    for word in yanomami_words:
        if not word:
            continue
            
        # Add the original word (with diacritics and special chars)
        all_word_forms.add(word)
        
        # Add the version without diacritics if it's different
        normalized_word = remove_diacritics(word)
        if normalized_word != word:
            all_word_forms.add(normalized_word)
        
        # Add version with special characters replaced
        special_replaced = replace_special_chars(word)
        if special_replaced != word:
            all_word_forms.add(special_replaced)
            
        # Add version with both diacritics removed and special chars replaced
        normalized_special = replace_special_chars(normalized_word)
        if normalized_special != normalized_word and normalized_special != special_replaced:
            all_word_forms.add(normalized_special)
    
    # Then add all Yanomami words (both forms) in batches for better performance
    yanomami_added = 0
    words_to_add = []
    
    for word in all_word_forms:
        # Skip words that are already in the vocabulary
        if word in base_tokenizer.get_vocab():
            continue
        words_to_add.append(word)
    
    # Add words in batches of 1000
    batch_size = 1000
    for i in range(0, len(words_to_add), batch_size):
        batch = words_to_add[i:i+batch_size]
        base_tokenizer.add_tokens(batch)
        yanomami_added += len(batch)
        print(f"Added batch {i//batch_size + 1}/{(len(words_to_add) + batch_size - 1)//batch_size}, total so far: {yanomami_added}")
    
    print(f"Added {yanomami_added} Yanomami word forms (with and without diacritics)")
    print(f"New tokenizer size: {len(base_tokenizer)}")
    
    # Save the updated tokenizer
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'yanomami_tokenizer', 'complete_yanomami_tokenizer')
    os.makedirs(output_dir, exist_ok=True)
    
    base_tokenizer.save_pretrained(output_dir)
    print(f"Saved complete Yanomami tokenizer to {output_dir}")
    
    # Create a test file to show examples of tokenization
    test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'yanomami_tokenizer', 'tokenization_test_examples.py')
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('''from transformers import GPT2Tokenizer
import os

# Load the Yanomami-extended tokenizer
tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'complete_yanomami_tokenizer')
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# Test examples - words with diacritics
test_examples = [
    "napë",  # with diacritics
    "nape",  # without diacritics
    "hëtëmou",  # with multiple diacritics
    "hetemou",  # without diacritics
    "<WORD>napë</WORD>",  # with special tokens
    "<WORD>nape</WORD>",  # with special tokens, no diacritics
    "Yanomamɨ thëpë",  # phrase with special characters
    "Yanomami thepe"  # phrase without special characters
]

print("Tokenization Examples:\\n")

for text in test_examples:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    print(f"Text: '{text}'\\n")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(tokens)}\\n")
    print("-" * 50 + "\\n")
''')
    
    print(f"\nCreated test script at {test_file}")
    
    # Reminder about model resizing
    print("\nRemember to resize the model's token embeddings when loading:")
    print('''from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('./yanomami_tokenizer/complete_yanomami_tokenizer')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))''')

if __name__ == "__main__":
    main()
