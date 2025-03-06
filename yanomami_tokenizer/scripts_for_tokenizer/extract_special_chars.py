# Extract Special Characters Script
# This script analyzes the dataset to find and extract words containing the special character 'ɨ'
# and creates a vocabulary list for tokenizer enhancement

import os
import json
import glob
import re
from collections import Counter

# Import the load_jsonl function
from improvements_finetuning import load_jsonl

def extract_words_with_special_chars(text, special_chars=['ɨ']):
    """Extract all words containing the specified special characters."""
    # Create a regex pattern that matches words containing any of the special characters
    pattern = r'\b\w*[{}]\w*\b'.format(''.join(special_chars))
    words = re.findall(pattern, text, re.UNICODE)
    return words

def analyze_special_chars_in_dataset():
    """Analyze special characters in the dataset files."""
    dataset_path = '/Users/renanserrano/CascadeProjects/Yanomami/finetunning/yanomami_dataset/'
    dataset_files = glob.glob(os.path.join(dataset_path, '*.jsonl'))
    
    special_chars = ['ɨ']  # Add more special characters if needed
    word_counter = Counter()
    char_counter = Counter()
    
    print(f"Analyzing {len(dataset_files)} dataset files for words with special characters: {special_chars}")
    
    for file_path in dataset_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        try:
            examples = load_jsonl(file_path)
            
            for example in examples:
                if 'messages' in example:
                    for message in example['messages']:
                        content = message.get('content', '')
                        
                        # Count all occurrences of special characters
                        for char in special_chars:
                            char_counter[char] += content.count(char)
                        
                        # Extract and count words with special characters
                        words = extract_words_with_special_chars(content, special_chars)
                        for word in words:
                            word_counter[word] += 1
        
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return word_counter, char_counter

def main():
    word_counter, char_counter = analyze_special_chars_in_dataset()
    
    # Print character statistics
    print("\nSpecial Character Statistics:")
    for char, count in char_counter.items():
        print(f"Character '{char}' (U+{ord(char):04X}): {count} occurrences")
    
    # Print most common words with special characters
    print("\nMost Common Words with Special Characters:")
    for word, count in word_counter.most_common(50):
        print(f"{word}: {count} occurrences")
    
    # Save vocabulary list to a file
    vocab_file = '/Users/renanserrano/CascadeProjects/Yanomami/finetunning/special_char_vocabulary.txt'
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for word, count in word_counter.most_common():
            f.write(f"{word}\n")
    
    print(f"\nSaved {len(word_counter)} words to {vocab_file}")
    
    # Generate code snippet for tokenizer enhancement
    print("\nTokenizer Enhancement Code Snippet:")
    print("# Add these words to the tokenizer vocabulary")
    print("special_char_words = [")
    for word, _ in word_counter.most_common(30):
        print(f'    "{word}",')  
    print("]")
    print("")
    print("# Add special tokens to the tokenizer")
    print("for word in special_char_words:")
    print("    tokenizer.add_tokens(word)")
    print("")
    print("# Special character replacement function")
    print("def replace_special_chars(text):")
    print("    \"\"\"Replace special characters with ASCII equivalents for fallback tokenization.\"\"\"")
    print("    replacements = {")
    print("        '\\u0268': 'i',  # Replace \\u0268 with i")
    print("        # Add more replacements as needed")
    print("    }")
    print("    ")
    print("    for char, replacement in replacements.items():")
    print("        text = text.replace(char, replacement)")
    print("    ")
    print("    return text")

if __name__ == "__main__":
    main()
