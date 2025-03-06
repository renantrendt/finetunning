import json
import os
import re
from collections import Counter
import glob

def extract_yanomami_words(file_path):
    """Extract all Yanomami words from a JSONL file."""
    yanomami_words = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'messages' in data:
                        for message in data['messages']:
                            if 'content' in message:
                                content = message['content']
                                
                                # Extract words from <WORD> tags
                                word_matches = re.findall(r'<WORD>([^<]+)</WORD>', content)
                                yanomami_words.extend(word_matches)
                                
                                # Extract phrases from <YANOMAMI> tags
                                yanomami_matches = re.findall(r'<YANOMAMI>([^<]+)</YANOMAMI>', content)
                                for match in yanomami_matches:
                                    # Split phrases into individual words
                                    words = re.findall(r'\w+', match)
                                    yanomami_words.extend(words)
                                    
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return yanomami_words

def main():
    # Find the dataset directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = None
    
    for root, dirs, files in os.walk(base_dir):
        if 'yanomami_dataset' in dirs:
            dataset_dir = os.path.join(root, 'yanomami_dataset')
            break
    
    if not dataset_dir:
        print("Could not find yanomami_dataset directory.")
        return
    
    # Find all JSONL files in the dataset directory (recursively)
    jsonl_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jsonl') and not file.startswith('combined-'):
                jsonl_files.append(os.path.join(root, file))
    
    if not jsonl_files:
        print("No JSONL files found in the dataset directory.")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process.")
    
    # Extract all Yanomami words from all files
    all_words = []
    for file_path in jsonl_files:
        print(f"Processing {file_path}...")
        words = extract_yanomami_words(file_path)
        all_words.extend(words)
        print(f"  Found {len(words)} Yanomami words")
    
    # Count word frequencies
    word_counter = Counter(all_words)
    
    # Save words sorted by frequency
    output_file = os.path.join(base_dir, 'yanomami_tokenizer', 'all_yanomami_words.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in word_counter.most_common():
            f.write(f"{word}\t{count}\n")
    
    # Print statistics
    print("\nWord Statistics:")
    print(f"Total words found: {len(all_words)}")
    print(f"Unique words: {len(word_counter)}")
    print(f"Words saved to: {output_file}")
    
    # Create a file with just the words (for tokenizer training)
    words_only_file = os.path.join(base_dir, 'yanomami_tokenizer', 'yanomami_words_for_tokenizer.txt')
    with open(words_only_file, 'w', encoding='utf-8') as f:
        for word, _ in word_counter.most_common():
            f.write(f"{word}\n")
    
    print(f"Words list for tokenizer saved to: {words_only_file}")
    
    # Create a script to update the tokenizer with all these words
    update_script = os.path.join(base_dir, 'scripts_for_tokenizer', 'update_tokenizer_with_all_words.py')
    with open(update_script, 'w', encoding='utf-8') as f:
        f.write("""from transformers import GPT2Tokenizer
import os

def main():
    # Load the base GPT-2 tokenizer
    base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"Loaded base GPT-2 tokenizer with {len(base_tokenizer)} tokens")
    
    # Load the Yanomami words
    words_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'yanomami_tokenizer', 'yanomami_words_for_tokenizer.txt')
    
    with open(words_file, 'r', encoding='utf-8') as f:
        yanomami_words = [line.strip() for line in f if line.strip()]
    
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
    
    # Then add all Yanomami words
    yanomami_added = 0
    for word in yanomami_words:
        # Skip empty words or words that are already in the vocabulary
        if not word or word in base_tokenizer.get_vocab():
            continue
            
        base_tokenizer.add_tokens([word])
        yanomami_added += 1
    
    print(f"Added {yanomami_added} Yanomami words")
    print(f"New tokenizer size: {len(base_tokenizer)}")
    
    # Save the updated tokenizer
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'yanomami_tokenizer', 'complete_yanomami_tokenizer')
    os.makedirs(output_dir, exist_ok=True)
    
    base_tokenizer.save_pretrained(output_dir)
    print(f"Saved complete Yanomami tokenizer to {output_dir}")
    
    # Reminder about model resizing
    print("\nRemember to resize the model's token embeddings when loading:")
    print('''from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('./yanomami_tokenizer/complete_yanomami_tokenizer')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))''')

if __name__ == "__main__":
    main()
""")
    
    print(f"\nCreated script to update tokenizer with all words: {update_script}")
    print("Run this script to create a complete Yanomami tokenizer with all words as single tokens.")

if __name__ == "__main__":
    main()
