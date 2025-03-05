import unicodedata
import json
import os
import re
import glob

def normalize_yanomami_text(text):
    """Remove diacritical marks and replace special Yanomami characters."""
    # First remove standard diacritics
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')
    
    # Then replace special Yanomami characters
    replacements = {
        '\u0268': 'i',  # Latin Small Letter I with Stroke → i
        '\u0197': 'I',  # Latin Capital Letter I with Stroke → I
        '\u00e3': 'a',  # a with tilde
        '\u1ebd': 'e',  # e with tilde
        '\u0129': 'i',  # i with tilde
        '\u00f5': 'o',  # o with tilde
        '\u0169': 'u',  # u with tilde
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text

def process_file(input_file, output_file):
    """Process a single file to create a version with and without diacritics."""
    try:
        # Load the original data
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = [json.loads(line) for line in f]
        
        print(f"Processing {input_file} with {len(original_data)} items")
        
        # Create versions without diacritics
        no_diacritics_data = []
        for item in original_data:
            # Create a copy of the item
            new_item = item.copy()
            
            # Process all text in the messages
            if 'messages' in new_item:
                for message in new_item['messages']:
                    if 'content' in message and message['role'] == 'user':
                        content = message['content']
                        
                        # Find all words inside tags
                        patterns = [
                            (r'<WORD>([^<]+)</WORD>', 'WORD'),
                            (r'<YANOMAMI>([^<]+)</YANOMAMI>', 'YANOMAMI')
                        ]
                        
                        for pattern, tag_name in patterns:
                            regex = re.compile(pattern)
                            matches = regex.findall(content)
                            
                            for match in matches:
                                normalized = normalize_yanomami_text(match)
                                if normalized != match:
                                    # Replace only the word, keeping the tags
                                    content = content.replace(f"<{tag_name}>{match}</{tag_name}>", 
                                                             f"<{tag_name}>{normalized}</{tag_name}>")
                        
                        message['content'] = content
            
            no_diacritics_data.append(new_item)
        
        # Combine both datasets
        combined_data = original_data + no_diacritics_data
        
        # Save the combined dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in combined_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  Original: {len(original_data)} items")
        print(f"  Combined: {len(combined_data)} items")
        print(f"  Saved to: {output_file}")
        
        return len(original_data), len(combined_data)
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return 0, 0

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
    
    # Process each file
    total_original = 0
    total_combined = 0
    
    for input_file in jsonl_files:
        dir_name = os.path.dirname(input_file)
        file_name = os.path.basename(input_file)
        output_file = os.path.join(dir_name, f"combined-{file_name}")
        
        # Skip if the output file already exists and is not the one we just created
        if os.path.exists(output_file) and not file_name.startswith('ok-translations'):
            print(f"Skipping {input_file} as {output_file} already exists.")
            continue
        
        orig_count, combined_count = process_file(input_file, output_file)
        total_original += orig_count
        total_combined += combined_count
    
    print("\nSummary:")
    print(f"Total original items: {total_original}")
    print(f"Total combined items: {total_combined}")
    print(f"Increase: {total_combined - total_original} items")

if __name__ == "__main__":
    main()
