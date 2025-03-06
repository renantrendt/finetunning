# Data Processing Analysis Script
# This script analyzes how the prepare_data_for_training function processes different dataset files
# and shows examples of both raw and processed data, including what gets discarded.

import os
import json
import glob
from collections import Counter

# Import the prepare_data_for_training function from your main script
from improvements_finetuning import prepare_data_for_training, load_jsonl

def analyze_dataset_file(file_path, num_examples=5):
    """Analyze a dataset file and show examples before and after processing."""
    print(f"\n{'='*80}\nAnalyzing file: {os.path.basename(file_path)}\n{'='*80}")
    
    # Load the raw data
    try:
        raw_examples = load_jsonl(file_path)
        print(f"Total examples in file: {len(raw_examples)}")
        
        # Process the data
        processed_examples = prepare_data_for_training(raw_examples)
        print(f"Examples after processing: {len(processed_examples)}")
        print(f"Discarded examples: {len(raw_examples) - len(processed_examples)}")
        
        # Show example pairs (before and after)
        print(f"\n{'-'*40}\nSample Examples (Before and After Processing)\n{'-'*40}")
        for i, raw_example in enumerate(raw_examples[:num_examples]):
            print(f"\nExample {i+1}:")
            print("\nRAW DATA:")
            print(json.dumps(raw_example, indent=2, ensure_ascii=False))
            
            # Find if this example was processed
            processed = None
            if i < len(processed_examples):
                processed = processed_examples[i]
                print("\nPROCESSED DATA:")
                print(json.dumps(processed, indent=2, ensure_ascii=False))
            else:
                print("\nDISCARDED - Not processed")
        
        # Analyze translation directions
        directions = Counter()
        for ex in processed_examples:
            if "English:" in ex["input"] and "=> Yanomami:" in ex["input"]:
                directions["English-to-Yanomami"] += 1
            elif "Yanomami:" in ex["input"] and "=> English:" in ex["input"]:
                directions["Yanomami-to-English"] += 1
            else:
                directions["Unknown"] += 1
        
        print(f"\n{'-'*40}\nTranslation Directions\n{'-'*40}")
        for direction, count in directions.items():
            print(f"{direction}: {count} examples ({count/len(processed_examples)*100:.1f}%)")
            
    except Exception as e:
        print(f"Error analyzing file {file_path}: {str(e)}")

def main():
    # Get all dataset files
    dataset_path = '/Users/renanserrano/CascadeProjects/Yanomami/finetunning/yanomami_dataset/'
    dataset_files = glob.glob(os.path.join(dataset_path, '*.jsonl'))
    
    print(f"Found {len(dataset_files)} dataset files")
    
    # Analyze each file
    for file_path in dataset_files:
        analyze_dataset_file(file_path)

if __name__ == "__main__":
    main()
