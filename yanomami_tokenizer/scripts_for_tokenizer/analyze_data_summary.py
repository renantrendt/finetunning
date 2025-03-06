# Data Processing Summary Script
# This script analyzes how the prepare_data_for_training function processes different dataset files
# and provides a summary of what gets processed vs. discarded for each file.

import os
import json
import glob
from collections import Counter

# Import the prepare_data_for_training function from your main script
from improvements_finetuning import prepare_data_for_training, load_jsonl

def analyze_dataset_file(file_path):
    """Analyze a dataset file and summarize processing results."""
    filename = os.path.basename(file_path)
    
    # Load the raw data
    try:
        raw_examples = load_jsonl(file_path)
        total_raw = len(raw_examples)
        
        # Process the data
        processed_examples = prepare_data_for_training(raw_examples)
        total_processed = len(processed_examples)
        discarded = total_raw - total_processed
        
        # Analyze translation directions
        directions = Counter()
        for ex in processed_examples:
            if "English:" in ex["input"] and "=> Yanomami:" in ex["input"]:
                directions["English-to-Yanomami"] += 1
            elif "Yanomami:" in ex["input"] and "=> English:" in ex["input"]:
                directions["Yanomami-to-English"] += 1
            else:
                directions["Unknown"] += 1
        
        # Show one example if available
        example_raw = raw_examples[0] if raw_examples else None
        example_processed = processed_examples[0] if processed_examples else None
        
        return {
            "filename": filename,
            "total_raw": total_raw,
            "total_processed": total_processed,
            "discarded": discarded,
            "discard_rate": (discarded / total_raw * 100) if total_raw > 0 else 0,
            "directions": directions,
            "example_raw": example_raw,
            "example_processed": example_processed
        }
            
    except Exception as e:
        print(f"Error analyzing file {file_path}: {str(e)}")
        return None

def main():
    # Get all dataset files
    dataset_path = '/Users/renanserrano/CascadeProjects/Yanomami/finetunning/yanomami_dataset/'
    dataset_files = glob.glob(os.path.join(dataset_path, '*.jsonl'))
    
    print(f"Found {len(dataset_files)} dataset files\n")
    
    # Analyze each file
    results = []
    for file_path in dataset_files:
        result = analyze_dataset_file(file_path)
        if result:
            results.append(result)
    
    # Print summary table
    print(f"{'File':<40} | {'Raw':<6} | {'Processed':<9} | {'Discarded':<9} | {'Discard %':<9} | {'Direction':<20}")
    print(f"{'-'*40}-+-{'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*20}")
    
    for result in results:
        # Get main direction
        main_direction = "Mixed"
        if result["directions"]:
            main_direction = max(result["directions"].items(), key=lambda x: x[1])[0]
        
        print(f"{result['filename']:<40} | {result['total_raw']:<6} | {result['total_processed']:<9} | {result['discarded']:<9} | {result['discard_rate']:<8.1f}% | {main_direction:<20}")
    
    # Print overall statistics
    total_raw = sum(r["total_raw"] for r in results)
    total_processed = sum(r["total_processed"] for r in results)
    total_discarded = sum(r["discarded"] for r in results)
    overall_discard_rate = (total_discarded / total_raw * 100) if total_raw > 0 else 0
    
    print(f"\n{'-'*100}")
    print(f"{'TOTAL':<40} | {total_raw:<6} | {total_processed:<9} | {total_discarded:<9} | {overall_discard_rate:<8.1f}% |")
    
    # Print example of raw vs processed for one file
    if results:
        print(f"\n{'='*100}")
        print(f"EXAMPLE FROM {results[0]['filename']}:")
        print(f"\nRAW DATA:")
        print(json.dumps(results[0]["example_raw"], indent=2, ensure_ascii=False)[:500] + "..." if len(json.dumps(results[0]["example_raw"], indent=2, ensure_ascii=False)) > 500 else json.dumps(results[0]["example_raw"], indent=2, ensure_ascii=False))
        print(f"\nPROCESSED DATA:")
        print(json.dumps(results[0]["example_processed"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
