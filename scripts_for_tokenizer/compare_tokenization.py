from transformers import GPT2Tokenizer
import os
import json

def print_tokenization_comparison(text, standard_tokenizer, yanomami_tokenizer):
    """Compare tokenization between standard GPT-2 and Yanomami-extended tokenizer."""
    # Tokenize with standard GPT-2
    standard_tokens = standard_tokenizer.tokenize(text)
    standard_ids = standard_tokenizer.encode(text)
    
    # Tokenize with Yanomami-extended tokenizer
    yanomami_tokens = yanomami_tokenizer.tokenize(text)
    yanomami_ids = yanomami_tokenizer.encode(text)
    
    # Print comparison
    print(f"\nText: '{text}'")
    print(f"\nStandard GPT-2 Tokenization ({len(standard_tokens)} tokens):")
    print(f"Tokens: {standard_tokens}")
    print(f"IDs: {standard_ids}")
    print(f"Token count: {len(standard_ids)}")
    
    print(f"\nYanomami-extended Tokenization ({len(yanomami_tokens)} tokens):")
    print(f"Tokens: {yanomami_tokens}")
    print(f"IDs: {yanomami_ids}")
    print(f"Token count: {len(yanomami_ids)}")
    
    # Calculate token reduction
    reduction = len(standard_ids) - len(yanomami_ids)
    reduction_percent = (reduction / len(standard_ids)) * 100 if len(standard_ids) > 0 else 0
    
    print(f"\nToken reduction: {reduction} tokens ({reduction_percent:.1f}%)")
    print("-" * 80)

def main():
    # Load tokenizers
    try:
        standard_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        yanomami_tokenizer = GPT2Tokenizer.from_pretrained('./yanomami_tokenizer/yanomami_gpt2_tokenizer')
        
        print("Tokenizers loaded successfully.")
        print(f"Standard GPT-2 vocabulary size: {len(standard_tokenizer)}")
        print(f"Yanomami-extended vocabulary size: {len(yanomami_tokenizer)}")
        print(f"Added {len(yanomami_tokenizer) - len(standard_tokenizer)} new tokens")
        
        # Example Yanomami words and phrases to test
        examples = [
            # Common Yanomami words with diacritics
            "pë",
            "napë", 
            "hëtëmou",
            "Yanomamɨ",
            "ãiãmoprou",
            
            # XML tags
            "<WORD>napë</WORD>",
            "<POS>Noun</POS>",
            "<DEFINITION>foreigner</DEFINITION>",
            
            # Longer phrases
            "thë aheai",
            "ãiãmori pë sherekapi si roo yaiokiri",
            
            # Complete examples
            "<QUERY>What does <WORD>napë</WORD> mean in Yanomami?</QUERY>",
            "<WORD>napë</WORD> <POS>Noun</POS> <DEFINITION>foreigner, non-indigenous person</DEFINITION>"
        ]
        
        # Compare tokenization for each example
        for example in examples:
            print_tokenization_comparison(example, standard_tokenizer, yanomami_tokenizer)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
