#!/usr/bin/env python3
"""
Script to evaluate whether a custom tokenizer is needed for the Yanomami dataset.
This script compares the tokenization results of general-purpose tokenizers with
the specific needs of the Yanomami language dataset.
"""

import json
import re
import os
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple

# Try to import tiktoken, but don't fail if it's not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with 'pip install tiktoken'")

# Try to import transformers for comparison with other tokenizers
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with 'pip install transformers'")


def extract_yanomami_words(jsonl_file: str) -> List[str]:
    """Extract Yanomami words from the dataset."""
    words = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Extract words between <WORD> tags
                for message in data.get('messages', []):
                    content = message.get('content', '')
                    word_matches = re.findall(r'<WORD>([^<]+)</WORD>', content)
                    words.extend(word_matches)
                    
                    # Also extract Yanomami examples
                    example_matches = re.findall(r'<EXAMPLE_YANOMAMI>([^<]+)</EXAMPLE_YANOMAMI>', content)
                    for example in example_matches:
                        if example.strip():  # Skip empty examples
                            # Split by spaces to get individual words
                            example_words = re.findall(r'\b\w+\b', example)
                            words.extend(example_words)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON: {line[:50]}...")
    return words


def analyze_tiktoken(words: List[str], encoding_name: str = "cl100k_base") -> Dict[str, Any]:
    """Analyze how tiktoken handles the Yanomami words."""
    if not TIKTOKEN_AVAILABLE:
        return {"error": "tiktoken not available"}
    
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        
        total_words = len(words)
        total_tokens = 0
        words_split_into_multiple = 0
        token_counts = Counter()
        
        for word in words:
            tokens = encoding.encode(word)
            total_tokens += len(tokens)
            if len(tokens) > 1:
                words_split_into_multiple += 1
            
            # Record the token IDs for analysis
            for token in tokens:
                token_counts[token] += 1
        
        return {
            "total_words": total_words,
            "total_tokens": total_tokens,
            "avg_tokens_per_word": total_tokens / total_words if total_words else 0,
            "words_split_into_multiple": words_split_into_multiple,
            "percent_split": (words_split_into_multiple / total_words * 100) if total_words else 0,
            "unique_tokens_used": len(token_counts),
            "most_common_tokens": token_counts.most_common(10)
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_transformers_tokenizer(words: List[str], model_name: str = "gpt2") -> Dict[str, Any]:
    """Analyze how a HuggingFace tokenizer handles the Yanomami words."""
    if not TRANSFORMERS_AVAILABLE:
        return {"error": "transformers not available"}
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        total_words = len(words)
        total_tokens = 0
        words_split_into_multiple = 0
        unknown_tokens = 0
        token_counts = Counter()
        
        for word in words:
            tokens = tokenizer.tokenize(word)
            total_tokens += len(tokens)
            if len(tokens) > 1:
                words_split_into_multiple += 1
            
            # Check for unknown tokens
            for token in tokens:
                if token in [tokenizer.unk_token, '<unk>', '[UNK]'] or token.startswith('##'):
                    unknown_tokens += 1
                token_counts[token] += 1
        
        return {
            "total_words": total_words,
            "total_tokens": total_tokens,
            "avg_tokens_per_word": total_tokens / total_words if total_words else 0,
            "words_split_into_multiple": words_split_into_multiple,
            "percent_split": (words_split_into_multiple / total_words * 100) if total_words else 0,
            "unknown_tokens": unknown_tokens,
            "percent_unknown": (unknown_tokens / total_tokens * 100) if total_tokens else 0,
            "unique_tokens_used": len(token_counts),
            "most_common_tokens": list(token_counts.most_common(10))
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_special_tokens(jsonl_file: str) -> Dict[str, int]:
    """Analyze the special tokens used in the dataset."""
    special_tokens = Counter()
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                for message in data.get('messages', []):
                    content = message.get('content', '')
                    # Find all XML-like tags
                    tags = re.findall(r'<([A-Z_]+)>', content)
                    for tag in tags:
                        special_tokens[f"<{tag}>"] += 1
                        special_tokens[f"</{tag}>"] += 1
            except json.JSONDecodeError:
                continue
    return dict(special_tokens)


def evaluate_tokenizer_need(jsonl_file: str) -> Dict[str, Any]:
    """Evaluate whether a custom tokenizer is needed for the dataset."""
    words = extract_yanomami_words(jsonl_file)
    
    results = {
        "dataset": jsonl_file,
        "unique_yanomami_words": len(set(words)),
        "total_yanomami_words": len(words),
        "special_tokens": analyze_special_tokens(jsonl_file),
    }
    
    # Analyze with tiktoken if available
    if TIKTOKEN_AVAILABLE:
        results["tiktoken_analysis"] = analyze_tiktoken(words)
    
    # Analyze with transformers tokenizers if available
    if TRANSFORMERS_AVAILABLE:
        tokenizer_models = ["gpt2", "bert-base-multilingual-cased"]
        results["transformers_analysis"] = {}
        for model in tokenizer_models:
            try:
                results["transformers_analysis"][model] = analyze_transformers_tokenizer(words, model)
            except Exception as e:
                results["transformers_analysis"][model] = {"error": str(e)}
    
    # Make recommendation
    need_custom_tokenizer = False
    reasons = []
    
    # Check tiktoken results
    if TIKTOKEN_AVAILABLE and "error" not in results.get("tiktoken_analysis", {}):
        tiktoken_analysis = results["tiktoken_analysis"]
        if tiktoken_analysis.get("percent_split", 0) > 50:
            need_custom_tokenizer = True
            reasons.append(f"High word splitting rate with tiktoken: {tiktoken_analysis.get('percent_split', 0):.1f}%")
    
    # Check transformers results
    if TRANSFORMERS_AVAILABLE:
        for model, analysis in results.get("transformers_analysis", {}).items():
            if "error" not in analysis:
                if analysis.get("percent_split", 0) > 50:
                    need_custom_tokenizer = True
                    reasons.append(f"High word splitting rate with {model}: {analysis.get('percent_split', 0):.1f}%")
                if analysis.get("percent_unknown", 0) > 10:
                    need_custom_tokenizer = True
                    reasons.append(f"High unknown token rate with {model}: {analysis.get('percent_unknown', 0):.1f}%")
    
    # Check special tokens
    if len(results.get("special_tokens", {})) > 5:
        need_custom_tokenizer = True
        reasons.append(f"Dataset uses {len(results.get('special_tokens', {}))} special tokens that should be handled specifically")
    
    results["recommendation"] = {
        "need_custom_tokenizer": need_custom_tokenizer,
        "reasons": reasons
    }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print the evaluation results in a readable format."""
    print("\n" + "=" * 80)
    print(f"TOKENIZER EVALUATION FOR: {results['dataset']}")
    print("=" * 80)
    
    print(f"\nDATASET STATISTICS:")
    print(f"  - Unique Yanomami words: {results['unique_yanomami_words']}")
    print(f"  - Total Yanomami words: {results['total_yanomami_words']}")
    
    print(f"\nSPECIAL TOKENS:")
    for token, count in results.get('special_tokens', {}).items():
        print(f"  - {token}: {count}")
    
    if 'tiktoken_analysis' in results and 'error' not in results['tiktoken_analysis']:
        print(f"\nTIKTOKEN ANALYSIS:")
        analysis = results['tiktoken_analysis']
        print(f"  - Average tokens per word: {analysis['avg_tokens_per_word']:.2f}")
        print(f"  - Words split into multiple tokens: {analysis['words_split_into_multiple']} ({analysis['percent_split']:.1f}%)")
        print(f"  - Unique tokens used: {analysis['unique_tokens_used']}")
    
    if 'transformers_analysis' in results:
        for model, analysis in results['transformers_analysis'].items():
            if 'error' not in analysis:
                print(f"\n{model.upper()} TOKENIZER ANALYSIS:")
                print(f"  - Average tokens per word: {analysis['avg_tokens_per_word']:.2f}")
                print(f"  - Words split into multiple tokens: {analysis['words_split_into_multiple']} ({analysis['percent_split']:.1f}%)")
                print(f"  - Unknown tokens: {analysis.get('unknown_tokens', 'N/A')} ({analysis.get('percent_unknown', 'N/A'):.1f}%)")
                print(f"  - Unique tokens used: {analysis['unique_tokens_used']}")
    
    print("\nRECOMMENDATION:")
    if results["recommendation"]["need_custom_tokenizer"]:
        print("  ✓ A CUSTOM TOKENIZER IS RECOMMENDED for this dataset")
        print("\nREASONS:")
        for reason in results["recommendation"]["reasons"]:
            print(f"  - {reason}")
    else:
        print("  ✗ A custom tokenizer is NOT necessary for this dataset")
        print("  - Existing tokenizers handle the vocabulary adequately")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate whether a custom tokenizer is needed for a dataset')
    parser.add_argument('jsonl_file', help='Path to the JSONL dataset file')
    parser.add_argument('--output', help='Path to save the JSON results (optional)')
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_file):
        print(f"Error: File {args.jsonl_file} does not exist")
        return
    
    results = evaluate_tokenizer_need(args.jsonl_file)
    print_results(results)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
