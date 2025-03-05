from transformers import GPT2Tokenizer
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

print("Tokenization Examples:\n")

# Test the first 8 examples
for text in test_examples[:8]:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    print(f"Text: '{text}'\n")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(tokens)}\n")
    print("-" * 50 + "\n")

# Test specific phrases with ɨkɨ
print("\nTesting phrases with ɨkɨ:\n")
special_phrases = [
    "kamiyë ɨkɨ-",  # I want
    "ɨkɨ-",  # want
    "ɨkɨ mai",  # don't want
]

for text in special_phrases:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    print(f"Text: '{text}'\n")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(tokens)}\n")
    print("-" * 50 + "\n")
