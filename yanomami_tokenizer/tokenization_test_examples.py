from transformers import GPT2Tokenizer
import os

# Load the Yanomami-extended tokenizer
tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'complete_yanomami_tokenizer')
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# Test examples - words with diacritics and special characters
test_examples = [
    # Basic words with/without diacritics
    "napë",  # with diacritics
    "nape",  # without diacritics
    "hëtëmou",  # with multiple diacritics
    "hetemou",  # without diacritics
    
    # Common Yanomami words
    "thëpë",  # people, plural marker
    "thepe",  # without diacritics
    "yano",  # house
    "urihi",  # forest, land
    "wamotima",  # food
    "xapono",  # communal house
    "wãyã",  # spirit
    "waya",  # without diacritics
    
    # Words with special character ɨ
    "ɨhɨ",  # yes
    "ihi",  # ASCII version
    "Yanomamɨ",  # Yanomami with special char
    "Yanomami",  # ASCII version
    "kamɨ",  # us
    "kami",  # ASCII version
    "kɨrɨ",  # in "Tha kɨrɨ ohore" (The banana is moldy)
    "kiri",  # ASCII version
    "ɨpɨtɨawɨ",  # created
    "ipitawi",  # ASCII version
    "hɨɨkɨrɨ",  # great
    "hiikiri",  # ASCII version
    "ɨkɨɨ",  # lament
    "ikii",  # ASCII version
    
    # Common words
    "maa",  # no
    "ipa",  # my
    "aho",  # your
    "pei",  # his/her/its
    "yamakɨ",  # we
    "yamaki",  # without diacritics
    "kaho",  # you (singular)
    "kamiyë",  # I
    "kamiye",  # without diacritics
    
    # Words with XML tags
    "<WORD>napë</WORD>",  # with special tokens
    "<WORD>nape</WORD>",  # with special tokens, no diacritics
    "<POS>Noun</POS>",  # part of speech tag
    "<DEFINITION>foreigner</DEFINITION>",  # definition tag
    
    # Phrases
    "Yanomamɨ thëpë",  # phrase with special characters
    "Yanomami thepe",  # phrase without special characters
    "kami yamakɨ",  # we (emphatic)
    "kami yamaki",  # without diacritics
    "urihi a pata",  # the great forest
    "yaro pë wamuu",  # to eat animals
    "yaro pe wamuu",  # without diacritics
    
    # Complete examples with tags
    "<QUERY>What does <WORD>napë</WORD> mean in Yanomami?</QUERY>",
    "<WORD>napë</WORD> <POS>Noun</POS> <DEFINITION>foreigner, non-indigenous person</DEFINITION>",
    "<YANOMAMI>kami ya napë pë taɨ</YANOMAMI> <TRANSLATION>I see the foreigners</TRANSLATION>"
]

print("Tokenization Examples:\n")

for text in test_examples:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    print(f"Text: '{text}'\n")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(tokens)}\n")
    print("-" * 50 + "\n")
