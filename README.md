# Yanomami Language Translation System

![Yanomami Translation](https://img.shields.io/badge/Translation-Yanomami%20%E2%86%94%20English-brightgreen)
![Model](https://img.shields.io/badge/Model-GPT--2%20Small-blue)
![Status](https://img.shields.io/badge/Status-Active-success)

A comprehensive offline translation system for the Yanomami language using fine-tuned GPT-2 and Retrieval-Augmented Generation (RAG).

## 🔗 Related Resources

### Models & Datasets
- **Fine-tuned Model (Hugging Face)**: [renanserrano/yanomami-finetuning](https://huggingface.co/renanserrano/yanomami-finetuning)
- **Dataset (Hugging Face)**: [renanserrano/yanomami](https://huggingface.co/datasets/renanserrano/yanomami)

### Related Repositories
- **Dataset Generator (NPM Package)**: [ai-dataset-generator](https://www.npmjs.com/package/ai-dataset-generator)
- **Dataset Generator (GitHub)**: [renantrendt/ai-dataset-generator](https://github.com/renantrendt/ai-dataset-generator)

## 🌟 Features

- **Bidirectional Translation**: Translate between Yanomami and English
- **Offline Functionality**: Works completely offline for use in remote areas
- **Context-Enhanced Translations**: Uses RAG to provide comprehensive linguistic information
- **Command-Line Interface**: Easy-to-use CLI for quick translations
- **Customizable**: Adaptable for various use cases and deployment scenarios

## 📋 Project Structure

```
yanomami-finetuning/
├── gpt2_yanomami_finetuning.py    # Main training script
├── yanomami_rag_cli.py            # CLI interface for translation with RAG
├── consulta_modelo.py             # Simple model query script
├── upload_to_hf.py                # Script to upload model to Hugging Face
├── upload_config_files.py         # Script to upload config files to Hugging Face
├── yanomami_dataset/              # Training data directory
│   ├── translations.jsonl         # 17,009 examples
│   ├── yanomami-to-english.jsonl  # 1,822 examples
│   ├── phrases.jsonl              # 2,322 examples
│   ├── grammar.jsonl              # 200 examples
│   ├── comparison.jsonl           # 2,072 examples
│   └── how-to.jsonl               # 5,586 examples
└── gpt2_yanomami_translator/      # Trained model directory
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- PyTorch
- Transformers
- huggingface_hub (for model upload/download)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/renantrendt/yanomami-finetuning.git
   cd yanomami-finetuning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model (option 1 - from Hugging Face):
   ```python
   from transformers import GPT2LMHeadModel, GPT2Tokenizer
   
   model_name = "renanserrano/yanomami-finetuning"
   tokenizer = GPT2Tokenizer.from_pretrained(model_name)
   model = GPT2LMHeadModel.from_pretrained(model_name)
   
   # Save locally if needed
   tokenizer.save_pretrained("./gpt2_yanomami_translator")
   model.save_pretrained("./gpt2_yanomami_translator")
   ```

### Usage

#### Command-Line Interface

Run the CLI interface:

```bash
python yanomami_rag_cli.py
```

Follow the on-screen prompts to:
1. Translate from English to Yanomami
2. Translate from Yanomami to English
3. Get comprehensive information about a Yanomami word or phrase

#### Direct Translation in Python

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_yanomami_translator")
model = GPT2LMHeadModel.from_pretrained("./gpt2_yanomami_translator")

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
model.to(device)

# Function for translation
def translate(text, direction="english_to_yanomami"):
    # Add appropriate prefix based on translation direction
    if direction == "english_to_yanomami":
        prompt = f"English: {text} => Yanomami:"
    else:
        prompt = f"Yanomami: {text} => English:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate translation
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_beams=4,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the actual translation part (after the prompt)
    if "=>" in translation:
        translation = translation.split("=>")[1].strip()
    
    return translation

# Examples
# English to Yanomami
print(translate("What does 'aheprariyo' mean in Yanomami?", "english_to_yanomami"))

# Yanomami to English
print(translate("ahetoimi", "yanomami_to_english"))
```

## 📊 Model Performance

- **Final training loss**: 1.0554 (Epoch 3)
- **Final validation loss**: 1.0557
- **Overall average training loss**: 1.2102
- **Perplexity**: 2.87

## 🔄 Continuous Improvement

The model shows promising results for translating Yanomami words to English definitions but has limitations with more complex translations and conversational phrases. Ongoing work focuses on:

1. Expanding the training dataset
2. Improving translation accuracy for complex phrases
3. Enhancing the RAG system for better contextual understanding
4. Optimizing for deployment in resource-constrained environments

## 🌐 Resources

- [Hugging Face Model](https://huggingface.co/renanserrano/yanomami-finetuning)
- [GitHub Repository](https://github.com/renantrendt/yanomami-finetuning)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The Yanomami community for their language and cultural heritage
- Contributors to the dataset compilation and linguistic expertise
