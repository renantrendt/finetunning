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
├── run_local_training.py          # Main script for local training
├── train_yanomami.py              # Simple training script
├── yanomami_trainer/              # Training module
│   ├── improvements_finetuning.py # Core training implementation
│   └── visualization_utils.py     # Training visualization utilities
├── yanomami_tokenizer/            # Tokenizer module with special character support
│   ├── tokenizer_enhancement.py   # Enhanced tokenizer for Yanomami characters
│   └── special_char_vocabulary.txt # Special character vocabulary
├── yanomami_dataset/              # Training data directory
│   ├── combined-ok-translations.jsonl  # Basic translations
│   ├── grammar-plural.jsonl       # Grammar examples for plurals
│   ├── grammar-verb.jsonl         # Grammar examples for verbs
│   ├── combined-ok-phrases-*.jsonl # Phrase examples
│   └── combined-ok-how-to-*.jsonl  # How-to examples
├── checkpoints/                   # Model checkpoints during training
├── enhanced_yanomami_translator/  # Final trained model output
├── logs/                          # Training logs
├── visualization_results/         # Training visualizations
└── deprecated_scripts/           # Deprecated Lambda Cloud scripts
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- Transformers 4.30 or higher
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
   tokenizer.save_pretrained("./enhanced_yanomami_translator")
   model.save_pretrained("./enhanced_yanomami_translator")
   ```

### Training the Model Locally

To train the model on your local machine:

```bash
# Run the local training script
python run_local_training.py

# Options for training
python run_local_training.py --batch-size 8 --mixed-precision --epochs 10

# Resume training from a checkpoint
python run_local_training.py --resume

# Enable debug mode for verbose output
python run_local_training.py --debug
```

The training process uses a multi-phase approach:

1. **Phase 1**: Basic vocabulary training using combined translations
2. **Phase 2**: Grammar and structure training focusing on plurals and verbs
3. **Phase 3**: Advanced phrases and usage examples

### Usage

#### Using the Trained Model

After training or downloading the model, you can use it for translation:

```python
from yanomami_trainer.improvements_finetuning import load_yanomami_translator, generate_translation

# Load the model
model_path = "./enhanced_yanomami_translator"
model, tokenizer = load_yanomami_translator(model_path)

# Translate from English to Yanomami
english_text = "Hello, how are you?"
yanomami_translation = generate_translation(english_text, model, tokenizer, None, "english_to_yanomami")
print(yanomami_translation)

# Translate from Yanomami to English
yanomami_text = "Aheprariyo"
english_translation = generate_translation(yanomami_text, model, tokenizer, None, "yanomami_to_english")
print(english_translation)
```

#### Testing Translations

You can test the model with sample translations:

```python
from yanomami_trainer.improvements_finetuning import load_yanomami_translator, test_translations

# Load the model
model_path = "./enhanced_yanomami_translator"
model, tokenizer = load_yanomami_translator(model_path)

# Run test translations
results = test_translations(model, tokenizer, None, save_results=True)
```

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
