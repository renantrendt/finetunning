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
├── train_yanomami_model.py        # Unified training script for local and cloud environments
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

### Training the Model

#### Local Training

To train the model on your local machine:

```bash
# Run the local training script
python train_yanomami_model.py

# Options for training
python train_yanomami_model.py --batch-size 8 --mixed-precision --epochs 10

# Resume training from a checkpoint
python train_yanomami_model.py --resume

# Enable debug mode for verbose output
python train_yanomami_model.py --debug
```

#### Lambda Cloud Training

To achieve faster training speeds and better performance, you can train the model on Lambda Cloud GPU instances. The training script automatically detects and configures itself for Lambda Cloud environments.

##### Prerequisites

1. **Lambda Cloud Account**:
   - Sign up at https://cloud.lambdalabs.com if you don't have an account
   - Create an API key at https://cloud.lambdalabs.com/api-keys
   - Set your API key as an environment variable:
     ```bash
     export LAMBDA_API_KEY=your_api_key_here
     ```

2. **Environment Setup**:

   a. **Required Environment Variables**:
   ```bash
   # Lambda Cloud API Key
   export LAMBDA_API_KEY=your_api_key_here

   # Training Configuration (Optional - will use defaults if not set)
   export YANOMAMI_BATCH_SIZE=32          # Default: Based on GPU
   export YANOMAMI_GRAD_ACCUM_STEPS=1     # Default: Based on GPU
   export YANOMAMI_MIXED_PRECISION=1       # Default: 1 (enabled)
   export YANOMAMI_RESUME_TRAINING=0       # Default: 0 (disabled)
   export YANOMAMI_DEBUG=0                 # Default: 0 (disabled)
   ```

   b. **Project Setup on Lambda Instance**:
   ```bash
   # Clone the repository
   git clone https://github.com/renantrendt/yanomami-finetuning.git
   cd yanomami-finetuning

   # Install dependencies
   pip install -r requirements.txt

   # Create required directories
   mkdir -p /yanomami_project/{yanomami_dataset,checkpoints,enhanced_yanomami_translator,logs,visualization_results}
   ```

   c. **Data Transfer to Lambda Instance**:
   ```bash
   # From your local machine, transfer dataset and any existing checkpoints
   rsync -avz ./yanomami_dataset/ ubuntu@<instance-ip>:/yanomami_project/yanomami_dataset/
   rsync -avz ./checkpoints/ ubuntu@<instance-ip>:/yanomami_project/checkpoints/  # If resuming training
   ```

##### Training Options

The script automatically detects the Lambda environment and GPU configuration, optimizing training parameters accordingly:

1. **8x A100 Configuration (Optimal Performance)**:
   ```bash
   python train_yanomami_model.py
   ```
   - GPU Type: 8x A100 (40GB SXM4)
   - Batch Size: 32
   - Gradient Accumulation: Disabled
   - Mixed Precision: Enabled
   - Distributed Training: Enabled
   - Training Epochs: 5
   - Region: us-east-1

2. **1x A100 Configuration (Fallback Option 1)**:
   ```bash
   python train_yanomami_model.py
   ```
   - GPU Type: 1x A100
   - Batch Size: 8
   - Gradient Accumulation: 4 steps
   - Mixed Precision: Enabled
   - Distributed Training: Disabled
   - Training Epochs: 8
   - Region: us-east-1

3. **MPS GPU Configuration (Fallback Option 2)**:
   ```bash
   python train_yanomami_model.py
   ```
   - GPU Type: 1x MPS
   - Batch Size: 4
   - Gradient Accumulation: 8 steps
   - Mixed Precision: Enabled
   - Distributed Training: Disabled
   - Training Epochs: 10
   - Region: us-east-1

The script uses Lambda's official PyTorch image (`lambdal/lambda-stack:latest`) and automatically configures all paths and settings. You don't need to manually specify training parameters as they are optimized based on the detected GPU configuration.

##### Monitoring Training

1. **View Live Training Progress**:
   ```bash
   # Attach to the training session
   tmux attach -t yanomami_training
   
   # To detach without stopping: press Ctrl+B, then D
   ```

2. **Check Training Logs**:
   ```bash
   # View latest log file
   tail -f /yanomami_project/logs/training.log
   ```

3. **Monitor GPU Usage**:
   ```bash
   nvidia-smi -l 1  # Updates every second
   ```

##### Directory Structure

The training script automatically manages the following directory structure on the Lambda instance:

```
/yanomami_project/
├── yanomami_dataset/        # Training dataset files
│   ├── combined-ok-translations.jsonl
│   ├── grammar-plural.jsonl
│   ├── grammar-verb.jsonl
│   └── combined-ok-phrases-*.jsonl
├── checkpoints/             # Training checkpoints and state
├── enhanced_yanomami_translator/  # Final model output
├── logs/                    # Training logs and metrics
└── visualization_results/   # Training visualizations
```

All paths are automatically configured when running on Lambda Cloud. The script detects the environment and adjusts paths accordingly - using `/yanomami_project` on Lambda instances and the local project directory when running locally.

##### Retrieving Results

After training completes, download your results:
```bash
# From your local machine
rsync -avz ubuntu@<instance-ip>:/yanomami_project/enhanced_yanomami_translator/ ./enhanced_yanomami_translator/
rsync -avz ubuntu@<instance-ip>:/yanomami_project/visualization_results/ ./visualization_results/
rsync -avz ubuntu@<instance-ip>:/yanomami_project/logs/ ./logs/
```

##### Cost Management and Instance Selection

1. **Instance Types and Costs**:
   - **8x A100 Configuration**:
     * Highest performance but most expensive
     * Best for large-scale training
     * Recommended for final model training
   
   - **1x A100 Configuration**:
     * Good balance of cost and performance
     * Suitable for most training needs
     * Recommended for development and testing
   
   - **MPS GPU Configuration**:
     * Most cost-effective option
     * Suitable for debugging and small experiments
     * Recommended for initial development

2. **Cost Optimization Tips**:
   - Start development with smaller instances
   - Use checkpointing to resume training if needed
   - Monitor training progress to avoid unnecessary runtime
   - Terminate instances when not in use
   - Use mixed precision to reduce training time

3. **Instance Management**:
   ```bash
   # List running instances and costs
   lambda instance list

   # Stop an instance (can be restarted)
   lambda instance stop <instance-id>

   # Terminate an instance (permanent)
   lambda instance terminate <instance-id>
   ```

##### Security Best Practices

1. **API Key Management**:
   - Store API keys in environment variables, never in code
   - Rotate API keys regularly
   - Use different API keys for development and production

2. **Access Control**:
   - Use SSH key-based authentication
   - Keep your SSH private key secure
   - Never share instance credentials

3. **Data Security**:
   - Encrypt sensitive data before transfer
   - Use secure channels (SSH/SFTP) for data transfer
   - Regularly backup important data
   - Clean up sensitive data when terminating instances

4. **Network Security**:
   - Only expose necessary ports
   - Use secure protocols for communication
   - Monitor instance access logs

5. **Instance Hygiene**:
   - Keep the system updated
   - Remove unused packages
   - Monitor system logs for suspicious activity
   - Terminate instances when not in use

##### Troubleshooting

1. **Out of Memory Errors**:
   - Reduce batch size
   - Enable mixed precision training
   - Increase gradient accumulation steps

2. **Training Not Progressing**:
   - Check logs in `/yanomami_project/logs/`
   - Enable debug mode: `python train_yanomami_model.py --debug`
   - Verify dataset paths

3. **GPU Not Detected**:
   - Run `nvidia-smi` to check GPU status
   - Verify CUDA installation
   - Check GPU drivers

4. **Lambda Cloud Issues**:
   - Check Lambda Cloud status page
   - Verify API key and permissions
   - Ensure sufficient account credits

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
