# HellaSwag Evaluation for Yanomami Translation Model

This document explains how to use the HellaSwag evaluation module to benchmark your Yanomami translation model against standard language models like GPT-2 and GPT-3.

## Overview

The HellaSwag evaluation module provides a way to assess your model's general language understanding capabilities using the HellaSwag benchmark dataset. This evaluation generates both numerical metrics and visual charts to help you track your model's performance.

## Testing the HellaSwag Evaluation

You can test if the HellaSwag evaluation is working correctly using the provided test script:

```bash
python test_hellaswag.py --examples 10 --output ./hellaswag_test_results
```

Options:
- `--examples`: Number of examples to evaluate (default: 20)
- `--model`: Model name or path (default: "gpt2")
- `--output`: Output directory for results and charts (default: "./hellaswag_test_results")

## Visualization Features

The HellaSwag evaluation generates two types of charts:

1. **Comparison Charts**: Bar charts comparing your model's performance against baseline models on:
   - Accuracy metrics (higher is better)
   - Perplexity metrics (lower is better)

2. **Progress Tracking Charts**: Line charts tracking your model's performance across training phases and epochs, showing:
   - Accuracy trends over time
   - Perplexity improvements throughout training

## Integration with Training

The HellaSwag evaluation is automatically integrated into the training process in `improvements_finetuning.py`. It will run:

- At specific epochs during training (configurable via `config.hellaswag_eval_epochs`)
- At the end of the entire training process

## Configuration Options

In the `TranslatorConfig` class, you can configure the following HellaSwag evaluation settings:

```python
# HellaSwag evaluation settings
self.enable_hellaswag_eval = True  # Enable HellaSwag evaluation
self.hellaswag_eval_epochs = [1, 3, 5]  # Epochs to run HellaSwag evaluation
self.hellaswag_num_examples = 100  # Number of examples to evaluate
self.hellaswag_compare_baselines = True  # Compare with baseline models
```

## Output Files

The HellaSwag evaluation generates the following files in the `hellaswag_results` directory inside your model output directory:

- `hellaswag_comparison_yanomami_phase{phase}_epoch{epoch}_{timestamp}.png`: Comparison chart for each evaluation
- `hellaswag_history_yanomami_{timestamp}.png`: Progress tracking chart updated after each evaluation
- `hellaswag_results_yanomami_phase{phase}_epoch{epoch}_{timestamp}.json`: Detailed metrics for each evaluation
- `hellaswag_history_yanomami.json`: Comprehensive history of all evaluations for tracking progress

## Interpreting Results

The key metrics to focus on are:

1. **Accuracy**: The percentage of correct predictions on multiple-choice questions. Higher is better.
2. **Perplexity**: A measure of how well the model predicts the text. Lower is better.
3. **Relative Performance**: How your model compares to baselines (shown as percentage differences).

## Troubleshooting

If you encounter issues with the HellaSwag evaluation:

1. Check the log files for detailed error messages
2. Ensure you have the required dependencies installed
3. Verify that your model and tokenizer are compatible with the evaluation format

## Dependencies

The HellaSwag evaluation requires:
- transformers
- datasets
- matplotlib
- numpy
- torch
