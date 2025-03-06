# HellaSwag Evaluation Module for Yanomami Translation Model
#
# This module implements HellaSwag benchmark evaluation for the Yanomami translation model.
# It allows comparing the model's performance against baseline GPT-2 and GPT-3 metrics.

import os
import json
import torch
import logging
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    AutoModelForCausalLM,
    AutoTokenizer
)

# Import the tokenizer enhancement module
from yanomami_tokenizer.tokenizer_enhancement import (
    enhance_tokenizer,
    load_enhanced_tokenizer,
    SPECIAL_CHAR_WORDS,
    replace_special_chars
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"hellaswag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

class HellaSwagEvaluator:
    """
    Evaluator for HellaSwag benchmark to assess model performance
    
    This class provides methods to evaluate language models on the HellaSwag benchmark,
    which tests commonsense reasoning and language understanding capabilities.
    """
    
    def __init__(self, model_path=None, model_type="gpt2", device=None):
        """
        Initialize the HellaSwag evaluator
        
        Args:
            model_path (str): Path to the fine-tuned model directory
            model_type (str): Type of model ('gpt2', 'yanomami', etc.)
            device (str): Device to run evaluation on ('cuda', 'mps', 'cpu')
        """
        self.model_path = model_path
        self.model_type = model_type
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                      "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 
                                      "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Load HellaSwag dataset
        logger.info("Loading HellaSwag validation dataset...")
        self.dataset = load_dataset("hellaswag", "default", split="validation[:20%]")
        logger.info(f"Loaded {len(self.dataset)} HellaSwag validation examples")
        
        # Metrics
        self.metrics = {
            "accuracy": 0.0,
            "perplexity": 0.0,
            "multiple_choice_accuracy": 0.0,
            "examples_per_second": 0.0
        }
    
    def _load_model(self):
        """
        Load the model and tokenizer based on model type
        
        Returns:
            tuple: (model, tokenizer)
        """
        if self.model_type == "yanomami" and self.model_path is not None:
            logger.info(f"Loading Yanomami fine-tuned model from {self.model_path}")
            try:
                # Load the tokenizer with Yanomami enhancements
                tokenizer = load_enhanced_tokenizer(self.model_path)
                
                # Load the model
                model = GPT2LMHeadModel.from_pretrained(self.model_path)
                model.to(self.device)
                model.eval()
                
                return model, tokenizer
            except Exception as e:
                logger.error(f"Error loading Yanomami model: {e}")
                raise
        else:
            # Load standard GPT-2 model for baseline comparison
            model_name = self.model_path if self.model_path else "gpt2"
            logger.info(f"Loading baseline model: {model_name}")
            
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                model = GPT2LMHeadModel.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                
                return model, tokenizer
            except Exception as e:
                logger.error(f"Error loading baseline model: {e}")
                raise
    
    def _format_hellaswag_prompt(self, example):
        """
        Format the HellaSwag example into a prompt for the model
        
        Args:
            example (dict): HellaSwag example
            
        Returns:
            str: Formatted prompt
        """
        # Extract context and endings
        context = example["ctx"]
        endings = example["endings"]
        
        # Format as a standard prompt
        prompt = f"{context}\nWhat happens next?\n"
        
        return prompt
    
    def _calculate_ending_scores(self, prompt, endings):
        """
        Calculate scores for each possible ending
        
        Args:
            prompt (str): The context prompt
            endings (list): List of possible endings
            
        Returns:
            list: Scores for each ending
        """
        scores = []
        
        for ending in endings:
            # Tokenize the full sequence (prompt + ending)
            full_text = f"{prompt}{ending}"
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            
            # Get the token IDs for just the ending part
            ending_tokens = self.tokenizer(ending, return_tensors="pt")["input_ids"].to(self.device)
            ending_length = ending_tokens.size(1)
            
            # Calculate loss for the ending tokens only
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
            # Convert loss to score (negative log-likelihood)
            score = -loss.item() * ending_length
            scores.append(score)
            
        return scores
    
    def evaluate(self, num_examples=None, verbose=True):
        """
        Evaluate the model on HellaSwag benchmark
        
        Args:
            num_examples (int): Number of examples to evaluate (None for all)
            verbose (bool): Whether to print detailed progress
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Starting HellaSwag evaluation for {self.model_type} model")
        
        # Limit number of examples if specified
        eval_dataset = self.dataset
        if num_examples is not None and num_examples < len(self.dataset):
            eval_dataset = self.dataset.select(range(num_examples))
            
        logger.info(f"Evaluating on {len(eval_dataset)} examples")
        
        # Initialize metrics
        correct = 0
        total_loss = 0.0
        multiple_choice_correct = 0
        
        # Start timing
        start_time = time.time()
        
        # Evaluate each example
        for i, example in enumerate(tqdm(eval_dataset, desc="Evaluating")):
            try:
                # Format prompt
                prompt = self._format_hellaswag_prompt(example)
                
                # Get scores for each ending
                scores = self._calculate_ending_scores(prompt, example["endings"])
                
                # Get the predicted ending index (highest score)
                predicted_idx = np.argmax(scores)
                
                # Check if prediction matches the label
                # Convert label from string to integer
                correct_idx = int(example["label"])
                if predicted_idx == correct_idx:
                    multiple_choice_correct += 1
                
                # Calculate perplexity for the correct ending
                correct_ending = example["endings"][correct_idx]
                full_text = f"{prompt}{correct_ending}"
                inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                total_loss += loss.item()
                
                # Log progress if verbose
                if verbose and (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(eval_dataset)} examples")
                    logger.info(f"Current multiple-choice accuracy: {multiple_choice_correct / (i + 1):.4f}")
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue
        
        # Calculate final metrics
        elapsed_time = time.time() - start_time
        examples_per_second = len(eval_dataset) / elapsed_time
        multiple_choice_accuracy = multiple_choice_correct / len(eval_dataset)
        perplexity = torch.exp(torch.tensor(total_loss / len(eval_dataset))).item()
        
        # Store metrics
        self.metrics = {
            "accuracy": multiple_choice_accuracy,
            "perplexity": perplexity,
            "examples_per_second": examples_per_second,
            "eval_time": elapsed_time
        }
        
        # Log results
        logger.info(f"\n{'='*50}\nHellaSwag Evaluation Results for {self.model_type}\n{'='*50}")
        logger.info(f"Multiple-choice accuracy: {multiple_choice_accuracy:.4f}")
        logger.info(f"Perplexity: {perplexity:.4f}")
        logger.info(f"Evaluation speed: {examples_per_second:.2f} examples/second")
        logger.info(f"Total evaluation time: {elapsed_time:.2f} seconds")
        logger.info(f"{'='*50}")
        
        return self.metrics
    
    def compare_with_baselines(self, baseline_results=None):
        """
        Compare evaluation results with baseline models
        
        Args:
            baseline_results (dict): Dictionary of baseline results
            
        Returns:
            dict: Comparison metrics
        """
        # Default GPT-2 and GPT-3 baseline results on HellaSwag
        # These are approximate values based on published results
        default_baselines = {
            "gpt2": {"accuracy": 0.3382, "perplexity": 21.5},
            "gpt2-medium": {"accuracy": 0.4108, "perplexity": 17.2},
            "gpt2-large": {"accuracy": 0.4650, "perplexity": 14.8},
            "gpt2-xl": {"accuracy": 0.5021, "perplexity": 12.9},
            "gpt3": {"accuracy": 0.7866, "perplexity": 7.8}
        }
        
        # Use provided baselines or defaults
        baselines = baseline_results if baseline_results else default_baselines
        
        # Compare with baselines
        comparison = {}
        for model_name, metrics in baselines.items():
            accuracy_diff = self.metrics["accuracy"] - metrics["accuracy"]
            perplexity_diff = metrics["perplexity"] - self.metrics["perplexity"]
            
            comparison[model_name] = {
                "accuracy_diff": accuracy_diff,
                "accuracy_relative": accuracy_diff / metrics["accuracy"] * 100,
                "perplexity_diff": perplexity_diff,
                "perplexity_relative": perplexity_diff / metrics["perplexity"] * 100
            }
        
        # Log comparison
        logger.info(f"\n{'='*50}\nComparison with Baseline Models\n{'='*50}")
        for model_name, diff in comparison.items():
            logger.info(f"Compared to {model_name}:")
            logger.info(f"  Accuracy: {diff['accuracy_diff']:.4f} absolute ({diff['accuracy_relative']:.2f}% relative)")
            logger.info(f"  Perplexity: {diff['perplexity_diff']:.4f} absolute ({diff['perplexity_relative']:.2f}% relative)")
        logger.info(f"{'='*50}")
        
        return comparison
    
    def plot_comparison_chart(self, comparison, output_dir="./evaluation_results", phase=None, epoch=None):
        """
        Generate a bar chart comparing model performance with baselines
        
        Args:
            comparison (dict): Comparison metrics dictionary
            output_dir (str): Directory to save the chart
            phase (int): Current training phase number
            epoch (int): Current epoch number
            
        Returns:
            str: Path to the saved chart file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract model names and accuracy values
        model_names = list(comparison.keys()) + [self.model_type]
        accuracy_values = [0.0] * len(model_names)  # Initialize with zeros
        perplexity_values = [0.0] * len(model_names)  # Initialize with zeros
        
        # Fill in baseline values
        for i, model_name in enumerate(comparison.keys()):
            # Calculate absolute accuracy for baseline models
            baseline_accuracy = self.metrics['accuracy'] - comparison[model_name]['accuracy_diff']
            accuracy_values[i] = baseline_accuracy
            
            # Calculate absolute perplexity for baseline models
            baseline_perplexity = self.metrics['perplexity'] - comparison[model_name]['perplexity_diff']
            perplexity_values[i] = baseline_perplexity
        
        # Add current model's values
        accuracy_values[-1] = self.metrics['accuracy']
        perplexity_values[-1] = self.metrics['perplexity']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot accuracy comparison
        bars1 = ax1.bar(model_names, accuracy_values, color=['lightblue'] * (len(model_names) - 1) + ['darkblue'])
        ax1.set_title('Accuracy Comparison', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, max(accuracy_values) * 1.2)  # Add 20% headroom
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Plot perplexity comparison
        bars2 = ax2.bar(model_names, perplexity_values, color=['lightcoral'] * (len(model_names) - 1) + ['darkred'])
        ax2.set_title('Perplexity Comparison (Lower is Better)', fontsize=14)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Add title with phase and epoch information if provided
        title = f'HellaSwag Benchmark Comparison for {self.model_type.capitalize()}'
        if phase is not None and epoch is not None:
            title += f' - Phase {phase}, Epoch {epoch}'
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        phase_epoch_str = f"_phase{phase}_epoch{epoch}" if phase is not None and epoch is not None else ""
        chart_file = os.path.join(output_dir, f"hellaswag_comparison_{self.model_type}{phase_epoch_str}_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison chart saved to {chart_file}")
        return chart_file
    
    def plot_metrics_history(self, history_file, output_dir="./evaluation_results"):
        """
        Generate a line chart showing metrics history across epochs
        
        Args:
            history_file (str): Path to the metrics history JSON file
            output_dir (str): Directory to save the chart
            
        Returns:
            str: Path to the saved chart file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metrics history
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Could not load metrics history from {history_file}")
            return None
        
        # Extract data for plotting
        phases = []
        epochs = []
        accuracies = []
        perplexities = []
        
        for entry in history:
            if 'phase' in entry and 'epoch' in entry:
                phases.append(entry['phase'])
                epochs.append(entry['epoch'])
                accuracies.append(entry['metrics']['accuracy'])
                perplexities.append(entry['metrics']['perplexity'])
        
        if not epochs:
            logger.warning("No epoch data found in metrics history")
            return None
        
        # Create x-axis labels
        x_labels = [f"P{p}E{e}" for p, e in zip(phases, epochs)]
        x_positions = list(range(len(x_labels)))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot accuracy history
        ax1.plot(x_positions, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
        ax1.set_title('HellaSwag Accuracy Across Training', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, acc in enumerate(accuracies):
            ax1.text(x_positions[i], acc + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot perplexity history
        ax2.plot(x_positions, perplexities, 'o-', color='red', linewidth=2, markersize=8)
        ax2.set_title('HellaSwag Perplexity Across Training (Lower is Better)', fontsize=14)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(x_labels, rotation=45)
        ax2.set_xlabel('Phase and Epoch', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, perp in enumerate(perplexities):
            ax2.text(x_positions[i], perp + 0.5, f'{perp:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Add overall title
        fig.suptitle(f'HellaSwag Metrics History for {self.model_type.capitalize()}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = os.path.join(output_dir, f"hellaswag_history_{self.model_type}_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics history chart saved to {chart_file}")
        return chart_file
    
    def save_results(self, output_dir="./evaluation_results", phase=None, epoch=None):
        """
        Save evaluation results to a file
        
        Args:
            output_dir (str): Directory to save results
            phase (int): Current training phase number
            epoch (int): Current epoch number
            
        Returns:
            str: Path to the saved results file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results dictionary
        results = {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": str(self.device),
            "dataset": "hellaswag",
            "num_examples": len(self.dataset),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": self.metrics
        }
        
        # Add phase and epoch information if provided
        if phase is not None:
            results["phase"] = phase
        if epoch is not None:
            results["epoch"] = epoch
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        phase_epoch_str = f"_phase{phase}_epoch{epoch}" if phase is not None and epoch is not None else ""
        output_file = os.path.join(output_dir, f"hellaswag_results_{self.model_type}{phase_epoch_str}_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also append to history file for tracking progress across epochs
        history_file = os.path.join(output_dir, f"hellaswag_history_{self.model_type}.json")
        
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
                
            history.append(results)
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            # Generate history chart if we have multiple entries
            if len(history) > 1:
                self.plot_metrics_history(history_file, output_dir)
                
        except Exception as e:
            logger.error(f"Error updating history file: {e}")
        
        logger.info(f"Results saved to {output_file}")
        return output_file


def main():
    """
    Main function to run HellaSwag evaluation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate models on HellaSwag benchmark")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model directory")
    parser.add_argument("--model_type", type=str, default="gpt2", choices=["gpt2", "yanomami"], help="Type of model to evaluate")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on (cuda, mps, cpu)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save results")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline models")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HellaSwagEvaluator(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(num_examples=args.num_examples)
    
    # Compare with baselines if requested
    if args.compare:
        evaluator.compare_with_baselines()
    
    # Save results
    evaluator.save_results(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
