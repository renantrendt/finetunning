# Test script for HellaSwag evaluation functionality
#
# This script tests the HellaSwag evaluation module by running an evaluation
# on a pre-trained GPT-2 model and generating visualization charts

import os
import torch
import argparse
import logging
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from hellaswag_evaluation import HellaSwagEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hellaswag_test.log')
    ]
)
logger = logging.getLogger(__name__)

def test_hellaswag_evaluation(model_name="gpt2", num_examples=20, output_dir="./hellaswag_test_results"):
    """
    Test the HellaSwag evaluation module with a pre-trained model
    
    Args:
        model_name (str): Name of the pre-trained model to test
        num_examples (int): Number of examples to evaluate
        output_dir (str): Directory to save results and charts
    """
    logger.info(f"\n{'='*60}\nTesting HellaSwag evaluation with {model_name} model\n{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize the evaluator directly with model name
        logger.info(f"Initializing HellaSwag evaluator with {model_name}")
        evaluator = HellaSwagEvaluator(
            model_path=model_name,  # Use model name directly from Hugging Face
            model_type="gpt2",
            device=str(device)
        )
        
        # Run evaluation
        logger.info(f"Evaluating on {num_examples} HellaSwag examples")
        metrics = evaluator.evaluate(num_examples=num_examples)
        
        # Compare with baselines
        logger.info("Comparing with baseline models...")
        comparison = evaluator.compare_with_baselines()
        
        # Generate comparison chart
        logger.info("Generating comparison chart...")
        chart_path = evaluator.plot_comparison_chart(
            comparison=comparison,
            output_dir=output_dir,
            phase=1,  # Simulate phase 1
            epoch=1   # Simulate epoch 1
        )
        logger.info(f"Comparison chart saved to: {chart_path}")
        
        # Save results with phase and epoch information
        logger.info("Saving evaluation results...")
        results_file = evaluator.save_results(
            output_dir=output_dir,
            phase=1,
            epoch=1
        )
        
        # Run another evaluation to simulate multiple epochs
        logger.info("\nRunning second evaluation to simulate epoch 2...")
        metrics2 = evaluator.evaluate(num_examples=num_examples)
        
        # Save results for "epoch 2"
        results_file2 = evaluator.save_results(
            output_dir=output_dir,
            phase=1,
            epoch=2
        )
        
        # Log summary
        logger.info(f"\n{'*'*60}\nHellaSwag evaluation test completed successfully\n")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Metrics: Accuracy: {metrics['accuracy']:.4f}, Perplexity: {metrics['perplexity']:.4f}")
        logger.info(f"{'*'*60}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during HellaSwag evaluation test: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Test HellaSwag evaluation functionality")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--examples", type=int, default=20, help="Number of examples to evaluate")
    parser.add_argument("--output", type=str, default="./hellaswag_test_results", help="Output directory")
    
    args = parser.parse_args()
    
    success = test_hellaswag_evaluation(
        model_name=args.model,
        num_examples=args.examples,
        output_dir=args.output
    )
    
    if success:
        print("\n✅ HellaSwag evaluation test completed successfully!")
        print(f"Results and charts saved to: {args.output}")
    else:
        print("\n❌ HellaSwag evaluation test failed. Check the logs for details.")

if __name__ == "__main__":
    main()
