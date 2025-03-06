# Running Yanomami Translation Model Training on Lambda Cloud

This guide explains how to run the Yanomami translation model training on Lambda Cloud GPU instances for faster training and better performance.

## Prerequisites

- A Lambda Cloud account (sign up at https://cloud.lambdalabs.com if you don't have one)
- An API key from Lambda Cloud (create one at https://cloud.lambdalabs.com/api-keys)
- Python 3.8+ installed on your local machine

## Setup Instructions

### 1. Install Required Dependencies

Make sure you have the required Python packages installed on your local machine:

```bash
pip install requests
```

### 2. Set Up Your Lambda Cloud API Key

You need to set your Lambda Cloud API key as an environment variable:

```bash
export LAMBDA_API_KEY=your_api_key_here
```

Alternatively, you can pass it directly to the script with the `--api-key` option.

### 3. Run the Lambda Training Script

The `run_lambda_training.py` script handles everything needed to run your training on Lambda Cloud:

```bash
python run_lambda_training.py
```

This script will:
1. Connect to Lambda Cloud using your API key
2. Launch a Lambda instance with the specified GPU
3. Transfer your code and dataset to the Lambda instance
4. Set up the environment on the Lambda instance
5. Start the training process
6. Monitor the training logs

### 4. Command Line Options

The script supports several command line options:

```bash
python run_lambda_training.py --instance-type gpu_1x_a100 --region us-east-1 --resume
```

- `--api-key`: Your Lambda Cloud API key (if not set as environment variable)
- `--instance-type`: Specify the Lambda instance type (e.g., gpu_1x_a10, gpu_1x_a100)
- `--region`: Specify the Lambda region
- `--setup-only`: Only set up the Lambda instance without starting training
- `--resume`: Resume training from the latest checkpoint
- `--list-instances`: List your running instances and exit
- `--list-instance-types`: List available instance types and exit
- `--terminate`: Terminate an instance with the given ID and exit

## Monitoring Training

The training process runs in a tmux session on the Lambda instance, allowing it to continue even if your connection is lost.

To view the training progress:

1. SSH into your Lambda instance:
   ```bash
   ssh ubuntu@<instance-ip>
   ```

2. Attach to the tmux session:
   ```bash
   tmux attach -t yanomami_training
   ```

3. To detach from the tmux session without stopping training, press `Ctrl+B`, then `D`.

## Managing Training Data

Your training data and checkpoints are stored in the `/lambda_storage` directory on the Lambda instance. This is a persistent storage location that will survive instance restarts.

Important directories:
- `/lambda_storage/yanomami_dataset`: Dataset files
- `/lambda_storage/checkpoints`: Training checkpoints
- `/lambda_storage/logs`: Training logs
- `/lambda_storage/enhanced_yanomami_translator`: Saved model outputs
- `/lambda_storage/visualization_results`: Training visualizations

## Retrieving Results

To retrieve your trained model and results from the Lambda instance:

```bash
rsync -avz ubuntu@<instance-ip>:/lambda_storage/enhanced_yanomami_translator/ ./enhanced_yanomami_translator/
rsync -avz ubuntu@<instance-ip>:/lambda_storage/visualization_results/ ./visualization_results/
rsync -avz ubuntu@<instance-ip>:/lambda_storage/logs/ ./logs/
```

## Stopping the Instance

When you're done with training, stop the Lambda instance to avoid unnecessary charges:

```bash
lambda instance stop <instance-id>
```

Or terminate it completely if you don't need it anymore:

```bash
lambda instance terminate <instance-id>
```

## Troubleshooting

### Connection Issues

If you have trouble connecting to the Lambda instance, check:
1. Your internet connection
2. The instance status in the Lambda Cloud dashboard
3. Your SSH key configuration

### Training Issues

If training fails or performs poorly:
1. Check the training logs in `/lambda_storage/logs/`
2. Try reducing the batch size if you encounter out-of-memory errors
3. Enable debug mode for more verbose output

### Data Transfer Issues

If you encounter issues with data transfer:
1. Ensure your dataset files are not too large (consider compressing them)
2. Check your internet connection stability
3. Try transferring files in smaller batches

## Configuration

You can modify the Lambda configuration settings in `lambda_config.py` to customize:
- Instance type and region
- Batch size and gradient accumulation steps
- Storage directories
- Mixed precision settings
