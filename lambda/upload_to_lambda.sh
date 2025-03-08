#!/bin/bash

# Script to upload Yanomami repository to Lambda Cloud instance

# Configuration
LAMBDA_IP="192.222.56.75"
SSH_KEY="~/.ssh/yanomami_lambda_key"
LOCAL_DIR="/Users/renanserrano/CascadeProjects/Yanomami/finetunning"
REMOTE_DIR="~/yanomami-finetunning"

# Create the destination directory on the remote server
ssh -i $SSH_KEY ubuntu@$LAMBDA_IP "mkdir -p $REMOTE_DIR"

# Exclude patterns for files/directories we don't want to upload
EXCLUDE_PATTERNS=(
  ".git/"
  ".gitignore"
  "__pycache__/"
  "*.pyc"
  "*.pyo"
  "*.pyd"
  ".DS_Store"
  "logs/"
  "visualization_results/"
  "yanomami_translator_model/"
  "lambda/"
  "path/"
  "RAG/"
  "checkpoints/"
  "enhanced_yanomami_translator/"
  ".ipynb_checkpoints/"
  "*.ipynb"
  "*.pem"
  ".env"
)

# Build the exclude arguments for rsync
EXCLUDE_ARGS=""
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
  EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude='$pattern'"
 done

# Upload the repository using rsync
echo "Uploading Yanomami repository to Lambda Cloud instance..."
eval "rsync -avz --progress -e 'ssh -i $SSH_KEY' $EXCLUDE_ARGS $LOCAL_DIR/ ubuntu@$LAMBDA_IP:$REMOTE_DIR/"

echo "\nUpload complete! Your code is now available on the Lambda instance at $REMOTE_DIR"
