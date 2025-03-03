#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upload dataset folder to Hugging Face
"""

from huggingface_hub import HfApi
import os
import time

# Initialize the Hugging Face API
api = HfApi()

# Set the repository name
repo_id = "renanserrano/yanomami-finetuning"

# Dataset directory
dataset_dir = "/Users/renanserrano/CascadeProjects/Yanomami/finetunning/yanomami_dataset"

# Get all files in the dataset directory
dataset_files = [f for f in os.listdir(dataset_dir) if not f.startswith('.')]

# Upload each file
for file in dataset_files:
    file_path = os.path.join(dataset_dir, file)
    if os.path.isfile(file_path):
        print(f"Uploading {file}...")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"yanomami_dataset/{file}",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"{file} uploaded successfully!")
            # Add a small delay between uploads to avoid overwhelming the server
            time.sleep(2)
        except Exception as e:
            print(f"Error uploading {file}: {str(e)}")

print("\nAll dataset files have been processed!")
print(f"Your dataset is available at: https://huggingface.co/{repo_id}/tree/main/yanomami_dataset")
