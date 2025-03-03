#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upload configuration files to Hugging Face
"""

from huggingface_hub import HfApi
import os
import time

# Initialize the Hugging Face API
api = HfApi()

# Set the repository name
repo_id = "renanserrano/yanomami-finetuning"

# Upload necessary configuration files
model_dir = "/Users/renanserrano/CascadeProjects/Yanomami/finetunning/gpt2_yanomami_translator"
files_to_upload = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json"
]

for file in files_to_upload:
    file_path = os.path.join(model_dir, file)
    if os.path.exists(file_path):
        print(f"Uploading {file}...")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"{file} uploaded successfully!")
            # Add a small delay between uploads to avoid overwhelming the server
            time.sleep(2)
        except Exception as e:
            print(f"Error uploading {file}: {str(e)}")
    else:
        print(f"Warning: {file} not found in {model_dir}")

print("\nAll configuration files have been processed!")
print(f"Your model is available at: https://huggingface.co/{repo_id}")
