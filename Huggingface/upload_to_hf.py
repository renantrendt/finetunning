#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upload model to Hugging Face
"""

from huggingface_hub import HfApi
import os

# Initialize the Hugging Face API
api = HfApi()

# Set the repository name
repo_id = "renanserrano/yanomami-finetuning"

# Path to the model file
model_path = "/Users/renanserrano/CascadeProjects/Yanomami/finetunning/gpt2_yanomami_translator/model.safetensors"

# Upload the model file
print(f"Uploading {model_path} to {repo_id}...")
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model.safetensors",
    repo_id=repo_id,
    repo_type="model",
)
print("Upload complete!")

# Upload other necessary files
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
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"{file} uploaded successfully!")
    else:
        print(f"Warning: {file} not found in {model_dir}")

print("\nAll files have been uploaded to Hugging Face!")
print(f"Your model is available at: https://huggingface.co/{repo_id}")
