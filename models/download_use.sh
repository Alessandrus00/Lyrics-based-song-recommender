#!/bin/bash

# Base directory to save the model
base_directory="./models"
# Subdirectory for the USE model
use_directory="$base_directory/use"
# URL to download the USE model
url="https://www.kaggle.com/api/v1/models/google/universal-sentence-encoder/tensorFlow2/large/2/download"
# Output file path
output_file="$base_directory/model.tar.gz"

# Create the base directory if it does not exist
if [ ! -d "$base_directory" ]; then
    mkdir -p "$base_directory"
fi

# Download the USE model tarball
curl -L -o "$output_file" "$url"

# Create the USE subdirectory if it does not exist
if [ ! -d "$use_directory" ]; then
    mkdir -p "$use_directory"
fi

# Extract the tarball into the USE directory
tar xzf "$output_file" -C "$use_directory"

# Remove the tarball after extraction
rm "$output_file"