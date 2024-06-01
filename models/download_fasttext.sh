#!/bin/bash

# Directory to save the FastText model
directory="./models/fasttext"

# Create the directory if it does not exist
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
fi

# URL to download the FastText model
url="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
output_file="$directory/cc.en.300.bin.gz"

# Download the FastText model
curl -o "$output_file" "$url"

# Unzip the downloaded file
gunzip "$output_file"

# Remove the gz file after extraction
rm "$output_file"