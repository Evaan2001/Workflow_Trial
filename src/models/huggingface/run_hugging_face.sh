#!/bin/bash
# Shell script to run our NumPy code

# Get the directory where this script is located
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Hugging Face test from directory: $DIR"
echo "Python version:"
python3 --version

# Run the Python script
echo "Running Python script ..."
python3 "$DIR/transformers_client.py"

# Capture the exit code
exit_code=$?

echo "Python script completed with exit code: $exit_code"

# Return the same exit code
exit $exit_code
