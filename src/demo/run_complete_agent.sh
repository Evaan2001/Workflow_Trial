#!/bin/bash

# Get the directory where this script is located
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Hybrid Agent test from directory: $DIR"
echo "Python version:"
python3 --version

# Run the Python script with all arguments passed to the shell script
echo "Running Python script with arguments: $@"
python3 "$DIR/complete_agent.py" "$@"
# Capture the exit code
exit_code=$?

echo "Python script completed with exit code: $exit_code"

# Return the same exit code
exit $exit_code
