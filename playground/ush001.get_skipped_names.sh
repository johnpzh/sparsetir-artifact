#!/bin/bash

# Check if a directory is provided as an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: bash $0 <directory>"
  exit 1
fi

# Check if the provided directory exists
if [ ! -d "$1" ]; then
  echo "Error: The provided directory does not exist."
  exit 1
fi

# Get all folder names under the given directory and save them to an array
# file_names=($(find "$1" -mindepth 1 -maxdepth 1 -type f -name "skipped_*" -exec basename {} \;))
file_names=($(find "$1" -mindepth 1 -maxdepth 1 -type f -name "skipped_*"))

# Save all names to a text file
output_dir="output"
output_file="${output_dir}/output_skipped_collect.csv"
:> "${output_file}"

count=0
is_first=true
for name in "${file_names[@]}"; do
  if ${is_first}; then
    head -n 1 ${name} >> "${output_file}"
    is_first=false
  fi
  tail -n 1 ${name} >> "${output_file}"
  count=$((count + 1))
done

echo "Totally ${count} skipped matrices. Written to ${output_file}"
