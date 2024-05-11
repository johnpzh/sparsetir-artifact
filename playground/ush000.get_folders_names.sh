#!/bin/bash

# Check if a directory is provided as an argument
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <directory> <num_parts>"
  exit 1
fi

# Check if the provided directory exists
if [ ! -d "$1" ]; then
  echo "Error: The provided directory does not exist."
  exit 1
fi

num_parts=$2

# Get all folder names under the given directory and save them to an array
folder_names=($(find "$1" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

# Save all names to a text file
basename="dataset_names_2904"
output_file="${basename}.txt"
:> "${output_file}"
echo "MATRICES=( \\" >> "${output_file}"
for name in "${folder_names[@]}"; do
  echo "\"${name}\" \\" >> "${output_file}"
done
echo ")" >> "${output_file}"

echo "The folder names have been saved to ${output_file}."

total_size=${#folder_names[@]}
echo "total_size: ${total_size}"

names_per_file=$(( (total_size + num_parts - 1) / num_parts ))
echo "names_per_file: ${names_per_file}"

for ((f_i = 0; f_i < num_parts; ++f_i)); do
  loc_start=$((f_i * names_per_file))
  loc_end=$((loc_start + names_per_file))
  if [[ $loc_end  -ge $total_size ]]; then
    loc_end=$((total_size))
  fi
  length=$((loc_end - loc_start))
  part=("${folder_names[@]:$loc_start:$length}")

  # Write to txt
  ind=$((f_i + 1))
  output_file="${basename}_part${ind}.txt"
  :> "${output_file}"
  echo "MATRICES=( \\" >> "${output_file}"
  for name in "${part[@]}"; do
    echo "\"${name}\" \\" >> "${output_file}"
  done
  echo ")" >> "${output_file}"
done
