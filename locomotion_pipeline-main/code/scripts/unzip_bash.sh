#!/bin/bash

# 1. Check if a directory was provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/target_folder"
    exit 1
fi

# 2. Store the target directory and remove trailing slashes
TARGET_DIR="${1%/}"

# 3. Check if the path is a valid directory
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

# 4. Loop through zip files in that specific folder
for file in "$TARGET_DIR"/*.zip; do

    # Avoid errors if no zip files are found
    [ -e "$file" ] || { echo "No zip files found in $TARGET_DIR"; exit; }

    # Get the base name (filename without path) and remove .zip
    filename=$(basename "$file")
    folder_name="${filename%.zip}"
    
    # Define the full path for the new folder
    output_path="$TARGET_DIR/$folder_name"

    echo "Extracting $filename to $output_path..."

    # Create the folder and unzip
    mkdir -p "$output_path"
    unzip -q "$file" -d "$output_path"
done

echo "Done unzip! remove *.zip"
rm -r $TARGET_DIR/*.zip
