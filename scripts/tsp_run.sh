#!/bin/bash

# Check if the directory argument is provided
if [ -z "$1" ]; then
	echo "Usage: $0 <directory> [--debug]"
	exit 1
fi

# Parse arguments
DIRECTORY=$(realpath "$1")
CMD="${CMD:-tsp bash run.sh}"
DEBUG=false

# Check for debug flag
if [[ "$2" == "--debug" ]]; then
	DEBUG=true
fi

# Find all run.sh files, cd to their parent directories, and execute or print the command
find "$DIRECTORY" -type f -name "run.sh" | while read -r file; do
	# Get the parent directory of the run.sh file
	PARENT_DIR=$(dirname "$file")

	# Change to the parent directory
	cd "$PARENT_DIR" || {
		echo "Failed to cd to $PARENT_DIR"
		continue
	}

	# Print or execute the command based on the debug flag
	if [ "$DEBUG" = true ]; then
		echo "[DEBUG] Would execute in $PARENT_DIR: $CMD"
	else
		echo "Executing command in $PARENT_DIR"
		eval "$CMD"
	fi
done
