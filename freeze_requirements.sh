#!/bin/bash

# List of libraries to exclude (space-separated)
EXCLUDE_LIBS=("-e git")

# Get the list of installed packages using pip freeze
PIP_FREEZE_OUTPUT=$(pip freeze)

# Loop through each library to exclude
for LIB in "${EXCLUDE_LIBS[@]}"; do
    # Exclude the library by filtering it out from the pip freeze output
    PIP_FREEZE_OUTPUT=$(echo "$PIP_FREEZE_OUTPUT" | grep -v "$LIB")
done

# Save the filtered output to requirements.txt
echo "$PIP_FREEZE_OUTPUT" > requirements.txt

# Confirm the action
echo "Requirements saved to requirements.txt excluding: ${EXCLUDE_LIBS[@]}"
