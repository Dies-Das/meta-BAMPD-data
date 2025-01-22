#!/bin/bash

mkdir data

# Path to your C++ program
PROGRAM="bin/meta-BAMDP"  # Update this to the actual path of your compiled program

# Fixed parameters
ARMS=2
BOUND_TYPE=3
BOUND=3

# Varying parameters
T_VALUES=(6 9 12)
COST_MIN=0.0
COST_MAX=0.5
COST_SAMPLES=20

# Create an array of 20 evenly spaced cost values between COST_MIN and COST_MAX
COST_VALUES=($(python3 -c "import numpy as np; print(' '.join(map(str, np.linspace($COST_MIN, $COST_MAX, $COST_SAMPLES))))"))

# Loop over all combinations of t and cost
for T in "${T_VALUES[@]}"; do
  
    # Create an output file name based on the parameters
    OUTPUT_FILE="data/"

    # Call the program with the arguments
    $PROGRAM -o "$OUTPUT_FILE" -b $BOUND_TYPE -t $T -a $ARMS -n $BOUND --min $COST_MIN --max $COST_MAX --samples $COST_SAMPLES

    # Check if the program succeeded
    if [ $? -ne 0 ]; then
      echo "Error: Program failed for t=$T, cost=$COST" >&2
      exit 1
    fi

    echo "Generated: $OUTPUT_FILE"

done

echo "All runs completed successfully."