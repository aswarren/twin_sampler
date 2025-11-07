#!/bin/bash
#
# run_label_components.sh
#
# This worker script is designed to be called by a Slurm job array.
# It processes a SINGLE EpiHiper output file and labels its transmission
# components with variants based on a real-world importation schedule.
#
# It expects three command-line arguments:
#   $1: The full path to the input EpiHiper file.
#   $2: The full path to the importation schedule CSV file.
#   $3: The full path for the output file.
#

echo "Activating Conda environment..."
source /project/biocomplexity/asw3xp/miniconda3/etc/profile.d/conda.sh
conda activate # <-- Or specify your environment, e.g., 'conda activate escape'

# --- Argument Check ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_epihiper_file> <schedule_csv_file> <output_file>"
    exit 1
fi

EPIHIPER_FILE="$1"
SCHEDULE_FILE="$2"
OUTPUT_FILE="$3"

# --- Configuration ---
# Set the path to your Python script and the labeling mode.

# Path to the Python script to be executed
PYTHON_SCRIPT="label_components.py"

# The labeling mode to use:
# 1 = Temporal & Proportional Introduction Matching
# 2 = Bipartite Matching on Time & Cluster Volume
LABELING_MODE=2 # Defaulting to the more sophisticated Mode 2

# --- Processing Logic ---
echo "Processing input file: $EPIHIPER_FILE"
echo "Using schedule file: $SCHEDULE_FILE"
echo "Output will be saved to: $OUTPUT_FILE"
echo "" # Add a blank line for readability

# --- Assemble the command in a bash array for safety and clarity ---
cmd=(
    "python"
    "${PYTHON_SCRIPT}"
    "--epihiper_input"
    "${EPIHIPER_FILE}"
    "--schedule_input"
    "${SCHEDULE_FILE}"
    "--output"
    "${OUTPUT_FILE}"
    "--mode"
    "${LABELING_MODE}"
)

# --- Echo the exact command for logging ---
# The 'printf "%q "' command prints each element of the array with proper quoting,
# creating a command that can be copied and pasted directly into a terminal to rerun.
echo "--- EXECUTING COMMAND ---"
printf "%q " "${cmd[@]}"
echo # Add a newline for cleaner log output
echo "---------------------------"
echo "" # Add another blank line for readability

# --- Execute the command ---
# The "${cmd[@]}" syntax expands the array safely.
"${cmd[@]}"

echo "Python script finished for $EPIHIPER_FILE."
