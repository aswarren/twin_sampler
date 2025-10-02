#!/bin/bash
#
# worker_run_single_sim.sh
#
# This script is a "worker" designed to be called by a Slurm job array.
# It processes a SINGLE EpiHiper output file and generates a simulated line list.
#
# It expects two command-line arguments:
#   $1: The full path to the input EpiHiper file.
#   $2: The base path for the output file.
#

echo "Activating Conda environment..."
source /project/biocomplexity/asw3xp/miniconda3/etc/profile.d/conda.sh
conda activate # <-- Assuming a base environment, or specify one like 'conda activate my_env'

# --- Argument Check ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_epihiper_file> <output_base_path>"
    exit 1
fi

EPIHIPER_FILE="$1"
OUTPUT_FILE="$2"

# --- Configuration ---
# Set the paths to your static input files and parameters here.

# Path to the Python script to be executed
PYTHON_SCRIPT="simulate_linelist.py"

# Path to the synthetic population trait file
PERSONTRAIT_FILE="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/va_persontrait_epihiper.txt"

# Path to the synthetic household file
HOUSEHOLD_FILE="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/va_household.csv"

# Path to the Rural-Urban Continuum Codes file
RUCC_FILE="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/Ruralurbancontinuumcodes2023.csv"

# Path to the ascertainment model parameters YAML file
PARAMS_FILE="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/asw_test/twin_sampler/scripts/linelist_generation/ascertainment_parameters.yaml"

# Simulation parameters for date conversion. The start_date corresponds to the start_tick.
START_DATE="2021-04-07"
START_TICK=128
STOP_TICK=397

# The exit states to focus on
PREFIX_FILTER="[\"A2\", \"P2\", \"I2\", \"dM2\", \"hM2\"]"

# A fixed random seed for reproducibility
SEED=42

# --- Processing Logic ---
echo "Processing input file: $EPIHIPER_FILE"
echo "Output will be saved based on: $OUTPUT_FILE"
echo "" # Add a blank line for readability


# Execute the Python script with all the configured parameters
# NOTE: Corrected --ascertain to --params to match the python script
python "$PYTHON_SCRIPT" \
  --epihiper "$EPIHIPER_FILE" \
  --people "$PERSONTRAIT_FILE" \
  --households "$HOUSEHOLD_FILE" \
  --rucc "$RUCC_FILE" \
  --params "$PARAMS_FILE" \
  --start_date "$START_DATE" \
  --start_tick "$START_TICK" \
  --stop_tick "$STOP_TICK" \
  --out "$OUTPUT_FILE" \
  --seed "$SEED" \
  --prefix_override "$PREFIX_FILTER"\
  --output_all_events

echo "Python script finished for $EPIHIPER_FILE."