#!/bin/bash
#
# run_simulations.sh
#
# This script runs the simulate_linelist.py script on multiple raw EpiHiper
# output files in a batch process. It generates a separate simulated line list
# for each input file provided in the EPIHIPER_FILES array.
#
# Usage:
#   1. Configure the file paths in the "Configuration" section below.
#   2. Add the EpiHiper files you want to process to the "Input Files" array.
#   3. Make the script executable: chmod +x run_simulations.sh
#   4. Run the script: ./run_simulations.sh
#

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

#The exit states to focus on
#right now this is set to the second variant wave. we should change this once we are comfortable with perfomrance
PREFIX_FILTER="['A2', 'P2', 'I2', 'dm2', 'hM2']"

# A fixed random seed for reproducibility across all runs.
# Remove the --seed line in the command below to use a different random seed each time.
SEED=42

# Directory where the output line lists will be saved
OUTPUT_DIR="results"


# --- Input Files ---
# Add all the raw EpiHiper simulation output files you want to process to this array.
# The script will loop over each one.
EPIHIPER_FILES=(
  "/project/bii_nssac/biocomplexity/c4gc_asw3xp/demo/epihiper_batch_1_0.25_replicate_0.output.csv.gz"
  # "data/raw/another_simulation_run.csv"
)


# --- Processing Loop ---
echo "Starting batch processing of EpiHiper files..."

# Create the output directory if it doesn't already exist
mkdir -p "$OUTPUT_DIR"

# Loop through each file in the EPIHIPER_FILES array
for epi_file in "${EPIHIPER_FILES[@]}"; do
  echo "----------------------------------------------------"
  
  # Check if the input file actually exists before processing
  if [ ! -f "$epi_file" ]; then
    echo "Warning: Input file not found, skipping: $epi_file"
    continue
  fi
  
  echo "Processing input file: $epi_file"

  # Generate a unique output filename based on the input filename.
  # This takes a file like "data/raw/epihiper_run_alpha.csv" and creates
  # an output file named "results/epihiper_run_alpha_linelist.csv".
  base_name=$(basename "$epi_file")
  output_prefix="${base_name%.csv}"
  output_file="${OUTPUT_DIR}/${output_prefix}_linelist.csv"

  echo "Output will be saved to: $output_file"
  echo "" # Add a blank line for readability

  # Execute the Python script with all the configured parameters
  python "$PYTHON_SCRIPT" \
    --epihiper "$epi_file" \
    --people "$PERSONTRAIT_FILE" \
    --households "$HOUSEHOLD_FILE" \
    --rucc "$RUCC_FILE" \
    --ascertain "$PARAMS_FILE" \
    --start_date "$START_DATE" \
    --start_tick "$START_TICK" \
    --out "$output_file" \
    --seed "$SEED" \
    --prefix_override "$PREFIX_FILTER"

  echo "Finished processing $epi_file."
done

echo "----------------------------------------------------"
echo "All files processed. Check the '$OUTPUT_DIR' directory for results."
