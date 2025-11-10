#!/bin/bash
#
# generate_linelist.sh (previously run_label_component.sh)
#
# This worker script processes a SINGLE EpiHiper output file, labels it with
# variants, and generates a simulated line list.
#
# It expects three command-line arguments:
#   $1: The full path to the input EpiHiper file.
#   $2: The full path to the importation schedule CSV file.
#   $3: The base path for the output file.
#

echo "Activating Conda environment..."
source /project/biocomplexity/asw3xp/miniconda3/etc/profile.d/conda.sh
conda activate

# --- Argument Check ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_epihiper_file> <schedule_csv_file> <output_base_path>"
    exit 1
fi

EPIHIPER_FILE="$1"
SCHEDULE_FILE="$2" # <-- NEW ARGUMENT
OUTPUT_FILE="$3"

# --- Configuration ---
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
VARIANT_MODE=2 # The variant labeling mode (1 or 2)

# --- Processing Logic ---
echo "Processing input file: $EPIHIPER_FILE"
echo "Output will be saved based on: $OUTPUT_FILE"
echo "" # Add a blank line for readability

# --- Assemble the command ---
cmd=(
    "python"
    "${PYTHON_SCRIPT}"
    "--epihiper"
    "${EPIHIPER_FILE}"
    "--people"
    "${PERSONTRAIT_FILE}"
    "--households"
    "${HOUSEHOLD_FILE}"
    "--rucc"
    "${RUCC_FILE}"
    "--ascertain"
    "${PARAMS_FILE}"
    "--start_date"
    "${START_DATE}"
    "--start_tick"
    "${START_TICK}"
    "--stop_tick"
    "${STOP_TICK}"
    "--out"
    "${OUTPUT_FILE}"
    "--seed"
    "${SEED}"
    "--prefix_override"
    "${PREFIX_FILTER}"
    "--output_all_events"
    # --- NEW ARGUMENTS FOR VARIANT LABELING ---
    "--schedule_input"
    "${SCHEDULE_FILE}"
    "--variant_mode"
    "${VARIANT_MODE}"
)

# --- Echo and Execute the command ---
echo "--- EXECUTING COMMAND ---"
printf "%q " "${cmd[@]}"
echo
echo "---------------------------"
"${cmd[@]}"

echo "Python script finished for $EPIHIPER_FILE."
