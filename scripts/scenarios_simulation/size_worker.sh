#!/usr/bin/env bash
#
# worker_run_single_batch.sh
#
# This script is a "worker" that runs a SINGLE scenario simulation for a
# specific replicate and a specific batch size.
#
# It expects two command-line arguments:
#   $1: The replicate ID number (e.g., 0).
#   $2: The batch size to use (e.g., 750).
#

# --- Argument Check ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <replicate_id> <batch_size>"
    echo "Example: $0 0 750"
    exit 1
fi

REPLICATE_ID="$1"
BATCH_SIZE="$2"

# --- Configuration (Static Paths) ---
PYTHON_SCRIPT="run_all_scenarios.py"
ROOT_DIR="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/asw_test/twin_sampler/scripts/linelist_generation/results"
POP_FILE="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/va_persontrait_epihiper.txt"
SWEEP_OUTROOT="/project/bii_nssac/biocomperty/c4gc_asw3xp/LineList/asw_test/twin_sampler/scripts/scenarios_simulation/scenario_runs/size_sweep"

# --- Environment Setup ---
set +u
source /project/biocomplexity/asw3xp/miniconda3/etc/profile.d/conda.sh
conda activate
set -u
set -euo pipefail

# --- Prepare Input and Output Paths ---
REPL="replicate_${REPLICATE_ID}"
REPDIR="${ROOT_DIR}/${REPL}"

LINELIST_FILE="${REPDIR}/linelist.csv.xz"
INFECTIONS_FILE="${REPDIR}/linelist_allevents.csv.xz"

# Create a distinct output directory for this specific run
# Example: .../scenario_runs/replicate_0/batch_750
OUTDIR="${SWEEP_OUTROOT}/${REPL}/batch_${BATCH_SIZE}"
mkdir -p "${OUTDIR}"

# --- Sanity Checks ---
if [[ ! -f "${LINELIST_FILE}" ]]; then
  echo "ERROR: Linelist file not found: ${LINELIST_FILE}" >&2; exit 2
fi
if [[ ! -f "${INFECTIONS_FILE}" ]]; then
  echo "ERROR: Infections file not found: ${INFECTIONS_FILE}" >&2; exit 2
fi

# Use a reproducible seed. Tying it to both replicate and batch size makes it unique.
SEED=42 #$((42 + REPLICATE_ID + BATCH_SIZE))

echo "--- Worker Details ---"
echo "  Replicate ID: ${REPLICATE_ID}"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Output Dir:   ${OUTDIR}"
echo "  Seed:         ${SEED}"
echo "--------------------"

# --- Define the command in an array for robustness ---
cmd=(
    "python3"
    "${PYTHON_SCRIPT}"
    "--linelist"
    "${LINELIST_FILE}"
    "--population"
    "${POP_FILE}"
    "--infections"
    "${INFECTIONS_FILE}"
    "--outdir"
    "${OUTDIR}"
    "--batch-size"
    "${BATCH_SIZE}" # <-- Use the batch size passed as an argument
    "--no-replacement"
    "--seed"
    "${SEED}"
)

# --- Echo and execute the command ---
echo "Executing command:"
printf "  %q " "${cmd[@]}"
echo ""

"${cmd[@]}"

echo "--- Worker finished successfully ---"
