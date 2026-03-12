#!/usr/bin/env bash
#
# worker_calc_re.sh
# Usage: bash worker_calc_re.sh <replicate_id>

# Fail on error, unset variable, or pipe failure
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <replicate_id>"
    exit 1
fi

REPLICATE_ID="$1"
REPL="replicate_${REPLICATE_ID}"

# ------------------ EDIT THESE PATHS ------------------
# Path to the raw epihiper outputs
BASE_INPUT_DIR="/project/bii_nssac/epihiper-simulations/pipeline-jc/run/20250120_1/output_root/proj/20250120_1/batch_1/0.25/va"

# Where you want the Re CSVs to be saved
BASE_OUTPUT_DIR="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/asw_test/twin_sampler/scripts/re_calculations"
# ------------------------------------------------------

# Define specific files
EPIHIPER_FILE="${BASE_INPUT_DIR}/${REPL}/output.csv.gz"
OUTDIR="${BASE_OUTPUT_DIR}/${REPL}"
OUTFILE="${OUTDIR}/Re_over_time.csv"

# Sanity check
if [[ ! -f "${EPIHIPER_FILE}" ]]; then
  echo "ERROR: EpiHiper file not found: ${EPIHIPER_FILE}" >&2
  exit 2
fi

mkdir -p "${OUTDIR}"

echo "Activating Conda environment..."
set +u # Temporarily disable strict unbound variable checking for Conda
source /project/biocomplexity/asw3xp/miniconda3/etc/profile.d/conda.sh
conda activate
set -u # Re-enable strict checking

# Define command in an array
cmd=(
    "python"
    "calc_re.py"
    "--epihiper"
    "${EPIHIPER_FILE}"
    "--out"
    "${OUTFILE}"
)

echo "--- EXECUTING COMMAND FOR ${REPL} ---"
printf "%q " "${cmd[@]}"
echo ""
echo "---------------------------------------"

# Execute
"${cmd[@]}"

echo "--- Finished calculating Re for ${REPL} ---"
