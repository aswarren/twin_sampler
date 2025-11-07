#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=label_components   # A name for your job
#SBATCH -p bii                        # Partition (queue) name
#SBATCH -A bii_nssac                  # Account name
#SBATCH -c 30                         # Number of CPU cores per task
#SBATCH --mem=128G                    # Memory per node (adjust as needed)
#SBATCH -t 05:00:00                   # Run time (hh:mm:ss)
#SBATCH --array=0-19%10               # Job array: 20 tasks (0-19), 10 running at a time
#SBATCH --output=slurm_logs/label_components_%A_%a.out # Standard output log
#SBATCH --error=slurm_logs/label_components_%A_%a.err  # Standard error log

# --- Script Configuration ---

# The base directory where all the replicate folders are located
BASE_INPUT_DIR="/project/bii_nssac/epihiper-simulations/pipeline-jc/run/20250120_1/output_root/proj/20250120_1/batch_1/0.25/va"

# The base directory where all output will be saved
BASE_OUTPUT_DIR="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/asw_test/twin_sampler/scripts/linelist_generation/labeled_results" # Changed to a new directory

# The path to the importation schedule CSV file.
# This file is STATIC and used by all replicates.
SCHEDULE_FILE="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/asw_test/twin_sampler/Data/Virginia_importation_schedule.csv" # <-- IMPORTANT: SET THIS PATH

# The path to your worker script
WORKER_SCRIPT="./run_label_components.sh"


# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID" # This is the replicate number
echo "----------------------------------------------------"

# 1. Construct the specific input file path for this task
# The input file is expected to be named output.csv.gz inside the replicate folder
INPUT_REPLICATE_DIR="${BASE_INPUT_DIR}/replicate_${SLURM_ARRAY_TASK_ID}"
EPIHIPER_FILE="${INPUT_REPLICATE_DIR}/output.csv.gz"

# 2. Construct the specific output directory and file path for this task
OUTPUT_REPLICATE_DIR="${BASE_OUTPUT_DIR}/replicate_${SLURM_ARRAY_TASK_ID}"
OUTPUT_FILE="${OUTPUT_REPLICATE_DIR}/labeled_epihiper_output.csv" # A more descriptive output name

# 3. Create the output directory for this specific task
# The -p flag ensures it doesn't fail if the directory already exists
mkdir -p "$OUTPUT_REPLICATE_DIR"

# 4. Check if the input files exist before launching the worker
if [ ! -f "$EPIHIPER_FILE" ]; then
  echo "Error: Input EpiHiper file not found for replicate ${SLURM_ARRAY_TASK_ID}: $EPIHIPER_FILE"
  exit 1
fi
if [ ! -f "$SCHEDULE_FILE" ]; then
  echo "Error: Static schedule file not found: $SCHEDULE_FILE"
  exit 1
fi

# 5. Execute the worker script, passing the dynamic and static paths as arguments
echo "Starting worker for replicate ${SLURM_ARRAY_TASK_ID}"
echo "Input EpiHiper file: $EPIHIPER_FILE"
echo "Input Schedule file: $SCHEDULE_FILE"
echo "Output file path: $OUTPUT_FILE"

bash "$WORKER_SCRIPT" "$EPIHIPER_FILE" "$SCHEDULE_FILE" "$OUTPUT_FILE"

echo "Worker script finished for replicate ${SLURM_ARRAY_TASK_ID}"
