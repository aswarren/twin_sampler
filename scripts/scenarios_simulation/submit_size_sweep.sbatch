#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=batch_sweep
#SBATCH -p bii
#SBATCH -A bii_nssac
#SBATCH -c 30
#SBATCH -t 05:00:00                # Adjust time as needed for a single run
#SBATCH --array=0-4                # Job array: 5 tasks, one for each batch size
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# --- Argument Check ---
if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 <replicate_id>"
    echo "Example: sbatch $0 0"
    exit 1
fi

REPLICATE_ID="$1"

# --- Script Configuration ---

# Array of batch sizes to be tested. The Slurm task ID will be the index.
BATCH_SIZES=(750 500 250 100 50)

# The path to your worker script
WORKER_SCRIPT="./size_worker.sh"


# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Target Replicate ID: $REPLICATE_ID"
echo "----------------------------------------------------"

# 1. Get the specific batch size for this task from the array
CURRENT_BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

# 2. Check if the batch size is valid
if [ -z "$CURRENT_BATCH_SIZE" ]; then
    echo "Error: Invalid Slurm Array Task ID. No batch size found at index ${SLURM_ARRAY_TASK_ID}."
    exit 1
fi

# 3. Execute the worker script, passing the dynamic parameters as arguments
#    Worker expects: <replicate_id> <batch_size>
echo "Starting worker for Replicate ${REPLICATE_ID} with Batch Size ${CURRENT_BATCH_SIZE}"

bash "$WORKER_SCRIPT" "$REPLICATE_ID" "$CURRENT_BATCH_SIZE"

echo "Worker script finished for Batch Size ${CURRENT_BATCH_SIZE}"
