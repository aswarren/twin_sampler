#!/usr/bin/env bash
#
# submit_calc_re.sbatch
#
#SBATCH --job-name=calc_re
#SBATCH -p bii
#SBATCH -A bii_nssac
#SBATCH -c 4                       # Only 4 cores needed for this operation
#SBATCH --mem=64G                  # 64GB is plenty for graph creation/grouping
#SBATCH -t 01:00:00                # Should be very fast (under an hour)
#SBATCH --array=0-19%10            # Process replicates 0 through 19, 10 at a time
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# Create log directory if it doesn't exist
mkdir -p slurm_logs

echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID (Replicate): $SLURM_ARRAY_TASK_ID"
echo "----------------------------------------------------"

# Path to the worker script
WORKER_SCRIPT="./worker_calc_re.sh"

# Execute the worker script, passing the array task ID as the replicate ID
bash "$WORKER_SCRIPT" "$SLURM_ARRAY_TASK_ID"

