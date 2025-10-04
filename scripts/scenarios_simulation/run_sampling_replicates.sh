#!/usr/bin/env bash
#
#SBATCH --job-name=linelist_sim
#SBATCH -p bii
#SBATCH -A nssac_students
#SBATCH -c 30
#SBATCH --mem=128G
#SBATCH -t 05:00:00
#SBATCH --array=0-19%10
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# ------------------ EDIT THESE PATHS ------------------
# Parent directory that contains replicate_0 ... replicate_19
ROOT_DIR="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/asw_test/twin_sampler/scripts/linelist_generation/results"
POP_FILE="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/va_persontrait_epihiper.txt"
OUTROOT="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/asw_test/twin_sampler/scripts/scenarios_simulation/scenario_runs"
# ------------------------------------------------------


echo "Activating Conda environment..."
source /project/biocomplexity/asw3xp/miniconda3/etc/profile.d/conda.sh
conda activate # <-- Assuming a base environment, or specify one like 'conda activate my_env'

RID="${SLURM_ARRAY_TASK_ID}"
REPL="replicate_${RID}"
REPDIR="${ROOT_DIR}/${REPL}"

LINELIST_FILE="${REPDIR}/linelist.csv.xz"
INFECTIONS_FILE="${REPDIR}/linelist_allevents.csv.xz"

# Fallbacks if filenames are longer like earlier examples
if [[ ! -f "${LINELIST_FILE}" ]]; then
  alt="$(ls -1 "${REPDIR}"/*linelist.csv.xz 2>/dev/null | head -n1 || true)"
  [[ -n "${alt}" ]] && LINELIST_FILE="${alt}"
fi
if [[ ! -f "${INFECTIONS_FILE}" ]]; then
  alt="$(ls -1 "${REPDIR}"/*linelist_allevents.csv.xz 2>/dev/null | head -n1 || true)"
  [[ -n "${alt}" ]] && INFECTIONS_FILE="${alt}"
fi

# Sanity checks
if [[ ! -f "${LINELIST_FILE}" ]]; then
  echo "ERROR: linelist not found in ${REPDIR}" >&2
  exit 2
fi
if [[ ! -f "${INFECTIONS_FILE}" ]]; then
  echo "ERROR: infections file not found in ${REPDIR}" >&2
  exit 2
fi
if [[ ! -f "${POP_FILE}" ]]; then
  echo "ERROR: population file not found at ${POP_FILE}" >&2
  exit 2
fi

OUTDIR="${OUTROOT}/${REPL}"
mkdir -p "${OUTDIR}"

# Distinct seed per replicate (use a fixed value like 42 if you prefer)
SEED=42

echo "Running ${REPL}"
echo "  linelist:    ${LINELIST_FILE}"
echo "  infections:  ${INFECTIONS_FILE}"
echo "  population:  ${POP_FILE}"
echo "  outdir:      ${OUTDIR}"
echo "  seed:        ${SEED}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

cmd=(
    "python3"
    "run_all_scenarios.py"
    "--linelist"
    "${LINELIST_FILE}"
    "--population"
    "${POP_FILE}"
    "--infections"
    "${INFECTIONS_FILE}"
    "--outdir"
    "${OUTDIR}"
    "--batch-size"
    "1000"
    "--no-replacement"
    "--seed"
    "${SEED}"
    "--save-samples"
)

# --- Echo the command for logging ---
# The 'printf "%q "' command prints each element of the array,
# adding quotes if necessary, so it's a perfect representation of the command.
echo "--- EXECUTING COMMAND ---"
printf "%q " "${cmd[@]}"
echo # Add a newline for cleaner log output
echo "---------------------------"


# --- Execute the command ---
# The "${cmd[@]}" syntax expands the array correctly.
"${cmd[@]}"

echo "Python script finished for $EPIHIPER_FILE."