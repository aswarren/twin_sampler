# Testing Propensity Simulation

This repo simulates a linelist of people who tested positive based on
demographic, socioeconomic, occupation, mobility (vehicles), RUCC (rural/urban),
and symptom status features.

## Scripts

- `rucc_utils.py` — loads and pivots the USDA RUCC "long" CSV into a wide table.
- `testing_prob.py` — row-level probability function with robust handling of missing values.
- `simulate_linelist.py` — CLI that merges inputs, computes probabilities, simulates positives, and writes a linelist CSV.
- `plot_age_race.py` - plot the output file

## Usage

```bash
python simulate_linelist.py \
  --people ../../va_persontrait_epihiper.txt \
  --infection ../../run_03_vadelta_rate_limited_ticks.metadata.fixed_dates.tsv \
  --households ../../va_household.csv \
  --rucc ../../Ruralurbancontinuumcodes2023.csv \
  --out simulated_test_positive_linelist.csv \
  --seed 42

# For plotting the result:
python plot_age_race.py --csv simulated_test_positive_linelist.csv