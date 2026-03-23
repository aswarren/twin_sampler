# Scenario Sampling Runner

This repo provides code to reproduce the evaluation of different sampling algorithms across eight scenarios (Table 3 in the paper).  
It loads a synthetic linelist and population file, runs all 8 scenarios in one go, and outputs:

- Per-scenario CSVs of KL divergence values (one row per week, per algorithm).
- Three final plots (one per algorithm) comparing all 8 scenarios.
- Average running time (seconds) across the 8 scenarios for each algorithm.

---

## Requirements

- Python 3.8+
- Dependencies:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`


# Run similation with:
```bash
python3 run_all_scenarios.py \
  --linelist ../data/linelist.csv.xz \
  --population ../../va_persontrait_epihiper.txt \
  --infections ../data/linelist_allevents.csv.xz \
  --outdir ./result \
  --batch-size 100 \
  --no-replacement \
  --seed 42 \
  --algorithms "surs", "stratified", "LASSO-Greedy", "LASSO-Stratified" \
  --stratifiers "age", "race", "county", "sex" \
  --save-samples
```

# Run replicates test with:
```bash
python3 run_replicates.py \
  --replicates-dir ../data/replicate \
  --population ../../va_persontrait_epihiper.txt \
  --no-plots

python3 plot_kl_uncertainty.py --root scenario_runs --outdir result_graphs
```

# Run lite version with:
```bash
python3 run_sampling_lite.py \
  --linelist ../data/sliced_linelist/batch_week_4_ends_2021-06-27.csv \
  --population ../../va_persontrait_epihiper.txt \
  --history ./sampling_result_lite \
  --outdir ./sampling_result_lite \
  --batch-size 100 \
  --no-replacement \
  --seed 42 \
  --algorithms "surs" \
  --stratifiers "age", "race", "county", "sex"
```