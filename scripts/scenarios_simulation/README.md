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
python run_all_scenarios.py \
  --linelist ../linelist/simulated_test_positive_linelist.csv \
  --population ../../va_persontrait_epihiper.txt \
  --infections ../../run_03_vadelta_rate_limited_ticks.metadata.fixed_dates.tsv \
  --outdir ./result \
  --batch-size 1000 \
  --no-replacement \
  --seed 42