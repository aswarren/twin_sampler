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
  --linelist epihiper_batch_1_0.25_replicate_0.output.csv.gz_linelist.csv.xz \
  --population ../../va_persontrait_epihiper.txt \
  --infections epihiper_batch_1_0.25_replicate_0.output.gz_linelist_allevents.csv.xz \
  --outdir ./result \
  --batch-size 1000 \
  --no-replacement \
  --seed 42