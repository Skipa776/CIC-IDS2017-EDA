# Archived notebooks

Earlier analysis, kept so older reports stay traceable. The numbered notebooks in `notebooks/`
replace them. Their relative paths (`../data`, `../reports`) assumed `notebooks/`, so run them
from there if you need to re-execute one.

| Notebook | Why it was archived | Replaced by |
| --- | --- | --- |
| `cicids2017_eda.ipynb` | Mixes EDA, cleaning and random-split baselines | `00_data_and_cleaning`, `01_eda`, `02_why_random_splits_mislead` |
| `attack_types.ipynb` | Feature importance fit on all data before its split; kept as the multiclass (Layer 2) appendix | `02` explains the issue |
| `single_day_eda.ipynb` | Single train/test day pair, one seed, Isolation Forest at 200 trees | `03_forward_2017` |
| `cross_year_diagnostics.ipynb` | Its findings are summarized in the cross-year notebook | `06_cross_year` |
| `train_2017_test_2018.ipynb`, `train_2018_test_2017.ipynb` | Thresholds set on the test year's own benign traffic (retrospective, not deployable) | `06_cross_year` |
| `day_of_the_weeks.ipynb` | Unfinished stub | `01_eda` |
