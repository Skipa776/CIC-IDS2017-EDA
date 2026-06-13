# Skill: Complete Analysis Notebooks

## Description
This skill guides Claude in completing the two analysis notebooks specified in Phase 1 of the PLAN.md: attack_types.ipynb and day_of_the_weeks.ipynb. It involves data visualization, feature analysis, and temporal insights using libraries like pandas, matplotlib, seaborn, plotly, and umap-learn. Outputs include charts, tables, and insights for model design.

## Instructions for Claude
You are a code assistant implementing Phase 1 of the CICIDS2017 IDS plan. Use your standard tools (file read/write, bash, glob, grep) to create and modify files in the project directory.

First, verify the data file exists: Use grep or file read to check for 'data/processed/cicids2017_clean.parquet'. If missing, notify the user.

For notebooks/attack_types.ipynb:
- Load data using pandas.read_parquet.
- Implement sections:
  1. Attack distribution bar chart with sample counts (use seaborn or matplotlib).
  2. Feature distributions by attack (box plots for: Flow Duration, Flow Bytes/s, Total Fwd/Bwd Packets, Packet Length Mean/Std).
  3. Attack signatures: Top 5 discriminative features per attack via Random Forest importance (from scikit-learn).
  4. Confusion matrix showing misclassifications (train a simple RF model for this).
  5. UMAP projection colored by attack type (use umap-learn).
  6. Per-attack precision/recall/F1 table (use cross-validation).
- Save the notebook with executed cells and export insights as a markdown summary in notebooks/attack_types_summary.md.

For notebooks/day_of_the_weeks.ipynb:
- Map scenarios: Create a table (Meta_source → Day → Attack types).
- Stacked bar chart: Benign vs attack counts per day.
- Benign traffic comparison: Feature stability across days.
- Temporal train/test split: Mon-Thu train, Fri test; evaluate a simple model.
- Feature drift analysis: Identify stable vs day-dependent features.
- Save the notebook and export insights to notebooks/day_of_the_weeks_summary.md.

Use dependencies from requirements.txt. If errors occur, debug and suggest fixes. Output the completed file paths and key insights.

## Dependencies
- pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, umap-learn, pyarrow
- Data file: data/processed/cicids2017_clean.parquet