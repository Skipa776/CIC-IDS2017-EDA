# CICIDS-EDA

This project is an Exploratory Data Analysis (EDA) of the CICIDS2017 dataset, a widely used dataset for network intrusion detection research.

## Project Structure

```
cicids-eda/
│
├── data/
│   ├── interim/         # Intermediate data that has been transformed
│   ├── processed/       # Final data sets for modeling or analysis
│   └── raw/             # Original, immutable data dump
│       └── MachineLearningCVE/
│           ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│           ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│           ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│           ├── Monday-WorkingHours.pcap_ISCX.csv
│           ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│           ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│           ├── Tuesday-WorkingHours.pcap_ISCX.csv
│           └── Wednesday-workingHours.pcap_ISCX.csv
│
├── notebooks/
│   └── 01_eda.ipynb     # Jupyter notebook for EDA
│
└── README.md            # Project documentation
```

## Data
- **data/raw/MachineLearningCVE/**: Contains the original CSV files from the CICIDS2017 dataset, each representing network traffic for different days and attack scenarios.
- **data/interim/**: Intended for storing intermediate data files generated during processing.
- **data/processed/**: Intended for storing processed data ready for analysis or modeling.

## Notebooks
- **notebooks/01_eda.ipynb**: Main notebook for performing exploratory data analysis on the CICIDS2017 dataset.

## Getting Started
1. Place the original CICIDS2017 CSV files in `data/raw/MachineLearningCVE/` if not already present.
2. Open and run the EDA notebook in the `notebooks/` directory.

## About CICIDS2017
The CICIDS2017 dataset is designed for evaluating intrusion detection systems and includes benign and malicious traffic, with labels for various attack types.

## License
Specify your license here (e.g., MIT, Apache 2.0, etc.).
