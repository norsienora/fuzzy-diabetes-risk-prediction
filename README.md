# Dataset Instructions

Download the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and place this file in the current directory:

```text
diabetes_binary_5050split_health_indicators_BRFSS2015.csv
```

Expected properties:

- 70,692 data rows, plus one header row
- 22 columns in total
- Balanced target distribution: 35,346 non-diabetes and 35,346 diabetes observations
- Target column: `Diabetes_binary`

CSV files are excluded from Git through `.gitignore` to keep the repository lightweight and to make the dataset source explicit.
