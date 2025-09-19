# Logistic Regression Analysis on CSV Dataset

## Author
Ved Parkash
IIT Patna | Advance Pattern Recognition  
Date: 19-Sep-2025  

---

## Overview
This project applies a logistic regression model to predict whether the `VALUE` column of a record is above or below the median.

Workflow includes:
- Data preprocessing and cleaning
- One-hot encoding for categorical variables
- Standardizing numerical features
- Training logistic regression
- Evaluating performance with metrics and visualizations

---

## Requirements
Python 3.8+ and the following packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install packages using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

## Dataset
- CSV file: `37100106.csv`
- Predictors: `VALUE`, `GEO`, `Age group`, `Type of institution attended`, `REF_DATE`
- Target: `target_bin` (1 if VALUE > median, 0 otherwise)

## How to Run
1. Place `37100106.csv` in the same folder as the script.  
2. Run the Python script:

```bash
python logistic_regression_analysis.py



## Outputs
- Accuracy and ROC AUC in terminal
- Confusion matrix and classification report
- Feature importance coefficients

**Plots saved automatically:**
- `confusion_matrix.png`
- `classification_report.png`
- `roc_curve.png`

## Features
- Missing numeric values replaced with median
- Missing categorical values replaced with 'MISSING'
- One-hot encoding for categorical variables
- Logistic Regression (`liblinear`, `max_iter=1000`)
- Metrics: Accuracy, ROC AUC, confusion matrix, classification report
- Visualizations: Confusion matrix, classification report heatmap, ROC curve

## Notes
- Ensure the CSV file has all required columns.
