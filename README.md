# Freight Rate Prediction — Machine Learning Assessment

Machine learning solution for predicting freight load rates using historical shipment data.

## Overview

This project builds a freight-rate prediction model using historical load data and generates predictions for future validation loads.

The final approach uses:

- CatBoost Regression
- Log-transformed target (`log1p(posted_rate)`)
- Time-based validation
- Categorical features for pickup, delivery, and equipment
- Freight-specific feature engineering
- Explicit handling of invalid negative weight values

## Dataset

The labeled development dataset contains:

- 48,000 loads
- January 2025 – October 2025
- 14 columns
- Target: `posted_rate`

The supplied validation dataset contains:

- 12,000 loads
- November 2025 – December 2025

The assessment also provides a fixed December 2025 scenario for daily predictions.

## Data Quality

The exploratory analysis identified:

- Missing weight values
- Missing market index values
- Invalid negative weight values
- A small number of extreme high-rate observations

Negative weight values were treated as missing rather than valid measurements.

## Validation Strategy

A chronological split was used instead of a random split to better represent future prediction.

### September Holdout

Training:

January – August 2025

Validation:

September 2025

Results:

- MAE: **$106.42**
- RMSE: **$616.96**

### October Holdout

Training:

January – September 2025

Validation:

October 2025

Results:

- MAE: **$106.04**
- RMSE: **$646.81**

These are internal holdout results. The final hidden validation metrics are calculated by Spotter after submission.

## Feature Engineering

The final model uses:

- Pickup
- Delivery
- Pickup latitude/longitude
- Delivery latitude/longitude
- Distance
- Log-transformed distance
- Equipment
- Weight
- Weight per mile
- Month
- Day of week
- Day of month
- Week of year
- Weekend indicator
- Absolute latitude difference
- Absolute longitude difference

Several feature-selection experiments were performed.

The manually constructed route feature, `market_index`, and `quote_signal` were removed from the final feature set after controlled validation experiments.

## Model

The final model is a `CatBoostRegressor` trained on:

```text
log1p(posted_rate)

Predictions are converted back to dollar values using:
expm1(prediction)

Final configuration:

Iterations: 300
Depth: 8
Learning rate: 0.04
L2 regularization: 5
Random seed: 42

The final model was retrained using all 48,000 labeled observations before generating the final validation predictions.

Model Comparison
Model	October MAE	October RMSE
Random Forest baseline	$159.87	$689.32
CatBoost	$115.54	$651.06
Feature-engineered CatBoost	$107.32	$646.50
Log-target CatBoost	$107.12	$646.22
Log-target CatBoost without route	$106.39	$646.00
Final feature set	$106.04	$646.81
Final Outputs

The final model generates:

validation_predictions.csv

with the required format:

load_id,predicted_rate

The provided scorer validates:

12,000 final predictions
Unique validation IDs
Positive predicted rates
31 December predictions
Required December input format

The final files successfully passed the supplied score.py validation.

Running the Project
1. Create a virtual environment
python -m venv .venv
2. Activate the environment

Windows PowerShell:

.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Train the final model and generate predictions
python final_model.py

This generates:

validation_predictions.csv

and completes the December prediction file.

5. Validate the outputs
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv

The scorer generates:

scorer_results/candidate_december.png
Project Structure
.
├── README.md
├── requirements.txt
├── check_data.py
├── final_model.py
├── score.py
└── .gitignore
Notes

The assessment data files are intentionally excluded from this repository.

The December scenario does not provide market_index or quote_signal, so training-data median values were used for those unavailable inputs when generating the December predictions.

Conclusion

The final solution uses time-aware validation and a log-target CatBoost model with freight-specific feature engineering.

It achieved approximately $106 MAE on both September and October chronological holdouts while satisfying the supplied output validation requirements.