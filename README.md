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
- Handling of invalid negative weight values

## Dataset

The labeled development dataset contains:

- **48,000 loads**
- **January 2025 – October 2025**
- **14 columns**
- Target: `posted_rate`

The supplied validation dataset contains:

- **12,000 loads**
- **November 2025 – December 2025**

The assessment also provides a fixed December 2025 scenario for daily predictions.

## Data Quality

The exploratory analysis identified:

- 300 missing weight values in the training data
- 374 missing `market_index` values
- 292 negative weight values
- 145 negative weight values in the validation data
- A small number of extreme high-rate observations

Negative weight values were treated as missing rather than valid measurements.

## Validation Strategy

A chronological split was used instead of a random split to better represent future prediction.

### September Holdout

**Training:** January – August 2025

**Validation:** September 2025

| Metric | Result |
|---|---:|
| MAE | **$106.42** |
| RMSE | **$616.96** |

### October Holdout

**Training:** January – September 2025

**Validation:** October 2025

| Metric | Result |
|---|---:|
| MAE | **$106.04** |
| RMSE | **$646.81** |

These are internal chronological holdout results. The final hidden validation metrics are calculated by Spotter after submission.

## Feature Engineering

The final model uses:

- Pickup
- Delivery
- Pickup latitude
- Pickup longitude
- Delivery latitude
- Delivery longitude
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

```

Predictions are converted back to the original dollar scale using:

```text
expm1(prediction)
```

### Final Configuration

| Parameter | Value |
|---|---:|
| Model | CatBoostRegressor |
| Target transformation | `log1p(posted_rate)` |
| Iterations | 300 |
| Depth | 8 |
| Learning rate | 0.04 |
| L2 regularization | 5 |
| Random seed | 42 |

The final model was retrained using all 48,000 labeled observations before generating the final validation predictions.

## Model Comparison

| Model | October MAE | October RMSE |
|---|---:|---:|
| Random Forest baseline | $159.87 | $689.32 |
| CatBoost | $115.54 | $651.06 |
| Feature-engineered CatBoost | $107.32 | $646.50 |
| Log-target CatBoost | $107.12 | $646.22 |
| Log-target CatBoost without route | $106.39 | $646.00 |
| Final feature set | **$106.04** | $646.81 |

The final feature set was selected based on chronological validation performance rather than random cross-validation.

## Final Predictions

The final model generates:

`validation_predictions.csv`

with the required format:

`load_id,predicted_rate`

The final prediction file contains:

- 12,000 predictions
- 12,000 unique load IDs
- Positive predicted rates

The supplied `score.py` successfully validated all 12,000 final predictions.

The supplied scorer also successfully validated the 31 December predictions.

## December 2025 Scenario

The fixed December scenario uses:

- **Pickup:** Lexington
- **Delivery:** Fort Wayne
- **Distance:** 360 miles
- **Equipment:** Dry Van
- **Weight:** 32,000 lb
- **Date:** December 1–31, 2025

Only the date changes between the 31 predictions.

The generated December predictions ranged from approximately **$801 to $827**, with an average of approximately **$815.54**.

Because `market_index` and `quote_signal` are not provided in the December input file, training-data median values were used for those unavailable features.

## Running the Project

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the final model and generate predictions

```bash
python final_model.py
```

This generates the final validation predictions and December predictions.

### 5. Validate the outputs

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer generates:

`scorer_results/candidate_december.png`

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── check_data.py
├── final_model.py
├── score.py
└── .gitignore
```

## Repository Notes

The assessment datasets and locally generated prediction files are intentionally excluded from the public repository.

The repository contains the core modeling code, validation/scoring code, dependencies, and documentation required to understand and reproduce the solution when the assessment data is available.

## Limitations

- The hidden November–December target values are unavailable, so the final hidden-test score cannot be measured before submission.
- Rare extreme-rate observations remain more difficult to predict.
- The December scenario does not provide `market_index` or `quote_signal`, so training-data median values were used for those features.

## Conclusion

The final solution combines:

- Time-aware validation
- Freight-specific feature engineering
- Explicit handling of invalid weight values
- Log-target regression
- CatBoost

The model achieved approximately **$106 MAE** on both September and October chronological holdouts.

The final prediction files also passed the provided Spotter scorer successfully.