import pandas as pd
import numpy as np

from catboost import CatBoostRegressor


# ==========================================
# 1. LOAD TRAINING AND VALIDATION DATA
# ==========================================

train_df = pd.read_csv("data/train-test.csv")
validation_df = pd.read_csv("data/validation.csv")

train_df["date"] = pd.to_datetime(train_df["date"])
validation_df["date"] = pd.to_datetime(validation_df["date"])


# ==========================================
# 2. FEATURE ENGINEERING FUNCTION
# ==========================================

def create_features(df):
    df = df.copy()

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["lat_difference"] = (
        df["pickup_lat"] - df["delivery_lat"]
    ).abs()

    df["lon_difference"] = (
        df["pickup_lon"] - df["delivery_lon"]
    ).abs()

    df["weight_per_mile"] = (
        df["weight"] / df["distance"]
    )

    df["distance_log"] = np.log1p(
        df["distance"]
    )

    return df


train_df = create_features(train_df)
validation_df = create_features(validation_df)


# ==========================================
# 3. FEATURE LIST
# ==========================================

features = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "distance_log",
    "equipment",
    "weight",
    "weight_per_mile",
    "month",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "is_weekend",
]


# ==========================================
# 4. PREPARE DATA
# ==========================================

X_train = train_df[features].copy()
X_validation = validation_df[features].copy()

y_train = train_df["posted_rate"]


# ==========================================
# 5. HANDLE INVALID WEIGHTS
# ==========================================

for frame in [X_train, X_validation]:

    frame.loc[
        frame["weight"] < 0,
        "weight"
    ] = np.nan

    frame.loc[
        frame["weight_per_mile"] < 0,
        "weight_per_mile"
    ] = np.nan


# ==========================================
# 6. LOG TARGET
# ==========================================

y_train_log = np.log1p(y_train)


# ==========================================
# 7. CATEGORICAL FEATURES
# ==========================================

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
]

categorical_indices = [
    features.index(column)
    for column in categorical_features
]


# ==========================================
# 8. FINAL MODEL
# ==========================================

model = CatBoostRegressor(
    iterations=300,
    depth=8,
    learning_rate=0.04,
    loss_function="RMSE",
    random_seed=42,
    l2_leaf_reg=5,
    verbose=50,
)


# ==========================================
# 9. TRAIN ON ALL 48,000 ROWS
# ==========================================

print("Training final model on all labeled data...")
print("Training rows:", len(X_train))

model.fit(
    X_train,
    y_train_log,
    cat_features=categorical_indices,
)

print("Final model training complete.")


# ==========================================
# 10. PREDICT 12,000 VALIDATION LOADS
# ==========================================

print("Predicting validation data...")

validation_predictions_log = model.predict(
    X_validation
)

validation_predictions = np.expm1(
    validation_predictions_log
)

validation_predictions = np.maximum(
    validation_predictions,
    0.01
)


# ==========================================
# 11. CREATE SUBMISSION FILE
# ==========================================

submission = pd.DataFrame({
    "load_id": validation_df["load_id"],
    "predicted_rate": validation_predictions,
})


# ==========================================
# 12. VERIFY SUBMISSION
# ==========================================

print()
print("========== FINAL PREDICTIONS ==========")
print("Rows:", len(submission))
print("Unique IDs:", submission["load_id"].nunique())
print(
    "Minimum prediction:",
    submission["predicted_rate"].min()
)
print(
    "Maximum prediction:",
    submission["predicted_rate"].max()
)
print(
    "Average prediction:",
    submission["predicted_rate"].mean()
)

print()
print("First 5 predictions:")
print(submission.head().to_string(index=False))


# ==========================================
# 13. SAVE
# ==========================================

output_path = "validation_predictions.csv"

submission.to_csv(
    output_path,
    index=False
)

print()
print("Saved:", output_path)

# ==========================================
# 14. DECEMBER PREDICTIONS
# ==========================================

print()
print("Generating December predictions...")

december = pd.read_csv(
    "data/december-chart-inputs.csv"
)

december["date"] = pd.to_datetime(
    december["date"]
)

# Get the known coordinates for the fixed
# Lexington -> Fort Wayne route from training data.

lexington = train_df[
    train_df["pickup"] == "Lexington"
].iloc[0]

fort_wayne = train_df[
    train_df["delivery"] == "Fort Wayne"
].iloc[0]

# Add the missing model input columns.
december["pickup_lat"] = lexington["pickup_lat"]
december["pickup_lon"] = lexington["pickup_lon"]

december["delivery_lat"] = fort_wayne["delivery_lat"]
december["delivery_lon"] = fort_wayne["delivery_lon"]

# The December scenario does not provide
# market_index or quote_signal.
#
# Use the training-data median for these features.
december["market_index"] = train_df["market_index"].median()
december["quote_signal"] = train_df["quote_signal"].median()

# Create the same engineered features
# used by the final model.
december_features = create_features(december)

X_december = december_features[features].copy()

# Handle invalid weights
X_december.loc[
    X_december["weight"] < 0,
    "weight"
] = np.nan

X_december.loc[
    X_december["weight_per_mile"] < 0,
    "weight_per_mile"
] = np.nan

# Predict
december_predictions_log = model.predict(
    X_december
)

december_predictions = np.expm1(
    december_predictions_log
)

december_predictions = np.maximum(
    december_predictions,
    0.01
)

# Fill required column
december["predicted_rate"] = december_predictions

# Keep ONLY the seven columns required
# by Spotter, in the exact required order.
december = december[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
]

# Save
december.to_csv(
    "data/december-chart-inputs.csv",
    index=False
)

print("December rows:", len(december))
print(
    "December minimum:",
    december["predicted_rate"].min()
)
print(
    "December maximum:",
    december["predicted_rate"].max()
)
print(
    "December average:",
    december["predicted_rate"].mean()
)

print()
print("December predictions saved.")