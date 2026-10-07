"""
Capstone Project - Part 2: Regression Model to Predict Sales
Uses the cleaned dataset from Part 1 (cleaned_sales_data.csv)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("cleaned_sales_data.csv")

# -----------------------------------------------------------------
# Choose features and target
# We predict 'Sales' using the business drivers available BEFORE a
# sale is finalized: units, pricing, discount info, and categorical
# descriptors (segment, country, product, discount band, month).
# -----------------------------------------------------------------
features = [
    "Segment", "Country", "Product", "Discount Band",
    "Units Sold", "Manufacturing Price", "Sale Price",
    "Discounts", "Month Number"
]
target = "Sales"

X = df[features]
y = df[target]

categorical_cols = ["Segment", "Country", "Product", "Discount Band"]
numeric_cols = ["Units Sold", "Manufacturing Price", "Sale Price", "Discounts", "Month Number"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough"  # keep numeric columns as-is
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

# -----------------------------------------------------------------
# Train / test split
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=" * 60)
print("Regression Model Performance (predicting Sales)")
print("=" * 60)
print(f"R^2 score:  {r2:.4f}  (explains {r2*100:.1f}% of variance in Sales)")
print(f"MAE:        ${mae:,.2f}")
print(f"RMSE:       ${rmse:,.2f}")
print(f"Avg Sales:  ${y.mean():,.2f}  (for context)")
print()

# -----------------------------------------------------------------
# Show a few predictions vs actuals
# -----------------------------------------------------------------
comparison = pd.DataFrame({
    "Actual Sales": y_test.values[:10],
    "Predicted Sales": y_pred[:10].round(2)
})
print("Sample predictions vs actuals:")
print(comparison, "\n")

# -----------------------------------------------------------------
# Feature importance (coefficients) for the numeric features
# -----------------------------------------------------------------
feature_names = model.named_steps["preprocess"].get_feature_names_out()
coefs = model.named_steps["regressor"].coef_
coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
coef_df["abs_coef"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

print("=" * 60)
print("Top 10 features driving Sales predictions")
print("=" * 60)
print(coef_df.head(10).to_string(index=False))

# Save predictions for the report
results = X_test.copy()
results["Actual Sales"] = y_test.values
results["Predicted Sales"] = y_pred.round(2)
results.to_csv("regression_predictions.csv", index=False)
print("\nSaved regression_predictions.csv")
