"""
House Price Prediction Project
Production Prediction Script
"""

# ==============================
# Import Libraries
# ==============================

import pandas as pd
import joblib

# ==============================
# Load Saved Pipeline
# ==============================

pipeline = joblib.load(
    "models/house_price_pipeline.pkl"
)

print("Pipeline Loaded Successfully!")

# ==============================
# Load Test Dataset
# ==============================

test_df = pd.read_csv("data/test.csv")

print("Test Dataset Loaded Successfully!")

print("Test Shape :", test_df.shape)

# ==============================
# Store House IDs
# ==============================

house_ids = test_df["Id"]

# ==============================
# Predict House Prices
# ==============================

predictions = pipeline.predict(test_df)

print("Predictions Generated Successfully!")

# ==============================
# Create Submission File
# ==============================

submission = pd.DataFrame({

    "Id": house_ids,

    "SalePrice": predictions

})

# ==============================
# Preview Predictions
# ==============================

print("\nFirst Five Predictions\n")

print(submission.head())

# ==============================
# Save Submission
# ==============================

submission.to_csv(

    "outputs/submission.csv",

    index=False

)

print("\nSubmission Saved Successfully!")

print("Location : outputs/submission.csv")