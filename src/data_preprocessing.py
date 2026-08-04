"""
House Price Prediction Project
Data Preprocessing
"""

# ==============================
# Import Libraries
# ==============================

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# ==============================
# Load Dataset
# ==============================

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# ==============================
# Store Test IDs
# ==============================

test_ids = test_df["Id"]

# ==============================
# Separate Target Variable


X = train_df.drop("SalePrice", axis=1)

y = train_df["SalePrice"]

# ==============================
# Combine Train & Test Data
# ==============================

combined_df = pd.concat([X, test_df], axis=0)

print("Combined Dataset Shape :", combined_df.shape)

# ==============================
# Separate Numerical Columns
# ==============================

numerical_columns = combined_df.select_dtypes(
    include=["int64", "float64"]
).columns

# ==============================
# Separate Categorical Columns
# ==============================

categorical_columns = combined_df.select_dtypes(
    include=["object"]
).columns

print("Numerical Features :", len(numerical_columns))
print("Categorical Features :", len(categorical_columns))

# ==============================
# Missing Value Handling
# Numerical Features
# ==============================

num_imputer = SimpleImputer(strategy="median")

combined_df[numerical_columns] = num_imputer.fit_transform(
    combined_df[numerical_columns]
)

# ==============================
# Missing Value Handling
# Categorical Features
# ==============================

cat_imputer = SimpleImputer(strategy="most_frequent")

combined_df[categorical_columns] = cat_imputer.fit_transform(
    combined_df[categorical_columns]
)

# ==============================
# Encode Categorical Features
# ==============================

encoder = OrdinalEncoder()

combined_df[categorical_columns] = encoder.fit_transform(
    combined_df[categorical_columns]
)

# ==============================
# Check Missing Values
# ==============================

print("\nMissing Values After Preprocessing")

print(combined_df.isnull().sum().sum())

# ==============================
# Split Back into Train & Test
# ==============================

X_train = combined_df.iloc[:len(train_df)]

X_test = combined_df.iloc[len(train_df):]

# ==============================
# Final Shapes
# ==============================

print("\nTraining Features :", X_train.shape)

print("Testing Features :", X_test.shape)

print("Target Shape :", y.shape)

# ==============================
# Save Preprocessed Data
# ==============================

X_train.to_csv(
    "data/X_train_processed.csv",
    index=False
)

X_test.to_csv(
    "data/X_test_processed.csv",
    index=False
)

y.to_csv(
    "data/y_train.csv",
    index=False
)

print("\nPreprocessed files saved successfully.")