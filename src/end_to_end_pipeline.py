"""
House Price Prediction
End-to-End ML Pipeline
"""

import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

# ==========================
# Load Dataset
# ==========================

train_df = pd.read_csv("data/train.csv")

X = train_df.drop("SalePrice", axis=1)

y = train_df["SalePrice"]

# ==========================
# Custom Feature Engineering
# ==========================

class FeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["HouseAge"] = 2026 - X["YearBuilt"]

        X["YearsSinceRemodel"] = 2026 - X["YearRemodAdd"]

        X["TotalBathrooms"] = (
            X["FullBath"]
            + 0.5 * X["HalfBath"]
            + X["BsmtFullBath"]
            + 0.5 * X["BsmtHalfBath"]
        )

        X["TotalArea"] = (
            X["GrLivArea"]
            + X["TotalBsmtSF"]
        )

        X["HasGarage"] = (
            X["GarageArea"] > 0
        ).astype(int)

        return X

# ==========================
# Apply Feature Engineering
# ==========================

engineer = FeatureEngineering()

X = engineer.fit_transform(X)

# ==========================
# Column Names
# ==========================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

# ==========================
# Numerical Pipeline
# ==========================

num_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        )

    ]

)

# ==========================
# Categorical Pipeline
# ==========================

cat_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )

    ]

)

# ==========================
# Column Transformer
# ==========================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            num_pipeline,
            numerical_columns
        ),

        (
            "cat",
            cat_pipeline,
            categorical_columns
        )

    ]

)

# ==========================
# Complete Pipeline
# ==========================

pipeline = Pipeline(

    steps=[

        (
            "feature_engineering",
            engineer
        ),

        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42
            )
        )

    ]

)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    train_df.drop("SalePrice", axis=1),

    y,

    test_size=0.20,

    random_state=42

)

# ==========================
# Train Pipeline
# ==========================

pipeline.fit(
    X_train,
    y_train
)

# ==========================
# Prediction
# ==========================

prediction = pipeline.predict(X_test)

# ==========================
# Evaluation
# ==========================

mae = mean_absolute_error(y_test, prediction)

rmse = np.sqrt(mean_squared_error(y_test, prediction))

r2 = r2_score(y_test, prediction)

print("\nPipeline Performance")

print("MAE :", round(mae,2))

print("RMSE :", round(rmse,2))

print("R2 :", round(r2,4))

# ==========================
# Save Pipeline
# ==========================

joblib.dump(

    pipeline,

    "models/end_to_end_pipeline.pkl"

)

print("\nPipeline Saved Successfully!")
