"""
House Price Prediction Project
Professional Machine Learning Pipeline
"""

# ==============================
# Import Libraries
# ==============================

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OrdinalEncoder
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

# ==============================
# Load Dataset
# ==============================

train_df = pd.read_csv("data/train.csv")

# ==============================
# Features & Target
# ==============================

X = train_df.drop("SalePrice", axis=1)

y = train_df["SalePrice"]

# ==============================
# Numerical Columns
# ==============================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

# ==============================
# Categorical Columns
# ==============================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

# ==============================
# Numerical Pipeline
# ==============================

numerical_pipeline = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(strategy="median")

        )

    ]

)

# ==============================
# Categorical Pipeline
# ==============================

categorical_pipeline = Pipeline(

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

# ==============================
# Column Transformer
# ==============================

preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            numerical_pipeline,

            numerical_columns

        ),

        (

            "cat",

            categorical_pipeline,

            categorical_columns

        )

    ]

)

# ==============================
# Complete ML Pipeline
# ==============================

pipeline = Pipeline(

    steps=[

        (

            "preprocessor",

            preprocessor

        ),

        (

            "model",

            RandomForestRegressor(

                n_estimators=200,

                random_state=42,

                n_jobs=-1

            )

        )

    ]

)

# ==============================
# Train Validation Split
# ==============================

X_train, X_valid, y_train, y_valid = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

# ==============================
# Train Pipeline
# ==============================

pipeline.fit(

    X_train,

    y_train

)

# ==============================
# Prediction
# ==============================

predictions = pipeline.predict(

    X_valid

)

# ==============================
# Evaluation
# ==============================

mae = mean_absolute_error(

    y_valid,

    predictions

)

rmse = np.sqrt(

    mean_squared_error(

        y_valid,

        predictions

    )

)

r2 = r2_score(

    y_valid,

    predictions

)

print("\nPipeline Performance")

print("MAE :", round(mae,2))

print("RMSE :", round(rmse,2))

print("R2 Score :", round(r2,4))

# ==============================
# Save Pipeline
# ==============================

joblib.dump(

    pipeline,

    "models/house_price_pipeline.pkl"

)

print("\nPipeline Saved Successfully!")