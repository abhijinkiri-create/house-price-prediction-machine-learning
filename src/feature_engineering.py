"""
House Price Prediction
Feature Engineering
"""

# ==========================
# Import Libraries
# ==========================

import pandas as pd

# ==========================
# Load Dataset
# ==========================

train_df = pd.read_csv("data/train.csv")

test_df = pd.read_csv("data/test.csv")

# ==========================
# Function
# ==========================

def create_features(df):

    # ----------------------
    # House Age
    # ----------------------

    df["HouseAge"] = 2026 - df["YearBuilt"]

    # ----------------------
    # Years Since Remodel
    # ----------------------

    df["YearsSinceRemodel"] = 2026 - df["YearRemodAdd"]

    # ----------------------
    # Total Bathrooms
    # ----------------------

    df["TotalBathrooms"] = (

        df["FullBath"]

        +

        0.5 * df["HalfBath"]

        +

        df["BsmtFullBath"]

        +

        0.5 * df["BsmtHalfBath"]

    )

    # ----------------------
    # Total Porch Area
    # ----------------------

    df["TotalPorchArea"] = (

        df["OpenPorchSF"]

        +

        df["EnclosedPorch"]

        +

        df["3SsnPorch"]

        +

        df["ScreenPorch"]

    )

    # ----------------------
    # Total Living Area
    # ----------------------

    df["TotalArea"] = (

        df["GrLivArea"]

        +

        df["TotalBsmtSF"]

    )

    # ----------------------
    # Total Rooms
    # ----------------------

    df["TotalRooms"] = (

        df["TotRmsAbvGrd"]

        +

        df["BedroomAbvGr"]

    )

    # ----------------------
    # Has Garage
    # ----------------------

    df["HasGarage"] = (

        df["GarageArea"] > 0

    ).astype(int)

    # ----------------------
    # Has Basement
    # ----------------------

    df["HasBasement"] = (

        df["TotalBsmtSF"] > 0

    ).astype(int)

    # ----------------------
    # Has Second Floor
    # ----------------------

    df["HasSecondFloor"] = (

        df["2ndFlrSF"] > 0

    ).astype(int)

    return df

# ==========================
# Apply Feature Engineering
# ==========================

train_df = create_features(train_df)

test_df = create_features(test_df)

# ==========================
# Save Files
# ==========================

train_df.to_csv(

    "data/train_engineered.csv",

    index=False

)

test_df.to_csv(

    "data/test_engineered.csv",

    index=False

)

print("Feature Engineering Completed Successfully!")

print("New Train Shape :", train_df.shape)

print("New Test Shape :", test_df.shape)