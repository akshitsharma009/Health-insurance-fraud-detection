"""
preprocessing.py
-----------------
Reusable feature-engineering pipeline for the Health Insurance Fraud
Detection project.

This module is the single source of truth for turning raw claim data into
the exact feature set the model is trained and served on. It is used by
`train_final_model.py` (training) so that training and any future batch
scoring stay perfectly in sync.

Fixes applied over the original notebook version:
1. Missing values encoded as the literal string "?" (present in
   `collision_type`, `property_damage`, `police_report_available`) are
   converted to real NaN instead of being silently treated as their own
   category. Explicit `*_missing` indicator flags are added, since
   missingness itself carries fraud signal (e.g. a claim with no police
   report on file).
2. Pure-identifier / leakage-prone columns (`policy_number`,
   `incident_location`, `insured_zip`) are dropped. These are unique
   (or near-unique) per row, so one-hot encoding them effectively lets the
   model memorize individual rows instead of learning generalizable
   patterns.
"""

import numpy as np
import pandas as pd


# Columns that are pure identifiers or too high-cardinality to generalize
# from a ~1,000 row dataset. Kept as a named constant so it's obvious this
# is a deliberate modeling decision, not an oversight.
LEAKAGE_PRONE_COLUMNS = [
    "policy_number",
    "incident_location",
    "insured_zip",
]

# Columns known to contain missing values encoded as "?" in the raw export.
COLUMNS_WITH_HIDDEN_MISSING = [
    "collision_type",
    "property_damage",
    "police_report_available",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw claims dataframe (matching the original CSV schema)
    into the final modeling feature set.

    Parameters
    ----------
    df : pd.DataFrame
        Raw claims data, with `policy_bind_date`, `incident_date` as
        date-like columns and `fraud_reported` as 'Y'/'N' (if present).

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe, ready to be split into X/y and fed
        into the modeling pipeline.
    """

    df = df.copy()

    # ------------------------------------------------------------
    # Fix hidden missing values ("?" -> NaN)
    # ------------------------------------------------------------
    df = df.replace("?", np.nan)

    # ------------------------------------------------------------
    # Target encoding (only if present, e.g. not present at inference)
    # ------------------------------------------------------------
    if "fraud_reported" in df.columns and not pd.api.types.is_numeric_dtype(df["fraud_reported"]):
        df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

    # ------------------------------------------------------------
    # Date-derived features
    # ------------------------------------------------------------
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
    df["incident_date"] = pd.to_datetime(df["incident_date"])

    df["policy_age_days"] = (df["incident_date"] - df["policy_bind_date"]).dt.days
    df["incident_month"] = df["incident_date"].dt.month
    df["incident_dayofweek"] = df["incident_date"].dt.dayofweek

    # ------------------------------------------------------------
    # Claim ratio features
    # ------------------------------------------------------------
    df["claim_to_premium_ratio"] = df["total_claim_amount"] / df["policy_annual_premium"]
    df["injury_claim_ratio"] = df["injury_claim"] / df["total_claim_amount"]
    df["property_claim_ratio"] = df["property_claim"] / df["total_claim_amount"]
    df["vehicle_claim_ratio"] = df["vehicle_claim"] / df["total_claim_amount"]

    ratio_cols = [
        "claim_to_premium_ratio",
        "injury_claim_ratio",
        "property_claim_ratio",
        "vehicle_claim_ratio",
    ]
    df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], 0).fillna(0)

    # ------------------------------------------------------------
    # Consistency / investigation-signal flag features
    # ------------------------------------------------------------
    df["no_witness_high_severity"] = (
        (df["witnesses"] == 0)
        & (df["incident_severity"].isin(["Major Damage", "Total Loss"]))
    ).astype(int)

    df["no_police_but_injury"] = (
        (df["police_report_available"] == "NO") & (df["bodily_injuries"] > 0)
    ).astype(int)

    # ------------------------------------------------------------
    # NEW: explicit missingness indicator flags
    # ------------------------------------------------------------
    for col in COLUMNS_WITH_HIDDEN_MISSING:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    # ------------------------------------------------------------
    # Drop redundant / leakage-prone columns
    # ------------------------------------------------------------
    drop_cols = ["policy_bind_date", "incident_date"] + LEAKAGE_PRONE_COLUMNS
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def get_feature_columns(df: pd.DataFrame):
    """
    Split a feature-engineered dataframe's columns into numeric and
    categorical column name lists, matching how the modeling
    ColumnTransformer expects them.
    """

    feature_df = df.drop(columns=["fraud_reported"]) if "fraud_reported" in df.columns else df

    num_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

    return num_cols, cat_cols
