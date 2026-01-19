import pandas as pd


def build_input_df(user_inputs: dict) -> pd.DataFrame:
    """
    Build a complete input dataframe matching the training schema.
    Missing features are filled with safe default values.
    User-provided inputs override defaults.
    """

    # -----------------------------
    # Default feature values
    # -----------------------------
    default_data = {
        # -----------------
        # Core numeric features
        # -----------------
        "months_as_customer": 12,
        "age": 35,
        "policy_annual_premium": 40000,
        "policy_deductable": 500,
        "umbrella_limit": 0,
        "capital-gains": 0,
        "capital-loss": 0,
        "bodily_injuries": 0,
        "witnesses": 0,
        "number_of_vehicles_involved": 1,
        "total_claim_amount": 100000,
        "injury_claim": 30000,
        "property_claim": 30000,
        "vehicle_claim": 40000,
        "incident_hour_of_the_day": 12,
        "policy_age_days": 365,
        "incident_month": 6,
        "incident_dayofweek": 2,

        # -----------------
        # Engineered ratio features
        # -----------------
        "claim_to_premium_ratio": 2.0,
        "injury_claim_ratio": 0.3,
        "property_claim_ratio": 0.3,
        "vehicle_claim_ratio": 0.4,

        # -----------------
        # Fraud signal flags (initial)
        # -----------------
        "no_police_but_injury": 0,
        "no_witness_high_severity": 0,

        # -----------------
        # Categorical features
        # -----------------
        "policy_state": "OH",
        "policy_csl": "250/500",
        "insured_sex": "MALE",
        "insured_education_level": "High School",
        "insured_occupation": "adm-clerical",
        "insured_hobbies": "reading",
        "insured_relationship": "husband",
        "incident_type": "Single Vehicle Collision",
        "collision_type": "Front Collision",
        "incident_severity": "Minor Damage",
        "authorities_contacted": "Police",
        "incident_state": "OH",
        "incident_city": "Columbus",
        "incident_location": "Unknown",
        "property_damage": "NO",
        "police_report_available": "YES",
        "auto_make": "Toyota",
        "auto_model": "Camry",
        "auto_year": 2018,
        "insured_zip": "43004",

        # -----------------
        # Identifier-like numeric column
        # -----------------
        "policy_number": 0
    }

    # -----------------------------
    # Override defaults with user inputs
    # -----------------------------
    default_data.update(user_inputs)

    # -----------------------------
    # Recompute engineered ratios safely
    # -----------------------------
    premium = max(default_data["policy_annual_premium"], 1)

    default_data["claim_to_premium_ratio"] = (
        default_data["total_claim_amount"] / premium
    )

    total_claim = max(default_data["total_claim_amount"], 1)

    default_data["injury_claim_ratio"] = (
        default_data["injury_claim"] / total_claim
    )

    default_data["property_claim_ratio"] = (
        default_data["property_claim"] / total_claim
    )

    default_data["vehicle_claim_ratio"] = (
        default_data["vehicle_claim"] / total_claim
    )

    # -----------------------------
    # Derived fraud flags (KEY LOGIC)
    # -----------------------------
    if (
        default_data.get("police_report_available") == "NO"
        and default_data.get("injury_claim", 0) > 0
    ):
        default_data["no_police_but_injury"] = 1
    else:
        default_data["no_police_but_injury"] = 0

    if (
        default_data.get("witnesses", 0) == 0
        and default_data.get("incident_severity") == "Major Damage"
    ):
        default_data["no_witness_high_severity"] = 1
    else:
        default_data["no_witness_high_severity"] = 0

    # -----------------------------
    # Return dataframe
    # -----------------------------
    return pd.DataFrame([default_data])
