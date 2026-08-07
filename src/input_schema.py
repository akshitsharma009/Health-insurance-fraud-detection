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
        # Missingness indicator flags
        # (the app's form always collects a concrete value for these
        # fields, so they default to "not missing"; kept here so the
        # input row has the exact column set the model was trained on)
        # -----------------
        "collision_type_missing": 0,
        "property_damage_missing": 0,
        "police_report_available_missing": 0,

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
        "property_damage": "NO",
        "police_report_available": "YES",
        "auto_make": "Toyota",
        "auto_model": "Camry",
        "auto_year": 2018,
    }

    # -----------------------------
    # Override defaults with user inputs
    # -----------------------------
    default_data.update(user_inputs)

    # -----------------------------
    # Auto-derive claim sub-components from total_claim_amount
    # -----------------------------
    # BUG FIX: The original version left injury_claim / property_claim /
    # vehicle_claim at fixed defaults (30000 / 30000 / 40000) regardless of
    # what total_claim_amount the user entered. In the training data,
    # injury_claim + property_claim + vehicle_claim == total_claim_amount
    # for every single row (a strict, always-true relationship) -- so a
    # user entering a much larger or smaller total claim than the ~100000
    # default produced an internally-inconsistent row the model had never
    # seen anything like during training (e.g. sub-claims summing to only
    # 8% of the stated total claim amount), which made predictions
    # unreliable for anything but claim amounts near the default.
    #
    # Fix: unless the caller explicitly supplies injury_claim /
    # property_claim / vehicle_claim, derive them from total_claim_amount
    # using the average split observed across the training data
    # (~13.9% injury, ~13.9% property, ~72.2% vehicle), so the three
    # sub-claims always sum exactly to total_claim_amount, keeping the
    # row consistent with the pattern the model actually learned.
    INJURY_SPLIT = 0.1392
    PROPERTY_SPLIT = 0.1386
    VEHICLE_SPLIT = 0.7221

    user_gave_subclaims = any(
        k in user_inputs for k in ("injury_claim", "property_claim", "vehicle_claim")
    )

    if not user_gave_subclaims:
        total_claim_amount = default_data["total_claim_amount"]
        default_data["injury_claim"] = round(total_claim_amount * INJURY_SPLIT)
        default_data["property_claim"] = round(total_claim_amount * PROPERTY_SPLIT)
        # Vehicle claim absorbs the rounding remainder so the three always
        # sum exactly back to total_claim_amount.
        default_data["vehicle_claim"] = (
            total_claim_amount
            - default_data["injury_claim"]
            - default_data["property_claim"]
        )

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
