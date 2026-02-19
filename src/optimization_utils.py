import numpy as np
import pandas as pd
import joblib

from src import config


# Columns needed for patient_fixed dict – single source of truth is config
PATIENT_FIXED_COLS = config.PATIENT_FIXED_COLS


def load_patient_data(index=None, patient_id=None, data_path=None):
    """
    Load patient preop data from the cleaned dataset.
    
    Args:
        index: Row index (0-based) to load. If None and patient_id is None, returns first patient.
        patient_id: Patient ID to look up from 'id' column. Takes precedence over index.
        data_path: Path to CSV file. Defaults to config.DATA_PROCESSED.
        
    Returns:
        dict: Patient fixed parameters for optimization.
    """
    if data_path is None:
        data_path = config.DATA_PROCESSED
    
    df = pd.read_csv(data_path)
    
    # Look up by patient_id if provided
    if patient_id is not None:
        if "id" not in df.columns:
            raise ValueError("Dataset does not have 'id' column")
        mask = df["id"] == patient_id
        if not mask.any():
            raise ValueError(f"Patient ID {patient_id} not found in dataset")
        row = df.loc[mask].iloc[0]
    else:
        if index is None:
            index = 0
        if index < 0 or index >= len(df):
            raise ValueError(f"Index {index} out of range. Dataset has {len(df)} rows.")
        row = df.iloc[index]
    
    # Build patient_fixed dict
    patient_fixed = {}
    for col in PATIENT_FIXED_COLS:
        if col in df.columns:
            val = row[col]
            # Handle NaN values
            if pd.isna(val):
                patient_fixed[col] = None
            else:
                patient_fixed[col] = val
        else:
            print(f"Warning: Column '{col}' not found in dataset")
            patient_fixed[col] = None
    
    # Add GlobalTilt_preop alias for scoring module (keep original for model features)
    if "global_tilt_preop" in patient_fixed:
        patient_fixed["GlobalTilt_preop"] = patient_fixed["global_tilt_preop"]
    
    return patient_fixed


def get_patient_count(data_path=None):
    """Get total number of patients in dataset."""
    if data_path is None:
        data_path = config.DATA_PROCESSED
    df = pd.read_csv(data_path)
    return len(df)


def load_model_bundle(path):
    """
    Load sklearn pipeline bundle saved with joblib.
    Expected keys: 'pipe', 'features'
    """
    bundle = joblib.load(path)

    if "pipe" not in bundle or "features" not in bundle:
        raise ValueError(f"Invalid model bundle keys: {bundle.keys()}")

    bundle["features"] = list(bundle["features"])
    return bundle


def get_decision_config():
    """
    Returns decision variable configuration from config.
    
    Returns:
      - UIV_CHOICES: list of allowed categories for UIV_implant
      - xl: lower bounds for decision vector (ints)
      - xu: upper bounds for decision vector (ints)
    """
    return (
        config.UIV_CHOICES,
        config.DECISION_VAR_LOWER_BOUNDS,
        config.DECISION_VAR_UPPER_BOUNDS,
    )


def build_feature_row(full_dict, features):
    """
    Build a single-row DataFrame in training feature order.
    Missing values become NaN (handled by pipeline).
    """
    row = {f: full_dict.get(f, np.nan) for f in features}
    return pd.DataFrame([row], columns=features)


def predict_mech_fail_prob(full_dict, bundle):
    """
    Predict mechanical failure probability.
    """
    X = build_feature_row(full_dict, bundle["features"])
    return float(bundle["pipe"].predict_proba(X)[:, 1][0])


def decode_plan(x, uiv_choices=None):
    """
    Decode numeric decision vector into surgical plan dict.
    Uses DECISION_VAR_SPECS from config for column names and categorical mappings.
    
    Args:
        x: decision vector (ints)
        uiv_choices: deprecated, kept for backwards compatibility (uses config instead)
    """
    x = np.asarray(x).astype(int)
    
    plan = {}
    for i, var_spec in enumerate(config.DECISION_VAR_SPECS):
        col_name = var_spec["col_name"]
        if var_spec.get("categorical", False):
            # Map integer code to categorical value
            choices = var_spec["choices"]
            plan[col_name] = choices[x[i]]
        else:
            plan[col_name] = int(x[i])
    
    return plan


def build_full_input(patient_fixed, plan):
    """
    Merge preop fixed variables with surgical decisions.
    """
    return {**patient_fixed, **plan}


def fitness_mech_fail_only(x, patient_fixed, bundle, uiv_choices):
    """
    Scalar fitness to MINIMIZE: predicted mech failure probability.
    """
    plan = decode_plan(x, uiv_choices)
    full = build_full_input(patient_fixed, plan)
    return predict_mech_fail_prob(full, bundle)


def debug_candidate(x, patient_fixed, bundle, uiv_choices):
    """
    Debug helper for inspecting one candidate.
    """
    plan = decode_plan(x, uiv_choices)
    full = build_full_input(patient_fixed, plan)
    p_fail = predict_mech_fail_prob(full, bundle)

    return {
        "x": np.asarray(x).astype(int).tolist(),
        "plan": plan,
        "mech_fail_prob": p_fail,
        "fitness": p_fail,
    }


# ============================================================================
# Delta Model Predictions
# ============================================================================

def predict_delta(full_dict, bundle):
    """
    Predict a single delta value (e.g., delta_LL, delta_SS) using a regression model.
    """
    X = build_feature_row(full_dict, bundle["features"])
    return float(bundle["pipe"].predict(X)[0])


def predict_all_deltas(full_dict, delta_bundles):
    """
    Predict all delta values for a surgical plan.
    
    Args:
        full_dict: Combined patient + plan dict
        delta_bundles: dict with keys like 'L4S1', 'LL', 'SS', etc. and model bundles as values
        
    Returns:
        dict with keys like 'delta_L4S1', 'delta_LL', etc.
    """
    deltas = {}
    for name, bundle in delta_bundles.items():
        try:
            deltas[f"delta_{name}"] = predict_delta(full_dict, bundle)
        except Exception as e:
            raise ValueError(
                f"Delta model '{name}' failed: {e}\n"
                f"  Bundle features ({len(bundle['features'])}): {bundle['features']}"
            ) from e
    return deltas


# ============================================================================
# Composite Score Fitness
# ============================================================================

def fitness_composite_score(x, patient_fixed, delta_bundles, mech_fail_bundle=None, odi_bundle=None, weights=None):
    """
    Scalar fitness to MINIMIZE: composite score based on predicted outcomes.
    
    Args:
        x: decision vector (ints)
        patient_fixed: dict with patient preop values
        delta_bundles: dict with delta model bundles (keys: L4S1, LL, T4PA, L1PA, SVA, SS, GlobalTilt)
        mech_fail_bundle: mechanical failure model bundle (optional, for w_mech_fail)
        odi_bundle: ODI model bundle (optional, for w_odi)
        weights: dict with keys w1-w6, w_mech_fail, w_odi for composite score weights (optional)
        
    Returns:
        float: composite score (lower is better)
    """
    from src import scoring
    
    plan = decode_plan(x)
    full = build_full_input(patient_fixed, plan)
    
    # Predict all deltas
    deltas = predict_all_deltas(full, delta_bundles)
    
    # Build patient_preop dict for scoring function
    patient_preop = {
        "PI_preop": patient_fixed.get("PI_preop"),
        "LL_preop": patient_fixed.get("LL_preop"),
        "SS_preop": patient_fixed.get("SS_preop"),
        "L4S1_preop": patient_fixed.get("L4S1_preop"),
        "GlobalTilt_preop": patient_fixed.get("GlobalTilt_preop"),
        "T4PA_preop": patient_fixed.get("T4PA_preop"),
        "L1PA_preop": patient_fixed.get("L1PA_preop"),
        "age": patient_fixed.get("age"),
        "gap_category": patient_fixed.get("gap_category"),
    }
    
    # Map delta names to match scoring function expectations
    delta_predictions = {
        "delta_LL": deltas.get("delta_LL", 0),
        "delta_SS": deltas.get("delta_SS", 0),
        "delta_L4S1": deltas.get("delta_L4S1", 0),
        "delta_GlobalTilt": deltas.get("delta_GlobalTilt", 0),
        "delta_T4PA": deltas.get("delta_T4PA", 0),
        "delta_L1PA": deltas.get("delta_L1PA", 0),
    }
    
    # Calculate composite score
    if weights is None:
        weights = {}
    
    # Predict mechanical failure probability if bundle provided
    mech_fail_prob = 0.0
    if mech_fail_bundle is not None:
        mech_fail_prob = predict_mech_fail_prob(full, mech_fail_bundle)
    
    # Predict ODI postop if bundle provided
    odi_postop = None
    if odi_bundle is not None:
        delta_odi = predict_delta(full, odi_bundle)
        odi_preop = patient_fixed.get("ODI_preop")
        if odi_preop is not None:
            odi_postop = odi_preop + delta_odi
    
    composite, postop_values, gap_info = scoring.composite_score_from_predictions(
        patient_preop=patient_preop,
        delta_predictions=delta_predictions,
        weights=weights,
        mech_fail_prob=mech_fail_prob,
        odi_postop=odi_postop
    )
    
    return composite


def evaluate_solution(x, patient_fixed, delta_bundles, mech_fail_bundle, weights=None, odi_bundle=None):
    """
    Evaluate a solution and return detailed results including composite score,
    mechanical failure probability, predicted deltas, and GAP info.
    
    Args:
        x: decision vector (ints)
        patient_fixed: dict with patient preop values
        delta_bundles: dict with delta model bundles
        mech_fail_bundle: mechanical failure model bundle
        weights: dict with keys w1-w6 for composite score weights (optional)
        odi_bundle: ODI model bundle (optional, for w_odi)
        
    Returns:
        dict with plan, composite_score, mech_fail_prob, deltas, postop_values, gap_info
    """
    from src import scoring
    
    plan = decode_plan(x)
    full = build_full_input(patient_fixed, plan)
    
    # Predict mechanical failure
    mech_fail_prob = predict_mech_fail_prob(full, mech_fail_bundle)
    
    # Predict all deltas
    deltas = predict_all_deltas(full, delta_bundles)
    
    # Build patient_preop dict for scoring function
    patient_preop = {
        "PI_preop": patient_fixed.get("PI_preop"),
        "LL_preop": patient_fixed.get("LL_preop"),
        "SS_preop": patient_fixed.get("SS_preop"),
        "L4S1_preop": patient_fixed.get("L4S1_preop"),
        "GlobalTilt_preop": patient_fixed.get("GlobalTilt_preop"),
        "T4PA_preop": patient_fixed.get("T4PA_preop"),
        "L1PA_preop": patient_fixed.get("L1PA_preop"),
        "age": patient_fixed.get("age"),
        "gap_category": patient_fixed.get("gap_category"),
    }
    
    # Map delta names
    delta_predictions = {
        "delta_LL": deltas.get("delta_LL", 0),
        "delta_SS": deltas.get("delta_SS", 0),
        "delta_L4S1": deltas.get("delta_L4S1", 0),
        "delta_GlobalTilt": deltas.get("delta_GlobalTilt", 0),
        "delta_T4PA": deltas.get("delta_T4PA", 0),
        "delta_L1PA": deltas.get("delta_L1PA", 0),
    }
    
    if weights is None:
        weights = {}
    
    composite, postop_values, gap_info = scoring.composite_score_from_predictions(
        patient_preop=patient_preop,
        delta_predictions=delta_predictions,
        weights=weights,
        mech_fail_prob=mech_fail_prob,
        odi_postop=None  # computed below if odi_bundle provided
    )
    
    # Add SVA postop if we have the delta prediction
    if "delta_SVA" in deltas and patient_fixed.get("SVA_preop") is not None:
        postop_values["SVA_postop"] = patient_fixed["SVA_preop"] + deltas["delta_SVA"]
    
    # Predict ODI delta and postop if bundle provided
    odi_postop = None
    if odi_bundle is not None:
        delta_odi = predict_delta(full, odi_bundle)
        deltas["delta_ODI"] = delta_odi
        odi_preop = patient_fixed.get("ODI_preop")
        if odi_preop is not None:
            odi_postop = odi_preop + delta_odi
            postop_values["ODI_postop"] = odi_postop
    
    # Recompute composite with ODI if it's active
    if odi_postop is not None and weights.get("w_odi", 0) > 0:
        composite, _, _ = scoring.composite_score_from_predictions(
            patient_preop=patient_preop,
            delta_predictions=delta_predictions,
            weights=weights,
            mech_fail_prob=mech_fail_prob,
            odi_postop=odi_postop
        )
    
    # Also compute a display composite using equal weights on original 6 components only
    equal_weights = {f"w{i}": 1/6 for i in range(1, 7)}
    equal_weights["w_mech_fail"] = 0
    equal_weights["w_odi"] = 0
    display_composite, _, _ = scoring.composite_score_from_predictions(
        patient_preop=patient_preop,
        delta_predictions=delta_predictions,
        weights=equal_weights,
        mech_fail_prob=0.0,
        odi_postop=None
    )
    
    return {
        "plan": plan,
        "composite_score": composite,
        "display_composite_score": display_composite,
        "mech_fail_prob": mech_fail_prob,
        "deltas": deltas,
        "postop_values": postop_values,
        "gap_info": gap_info,
    }
