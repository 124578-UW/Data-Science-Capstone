import numpy as np
import pandas as pd
import joblib

from src import config
from src.scoring import calculate_gap_from_postop_values


# Columns needed for patient_fixed dict – single source of truth is config
PATIENT_FIXED_COLS = config.PATIENT_FIXED_COLS

_ALLOWED_INPUT_VALUES = {
    "sex": ["MALE", "FEMALE", "M", "F"],
    "revision": [0, 1],
    "smoking": [0, 1],
    "ASA_CLASS": [1, 2, 3, 4, 5],
}


def _load_new_patient_input_records(input_mode="dict", patient_input=None, csv_path=None):
    mode = str(input_mode).strip().lower()

    if mode == "dict":
        if not isinstance(patient_input, dict):
            raise ValueError("For dict mode, patient_input must be a dict")
        return [patient_input]

    if mode == "csv":
        if csv_path is None:
            raise ValueError("For csv mode, csv_path is required")
        csv_df = pd.read_csv(csv_path)
        if len(csv_df) == 0:
            raise ValueError(f"CSV file has no rows: {csv_path}")
        return csv_df.to_dict(orient="records")

    raise ValueError("input_mode must be 'dict' or 'csv'")


def review_new_patient_inputs(input_mode="dict", patient_input=None, csv_path=None) -> pd.DataFrame:
    """
    Build a validation review table for one or more new-patient input records.

        Status meanings:
            - ERROR: invalid or missing value; optimization should not run
            - OK: value is present and valid
    """
    records = _load_new_patient_input_records(
        input_mode=input_mode,
        patient_input=patient_input,
        csv_path=csv_path,
    )

    required_cols = [
        c for c in config.PATIENT_FIXED_COLS
        if c not in {"gap_score_preop", "gap_category"}
    ] + ["smoking"]

    review_rows = []
    for i, rec in enumerate(records, start=1):
        raw_id = rec.get("id", f"NEW_PATIENT_{i}")
        patient_id = f"NEW_PATIENT_{i}" if pd.isna(raw_id) else str(raw_id)

        for field in required_cols:
            value = rec.get(field, None)
            status = "OK"
            note = ""
            expected = "required"

            if value is None or pd.isna(value):
                status = "ERROR"
                note = "Missing required value"
                if field in _ALLOWED_INPUT_VALUES:
                    expected = f"One of {_ALLOWED_INPUT_VALUES[field]}"
                else:
                    expected = "Numeric"
                review_rows.append({
                    "Patient": patient_id,
                    "Field": field,
                    "Value": value,
                    "Expected": expected,
                    "Status": status,
                    "Note": note,
                })
                continue

            if field == "sex":
                try:
                    _normalize_sex(value)
                    expected = "MALE/FEMALE (or M/F)"
                except ValueError:
                    status = "ERROR"
                    expected = "MALE/FEMALE (or M/F)"
                    note = "Unrecognized categorical value"

            elif field in {"revision", "smoking"}:
                try:
                    _normalize_binary(value, field)
                    expected = "0 or 1"
                except ValueError:
                    status = "ERROR"
                    expected = "0 or 1"
                    note = "Must be binary"

            elif field == "ASA_CLASS":
                expected = "1, 2, 3, 4, or 5"
                try:
                    asa_val = float(value)
                    if asa_val not in _ALLOWED_INPUT_VALUES["ASA_CLASS"]:
                        status = "ERROR"
                        note = "ASA class must be one of the accepted integer values"
                except Exception:
                    status = "ERROR"
                    note = "ASA class must be numeric"

            else:
                expected = "Numeric"
                try:
                    float(value)
                except Exception:
                    status = "ERROR"
                    note = "Must be numeric"

            review_rows.append({
                "Patient": patient_id,
                "Field": field,
                "Value": value,
                "Expected": expected,
                "Status": status,
                "Note": note,
            })

    return pd.DataFrame(review_rows)


def _normalize_sex(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().upper()
    if s in {"M", "MALE"}:
        return "MALE"
    if s in {"F", "FEMALE"}:
        return "FEMALE"
    raise ValueError("sex must be one of: MALE/FEMALE (or M/F)")


def _normalize_binary(v, name):
    if v in [0, "0", False]:
        return 0
    if v in [1, "1", True]:
        return 1
    raise ValueError(f"{name} must be 0 or 1")


def prepare_new_patient_record(patient_input: dict, fallback_id: str = "NEW_PATIENT") -> dict:
    """
    Validate and normalize one new-patient input record for optimization.

    Returns:
        dict with keys: patient_id, patient_fixed, smoking, profile_row,
        gap_score_preop, gap_category_preop
    """
    required_cols = [
        c for c in config.PATIENT_FIXED_COLS
        if c not in {"gap_score_preop", "gap_category"}
    ] + ["smoking"]

    missing = [c for c in required_cols if c not in patient_input or pd.isna(patient_input[c])]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    patient_fixed = {k: patient_input.get(k, None) for k in config.PATIENT_FIXED_COLS}
    patient_fixed["sex"] = _normalize_sex(patient_fixed["sex"])
    patient_fixed["revision"] = _normalize_binary(patient_fixed["revision"], "revision")
    smoking = _normalize_binary(patient_input["smoking"], "smoking")

    gap_score_preop, gap_category_preop, *_ = calculate_gap_from_postop_values(
        ss_postop=float(patient_fixed["SS_preop"]),
        ll_postop=float(patient_fixed["LL_preop"]),
        l4s1_postop=float(patient_fixed["L4S1_preop"]),
        global_tilt_postop=float(patient_fixed["global_tilt_preop"]),
        pi=float(patient_fixed["PI_preop"]),
        age=float(patient_fixed["age"]),
    )
    patient_fixed["gap_score_preop"] = int(gap_score_preop)
    patient_fixed["gap_category"] = gap_category_preop
    patient_fixed["GlobalTilt_preop"] = patient_fixed["global_tilt_preop"]

    raw_id = patient_input.get("id", fallback_id)
    patient_id = fallback_id if pd.isna(raw_id) else str(raw_id)
    profile_row = pd.Series({**patient_fixed, "smoking": smoking, "id": patient_id})

    return {
        "patient_id": patient_id,
        "patient_fixed": patient_fixed,
        "smoking": smoking,
        "profile_row": profile_row,
        "gap_score_preop": int(gap_score_preop),
        "gap_category_preop": gap_category_preop,
    }


def prepare_new_patient_inputs(input_mode="dict", patient_input=None, csv_path=None) -> list:
    """
    Prepare one or more new-patient records from dict or CSV input.

    Args:
        input_mode: "dict" or "csv"
        patient_input: dict when input_mode="dict"
        csv_path: path to CSV when input_mode="csv"

    Returns:
        list of prepared patient-record dicts (see prepare_new_patient_record)
    """
    records = _load_new_patient_input_records(
        input_mode=input_mode,
        patient_input=patient_input,
        csv_path=csv_path,
    )

    prepared = []
    for i, row_dict in enumerate(records, start=1):
        fallback_id = "NEW_PATIENT" if len(records) == 1 else f"NEW_PATIENT_{i}"
        prepared.append(prepare_new_patient_record(row_dict, fallback_id=fallback_id))
    return prepared


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
    x = np.rint(np.asarray(x, dtype=float)).astype(int)
    
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


# ============================================================================
# Clinical Correction Rules
# ============================================================================

# Clinical delta_LL correction: when PSO (osteotomy) is present,
# the expected lordosis correction is 20–45.
_PSO_LL_LOWER = 20.0
_PSO_LL_UPPER = 45.0


def clamp_delta_ll(delta_ll_ml: float, plan: dict) -> float:
    """
    Clamp ML-predicted delta_LL to a clinically expected range when
    osteotomy (PSO) is present.

    If PSO is present and the prediction falls outside [20, 45],
    it is clamped to that range.  Otherwise the ML prediction is
    returned unchanged.

    Args:
        delta_ll_ml: raw ML prediction for delta_LL
        plan: decoded surgical plan dict (keys: osteotomy, …)

    Returns:
        Clamped delta_LL value
    """
    if plan.get("osteotomy", 0) != 1:
        return delta_ll_ml

    return float(min(max(delta_ll_ml, _PSO_LL_LOWER), _PSO_LL_UPPER))


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

def fitness_composite_score(x, patient_fixed, delta_bundles, mech_fail_bundle=None, odi_bundle=None, weights=None, pso_ll_override=False):
    """
    Scalar fitness to MINIMIZE: composite score based on predicted outcomes.
    
    Args:
        x: decision vector (ints)
        patient_fixed: dict with patient preop values
        delta_bundles: dict with delta model bundles (keys: L4S1, LL, T4PA, L1PA, SVA, SS, GlobalTilt)
        mech_fail_bundle: mechanical failure model bundle (optional, for w_mech_fail)
        odi_bundle: ODI model bundle (optional, for w_odi)
        weights: dict with keys w1-w6, w_mech_fail, w_odi for composite score weights (optional)
        pso_ll_override: if True, clamp predicted delta_LL to procedure-based correction range
        
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
    
    # Apply clinical correction to delta_LL if enabled
    if pso_ll_override:
        delta_predictions["delta_LL"] = clamp_delta_ll(delta_predictions["delta_LL"], plan)
    
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


def evaluate_solution(x, patient_fixed, delta_bundles, mech_fail_bundle, weights=None, odi_bundle=None, pso_ll_override=False):
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
        pso_ll_override: if True, clamp predicted delta_LL to procedure-based correction range
        
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
    
    # Apply clinical correction to delta_LL if enabled
    if pso_ll_override:
        delta_predictions["delta_LL"] = clamp_delta_ll(delta_predictions["delta_LL"], plan)
    
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


def evaluate_actual_plan(x, patient_fixed, holdout_row, mech_fail_bundle):
    """
    Evaluate the actual surgical plan using *real* postop values from the
    holdout data rather than ML-predicted deltas.

    The mechanical failure model is still applied to the plan.  The alignment
    composite score is computed from recorded postop alignment values.

    Args:
        x: decision vector (ints) for the actual plan
        patient_fixed: dict with patient preop values
        holdout_row: pandas Series (one row from holdout_patients.csv)
        mech_fail_bundle: mechanical failure model bundle

    Returns:
        dict compatible with evaluate_solution output (plan, composite_score,
        display_composite_score, mech_fail_prob, deltas, postop_values, gap_info)
    """
    from src import scoring

    plan = decode_plan(x)
    full = build_full_input(patient_fixed, plan)

    # Predict mechanical failure (model-based)
    mech_fail_prob = predict_mech_fail_prob(full, mech_fail_bundle)

    # Map holdout column names → internal postop names
    _COL_MAP = {
        "LL_postop": "LL_postop",
        "SS_postop": "SS_postop",
        "L4_S1_postop": "L4S1_postop",
        "global_tilt_postop": "GlobalTilt_postop",
        "T4PA_postop": "T4PA_postop",
        "L1PA_postop": "L1PA_postop",
        "SVA_postop": "SVA_postop",
    }

    postop_values = {}
    for src_col, dst_key in _COL_MAP.items():
        val = holdout_row.get(src_col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            postop_values[dst_key] = float(val)

    # Back-compute deltas from actual postop − preop
    _PREOP_MAP = {
        "delta_LL": ("LL_postop", "LL_preop"),
        "delta_SS": ("SS_postop", "SS_preop"),
        "delta_L4S1": ("L4S1_postop", "L4S1_preop"),
        "delta_GlobalTilt": ("GlobalTilt_postop", "GlobalTilt_preop"),
        "delta_T4PA": ("T4PA_postop", "T4PA_preop"),
        "delta_L1PA": ("L1PA_postop", "L1PA_preop"),
    }
    deltas = {}
    for delta_key, (post_key, pre_key) in _PREOP_MAP.items():
        post_val = postop_values.get(post_key)
        pre_val = patient_fixed.get(pre_key)
        if post_val is not None and pre_val is not None:
            deltas[delta_key] = post_val - pre_val
        else:
            deltas[delta_key] = 0.0

    if "SVA_postop" in postop_values and patient_fixed.get("SVA_preop") is not None:
        deltas["delta_SVA"] = postop_values["SVA_postop"] - patient_fixed["SVA_preop"]

    # Build patient_preop dict for scoring
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

    delta_predictions = {
        "delta_LL": deltas.get("delta_LL", 0),
        "delta_SS": deltas.get("delta_SS", 0),
        "delta_L4S1": deltas.get("delta_L4S1", 0),
        "delta_GlobalTilt": deltas.get("delta_GlobalTilt", 0),
        "delta_T4PA": deltas.get("delta_T4PA", 0),
        "delta_L1PA": deltas.get("delta_L1PA", 0),
    }

    # Composite with equal weights on alignment only (display score)
    equal_weights = {f"w{i}": 1 / 6 for i in range(1, 7)}
    equal_weights["w_mech_fail"] = 0
    equal_weights["w_odi"] = 0

    display_composite, _, gap_info = scoring.composite_score_from_predictions(
        patient_preop=patient_preop,
        delta_predictions=delta_predictions,
        weights=equal_weights,
        mech_fail_prob=0.0,
        odi_postop=None,
    )

    # Use the same equal weights for the "optimization" score in the Actual column
    composite = display_composite

    # Add ODI postop from holdout if available (skip non-numeric strings like "NR")
    odi_candidates = [
        holdout_row.get("ODI_postop"),
        holdout_row.get("ODI_12mo"),
        holdout_row.get("ODI_6mo"),
    ]
    for value in odi_candidates:
        odi_numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(odi_numeric):
            postop_values["ODI_postop"] = float(odi_numeric)
            break

    return {
        "plan": plan,
        "composite_score": composite,
        "display_composite_score": display_composite,
        "mech_fail_prob": mech_fail_prob,
        "deltas": deltas,
        "postop_values": postop_values,
        "gap_info": gap_info,
    }
