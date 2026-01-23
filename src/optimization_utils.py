import numpy as np
import pandas as pd
import joblib


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
    Returns:
      - UIV_CHOICES: list of allowed categories for UIV_implant
      - xl: lower bounds for decision vector (ints)
      - xu: upper bounds for decision vector (ints)

    Decision vector layout:
      x = [uiv_code, num_fused_levels, ALIF, XLIF, TLIF, num_rods, num_screws, osteotomy]
    """
    UIV_CHOICES = ["hook", "PS", "FS"]

    xl = np.array([
        0,   # uiv_code
        1,   # num_fused_levels
        1,   # ALIF
        0,   # XLIF
        0,   # TLIF
        2,   # num_rods
        1,   # num_screws
        0,   # osteotomy
    ], dtype=int)

    xu = np.array([
        len(UIV_CHOICES) - 1,
        4,   # num_fused_levels
        1,   # ALIF
        1,   # XLIF
        1,   # TLIF
        4,   # num_rods
        4,   # num_screws
        1,   # osteotomy
    ], dtype=int)

    return UIV_CHOICES, xl, xu


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


def decode_plan(x, uiv_choices):
    """
    Decode numeric decision vector into surgical plan dict.

    x layout (ints):
      0: UIV_implant_code
      1: num_fused_levels
      2: ALIF
      3: XLIF
      4: TLIF
      5: num_rods
      6: num_screws
      7: osteotomy
    """
    x = np.asarray(x).astype(int)

    return {
        "UIV_implant": uiv_choices[x[0]],
        "num_fused_levels": int(x[1]),
        "ALIF": int(x[2]),
        "XLIF": int(x[3]),
        "TLIF": int(x[4]),
        "num_rods": int(x[5]),
        "num_screws": int(x[6]),
        "osteotomy": int(x[7]),
    }


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
