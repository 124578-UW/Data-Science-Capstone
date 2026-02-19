"""
Configuration file for data paths, decision variables, model features, and optimization bounds.
This is the single source of truth for all decision and feature definitions.
"""
from pathlib import Path
import numpy as np

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root (assumes this file is in src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw data (input)
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "MSDS_cleaned_with_CCI_ODI.xlsx"

# Processed/cleaned data (output from 00_data_cleaning.ipynb)
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "cleaned_for_modeling.csv"

# Holdout test patients (excluded from training)
DATA_HOLDOUT = PROJECT_ROOT / "data" / "processed" / "holdout_patients.csv"

# ============================================================================
# MODEL ARTIFACTS
# ============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Mechanical failure model
MECH_FAIL_MODEL = ARTIFACTS_DIR / "MechanicalFailure" / "mech_fail_logreg.joblib"

# Parameter change models
L4S1_MODEL = ARTIFACTS_DIR / "L4S1" / "L4S1_ridge_reg.joblib"
LL_MODEL = ARTIFACTS_DIR / "LL" / "LL_ridge_reg.joblib"
T4PA_MODEL = ARTIFACTS_DIR / "T4PA" / "T4PA_ridge_reg.joblib"
L1PA_MODEL = ARTIFACTS_DIR / "L1PA" / "L1PA_rf_reg.joblib"
SVA_MODEL = ARTIFACTS_DIR / "SVA" / "delta_SVA_model.joblib"
SS_MODEL = ARTIFACTS_DIR / "SS" / "delta_SS_model.joblib"
GLOBAL_TILT_MODEL = ARTIFACTS_DIR / "GlobalTilt" / "delta_GlobalTilt_model.joblib"

# ODI Change model
ODI_MODEL = ARTIFACTS_DIR / "ODI" / "ODI_ridge_reg.joblib"

# ============================================================================
# DECISION VARIABLES
# ============================================================================

UIV_CHOICES = ["Hook", "PS", "FS"]
NUM_LEVELS_CAT_CHOICES = ["lower", "higher"]

DECISION_VAR_SPECS = [
    {
        "col_name": "UIV_implant",
        "vector_name": "uiv_code",
        "lower": 0,
        "upper": len(UIV_CHOICES) - 1,
        "categorical": True,
        "choices": UIV_CHOICES,
    },
    {
        "col_name": "num_levels_cat",
        "vector_name": "num_levels_cat_code",
        "lower": 0,
        "upper": len(NUM_LEVELS_CAT_CHOICES) - 1,
        "categorical": True,
        "choices": NUM_LEVELS_CAT_CHOICES,
    },
    {
        "col_name": "num_interbody_fusion_levels",
        "vector_name": "num_interbody_fusion_levels",
        "lower": 0,
        "upper": 5,
        "categorical": False,
    },
    {
        "col_name": "ALIF",
        "vector_name": "ALIF",
        "lower": 0,
        "upper": 1,
        "categorical": False,
    },
    {
        "col_name": "XLIF",
        "vector_name": "XLIF",
        "lower": 0,
        "upper": 1,
        "categorical": False,
    },
    {
        "col_name": "TLIF",
        "vector_name": "TLIF",
        "lower": 0,
        "upper": 1,
        "categorical": False,
    },
    {
        "col_name": "num_rods",
        "vector_name": "num_rods",
        "lower": 1,
        "upper": 6,
        "categorical": False,
    },
    {
        "col_name": "num_pelvic_screws",
        "vector_name": "num_pelvic_screws",
        "lower": 2,
        "upper": 4,
        "categorical": False,
    },
    {
        "col_name": "osteotomy",
        "vector_name": "osteotomy",
        "lower": 0,
        "upper": 1,
        "categorical": False,
    },
]

DECISION_VAR_NAMES = [var["vector_name"] for var in DECISION_VAR_SPECS]
PLAN_COLS = [var["col_name"] for var in DECISION_VAR_SPECS]
DECISION_VAR_LOWER_BOUNDS = np.array([var["lower"] for var in DECISION_VAR_SPECS], dtype=int)
DECISION_VAR_UPPER_BOUNDS = np.array([var["upper"] for var in DECISION_VAR_SPECS], dtype=int)
DECISION_VAR_MAPPING = {i: name for i, name in enumerate(PLAN_COLS)}

# ============================================================================
# PATIENT FIXED VARIABLES
# ============================================================================
PATIENT_FIXED_COLS = [
    "age",
    "sex",
    "bmi",
    "C7CSVL_preop",
    "SVA_preop",
    "T4PA_preop",
    "L1PA_preop",
    "LL_preop",
    "L4S1_preop",
    "PT_preop",
    "PI_preop",
    "SS_preop",
    "cobb_main_curve_preop",
    "FC_preop",
    "tscore_femneck_preop",
    "HU_UIV_preop",
    "HU_UIVplus1_preop",
    "HU_UIVplus2_preop",
    "gap_category",      # Preop GAP category for composite score
    "gap_score_preop",   # Preop GAP score
    "global_tilt_preop", # Preop Global Tilt (used as GlobalTilt_preop)
    "CCI",
    "ASA_CLASS",
    "revision",             # Revision indicator (interbody > 0 but no ALIF/XLIF/TLIF)
    "ODI_preop",            # Preop ODI score for postop ODI prediction
]
# ============================================================================
# TARGETS
# ============================================================================

MECH_FAIL_TARGET = "mech_fail_last"
# ============================================================================
# MODEL FEATURE SETS
# ============================================================================

# Mechanical failure uses smoking additionally
MECH_FAIL_FEATURES = PATIENT_FIXED_COLS + PLAN_COLS + ["smoking"]

# Delta models do NOT use smoking
DELTA_MODEL_FEATURES = PATIENT_FIXED_COLS + PLAN_COLS
