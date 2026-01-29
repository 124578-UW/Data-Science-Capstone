"""
Configuration file for decision variables, model features, and optimization bounds.
This is the single source of truth for all decision and feature definitions.
"""

import numpy as np

UIV_CHOICES = ["Hook", "PS", "FS"]

DECISION_VAR_SPECS = [
    {
        "col_name": "UIV_implant",
        "vector_name": "uiv_code",
        "lower": 0,
        "upper": len(UIV_CHOICES) - 1,
    },
    {
        "col_name": "num_interbody_fusion_levels",
        "vector_name": "num_interbody_fusion_levels",
        "lower": 0,
        "upper": 5,
    },
    {
        "col_name": "ALIF",
        "vector_name": "ALIF",
        "lower": 0,
        "upper": 1,
    },
    {
        "col_name": "XLIF",
        "vector_name": "XLIF",
        "lower": 0,
        "upper": 1,
    },
    {
        "col_name": "TLIF",
        "vector_name": "TLIF",
        "lower": 0,
        "upper": 1,
    },
    {
        "col_name": "num_rods",
        "vector_name": "num_rods",
        "lower": 1,
        "upper": 6,
    },
    {
        "col_name": "num_pelvic_screws",
        "vector_name": "num_pelvic_screws",
        "lower": 1,
        "upper": 6,
    },
    {
        "col_name": "osteotomy",
        "vector_name": "osteotomy",
        "lower": 0,
        "upper": 1,
    },
]

DECISION_VAR_NAMES = [var["vector_name"] for var in DECISION_VAR_SPECS]
PLAN_COLS = [var["col_name"] for var in DECISION_VAR_SPECS]
DECISION_VAR_LOWER_BOUNDS = np.array([var["lower"] for var in DECISION_VAR_SPECS], dtype=int)
DECISION_VAR_UPPER_BOUNDS = np.array([var["upper"] for var in DECISION_VAR_SPECS], dtype=int)

DECISION_VAR_MAPPING = {i: name for i, name in enumerate(PLAN_COLS)}


PREDICTORS = [
    "age", "sex",
    "PI_preop", "PT_preop", "LL_preop", "SS_preop",
    "T4PA_preop", "L1PA_preop", "SVA_preop",
    "cobb_main_curve_preop", "FC_preop", "tscore_femneck_preop",
    "HU_UIV_preop", "HU_UIVplus1_preop", "HU_UIVplus2_preop",
]

# All features used by the models (predictors + plan variables)
ALL_FEATURES = PREDICTORS + PLAN_COLS


TARGET_MECH_FAIL = "mech_fail_last"
TARGET_COMPOSITE = "composite_score"


MODEL_MECH_FAIL_XGB = "../artifacts/mech_fail_xgb.joblib"
MODEL_COMPOSITE = "../artifacts/composite_score_model.joblib"


DATA_CLEANED = "../data/cleaned/MSDS_cleaned_0122.csv"
DATA_INTERMEDIATE = "../data/intermediate/MSDS_database_with_composite_scores.csv"
