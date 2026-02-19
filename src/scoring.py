"""
Scoring utilities for spine surgery optimization.

Contains functions to calculate:
- GAP score components (RPV, RLL, LDI, RSA)
- GAP score and category
- Composite score for optimization
"""

import numpy as np
import pandas as pd


# ============================================================================
# GAP Score Component Calculations
# ============================================================================

def calculate_ideal_ss(pi: float) -> float:
    """Calculate ideal Sacral Slope from Pelvic Incidence."""
    return pi * 0.59 + 9


def calculate_ideal_ll(pi: float) -> float:
    """Calculate ideal Lumbar Lordosis from Pelvic Incidence.
    
    Per GAP score reference: Ideal LL = PI x 0.62 + 29
    """
    return pi * 0.54 + 27.6


def calculate_ideal_global_tilt(pi: float) -> float:
    """Calculate ideal Global Tilt from Pelvic Incidence."""
    return pi * 0.48 - 15


def calculate_rpv(ss: float, pi: float) -> float:
    """
    Calculate Relative Pelvic Version (RPV).
    
    RPV = Measured SS - Ideal SS
    where Measured SS = PI - PT (or directly use SS_postop)
    """
    ideal_ss = calculate_ideal_ss(pi)
    return ss - ideal_ss


def calculate_rll(ll: float, pi: float) -> float:
    """
    Calculate Relative Lumbar Lordosis (RLL).
    
    RLL = Measured LL - Ideal LL
    """
    ideal_ll = calculate_ideal_ll(pi)
    return ll - ideal_ll


def calculate_ldi(l4s1: float, ll: float) -> float:
    """
    Calculate Lordosis Distribution Index (LDI).
    
    LDI = L4S1 / LL * 100
    """
    if ll == 0:
        return 0  # Avoid division by zero
    return (l4s1 / ll) * 100


def calculate_rsa(global_tilt: float, pi: float) -> float:
    """
    Calculate Relative Spinopelvic Alignment (RSA).
    
    RSA = Measured Global Tilt - Ideal Global Tilt
    """
    ideal_gt = calculate_ideal_global_tilt(pi)
    return global_tilt - ideal_gt


# ============================================================================
# GAP Score Component Scoring (0-13 scale)
# ============================================================================

def score_rpv(rpv: float) -> int:
    """
    Score RPV component (0-3).
    
    Based on GAP score methodology:
    - 3: RPV < -15 (severe retroversion)
    - 2: -15 <= RPV < -7.1 (moderate retroversion)
    - 0: -7.1 <= RPV <= 5 (aligned)
    - 1: RPV > 5 (anteversion)
    """
    if rpv < -15:
        return 3
    elif -15 <= rpv < -7.1:
        return 2
    elif -7.1 <= rpv <= 5:
        return 0
    else:  # rpv > 5
        return 1


def score_rll(rll: float) -> int:
    """
    Score RLL component (0-3).
    
    Based on GAP score methodology (from reference image):
    - 3: RLL < -25 (severe hypolordosis)
    - 2: -25 <= RLL < -14.1 (moderate hypolordosis)
    - 0: -14.1 <= RLL <= 11 (aligned)
    - 3: RLL > 11 (hyperlordosis)
    
    Note: Ideal Lumbar Lordosis = PI x 0.62 + 29 (per image)
    """
    if rll < -25:
        return 3
    elif -25 <= rll < -14.1:
        return 2
    elif -14.1 <= rll <= 11:
        return 0
    else:  # rll > 11
        return 3


def score_ldi(ldi: float) -> int:
    """
    Score LDI component (0-3).
    
    Based on GAP score methodology:
    - 2: LDI < 40% (severe hypolordotic maldistribution)
    - 1: 40% <= LDI < 50% (moderate hypolordotic maldistribution)
    - 0: 50% <= LDI <= 80% (aligned)
    - 3: LDI > 80% (hyperlordotic maldistribution)
    """
    if ldi < 40:
        return 2
    elif 40 <= ldi < 50:
        return 1
    elif 50 <= ldi <= 80:
        return 0
    else:  # ldi > 80
        return 3


def score_rsa(rsa: float) -> int:
    """
    Score RSA component (0-3).
    
    Based on GAP score methodology:
    - 3: RSA > 18 (severe positive malalignment)
    - 1: 10.1 <= RSA <= 18 (moderate positive malalignment)
    - 0: -7 <= RSA <= 10 (aligned)
    - 1: RSA < -7 (negative malalignment)
    
    Note: Ideal Global Tilt = PI x 0.48 - 15
    """
    if rsa > 18:
        return 3
    elif 10.1 <= rsa <= 18:
        return 1
    elif -7 <= rsa < 10.1:
        return 0
    else:  # rsa < -7
        return 1


def score_age(age: float) -> int:
    """
    Score Age component (0-1).
    
    Based on GAP score methodology:
    - 0: Age < 60 (Adult)
    - 1: Age >= 60 (Elderly Adult)
    """
    if age < 60:
        return 0
    else:
        return 1


# ============================================================================
# GAP Score and Category Calculation
# ============================================================================

def calculate_gap_score(rpv: float, rll: float, ldi: float, rsa: float, age: float) -> int:
    """
    Calculate total GAP score (0-13) from components.
    
    GAP Score = RPV_score + RLL_score + LDI_score + RSA_score + Age_score
    """
    return (
        score_rpv(rpv) +
        score_rll(rll) +
        score_ldi(ldi) +
        score_rsa(rsa) +
        score_age(age)
    )


def calculate_gap_category(gap_score: int) -> str:
    """
    Convert GAP score to category.
    
    Categories:
    - P (Proportioned): 0-2
    - MD (Moderately Disproportioned): 3-6
    - SD (Severely Disproportioned): 7-13
    """
    if gap_score <= 2:
        return "P"
    elif gap_score <= 6:
        return "MD"
    else:
        return "SD"


def calculate_gap_from_postop_values(
    ss_postop: float,
    ll_postop: float,
    l4s1_postop: float,
    global_tilt_postop: float,
    pi: float,
    age: float
) -> tuple:
    """
    Calculate GAP score and category from postoperative values.
    
    Returns:
        tuple: (gap_score, gap_category, rpv, rll, ldi, rsa)
    """
    rpv = calculate_rpv(ss_postop, pi)
    rll = calculate_rll(ll_postop, pi)
    ldi = calculate_ldi(l4s1_postop, ll_postop)
    rsa = calculate_rsa(global_tilt_postop, pi)
    
    gap_score = calculate_gap_score(rpv, rll, ldi, rsa, age)
    gap_category = calculate_gap_category(gap_score)
    
    return gap_score, gap_category, rpv, rll, ldi, rsa


# ============================================================================
# Composite Score Calculation
# ============================================================================

def composite_score_calc(
    gap_score_postop: float,
    l1pa_ideal_mismatch_postop: float,
    l4_s1_postop: float,
    t4l1pa_ideal_mismatch_postop: float,
    ll_postop: float,
    pi_preop: float,
    gap_category_postop: str,
    gap_category_preop: str,
    mech_fail_prob: float = 0.0,
    odi_postop: float = None,
    w1: float = 1,
    w2: float = 1,
    w3: float = 1,
    w4: float = 1,
    w5: float = 1,
    w6: float = 1,
    w_mech_fail: float = 0,
    w_odi: float = 0
) -> float:
    """
    Compute composite score based on GAP score and quadratic penalties for constraint violations.
    
    Lower scores are better.
    
    Parameters:
        gap_score_postop: GAP score after surgery (1-13 scale)
        l1pa_ideal_mismatch_postop: L1PA mismatch from ideal (postop)
        l4_s1_postop: L4-S1 angle after surgery
        t4l1pa_ideal_mismatch_postop: T4L1PA mismatch from ideal (postop)
        ll_postop: Lumbar Lordosis after surgery
        pi_preop: Pelvic Incidence (preop, unchanged)
        gap_category_postop: GAP category after surgery ("P", "MD", "SD")
        gap_category_preop: GAP category before surgery
        mech_fail_prob: predicted mechanical failure probability (0-1)
        odi_postop: predicted postoperative ODI score (0-100 scale, optional)
        w1-w6, w_mech_fail, w_odi: weights for each component
        
    Returns:
        Composite score (lower is better)
    """
    # Calculate relative weights
    weights = [w1, w2, w3, w4, w5, w6, w_mech_fail, w_odi]
    total_weight = sum(weights)
    rel_weights = [w / total_weight for w in weights]
    
    # Convert GAP score to 0-100 scale
    gap_normalized = gap_score_postop * 100 / 13
    
    # 1) L1PA quadratic penalty
    l1pa_pen = 0 if abs(l1pa_ideal_mismatch_postop) <= 3 else (abs(l1pa_ideal_mismatch_postop) - 3) ** 2
    # Normalize to 0-100 scale (max penalty ~2500 for 53 degree mismatch)
    l1pa_pen = min(l1pa_pen / 25, 100)
    
    # 2) L4S1 quadratic penalty (ideal range 35-45)
    if 35 <= l4_s1_postop <= 45:
        l4s1_pen = 0
    elif l4_s1_postop < 35:
        l4s1_pen = (35 - l4_s1_postop) ** 2
    else:
        l4s1_pen = (l4_s1_postop - 45) ** 2
    # Normalize to 0-100 scale
    l4s1_pen = min(l4s1_pen / 25, 100)
    
    # 3) T4L1PA penalty
    t4l1pa_pen = 0 if abs(t4l1pa_ideal_mismatch_postop) <= 3 else (abs(t4l1pa_ideal_mismatch_postop) - 3) ** 2
    # Normalize to 0-100 scale
    t4l1pa_pen = min(t4l1pa_pen / 25, 100)
    
    # 4) LL penalty (based on ideal LL)
    ll_ideal = calculate_ideal_ll(pi_preop)
    ll_mismatch = ll_postop - ll_ideal
    ll_pen = 0 if abs(ll_mismatch) <= 3 else (abs(ll_mismatch) - 3) ** 2
    # Normalize to 0-100 scale
    ll_pen = min(ll_pen / 25, 100)
    
    # 5) GAP improvement score based on category change
    if gap_category_postop == "P" and gap_category_preop in ["SD", "MD", "P"]:
        gap_improvement_pen = 0
    elif gap_category_postop == "MD" and gap_category_preop == "SD":
        gap_improvement_pen = 30
    else:
        gap_improvement_pen = 100
    
    # 6) Mechanical failure penalty (probability scaled to 0-100)
    mech_fail_pen = mech_fail_prob * 100
    
    # 7) ODI postop penalty (already on 0-100 scale)
    odi_pen = max(0, odi_postop) if odi_postop is not None else 0
    
    # Composite score (weighted sum)
    composite = (
        rel_weights[0] * gap_normalized +
        rel_weights[1] * l1pa_pen +
        rel_weights[2] * l4s1_pen +
        rel_weights[3] * t4l1pa_pen +
        rel_weights[4] * ll_pen +
        rel_weights[5] * gap_improvement_pen +
        rel_weights[6] * mech_fail_pen +
        rel_weights[7] * odi_pen
    )
    
    return composite


def composite_score_from_predictions(
    patient_preop: dict,
    delta_predictions: dict,
    weights: dict = None,
    mech_fail_prob: float = 0.0,
    odi_postop: float = None
) -> tuple:
    """
    Calculate composite score from patient preop data and predicted deltas.
    
    This is the main function to use during optimization.
    
    Parameters:
        patient_preop: dict with keys:
            - PI_preop, LL_preop, SS_preop, L4S1_preop, GlobalTilt_preop
            - T4PA_preop, L1PA_preop, age, gap_category
        delta_predictions: dict with keys:
            - delta_LL, delta_SS, delta_L4S1, delta_GlobalTilt
            - delta_T4PA, delta_L1PA
        weights: optional dict with keys w1-w6, w_mech_fail, w_odi
        mech_fail_prob: predicted mechanical failure probability (0-1)
        odi_postop: predicted postoperative ODI score (0-100, optional)
            
    Returns:
        tuple: (composite_score, postop_values_dict, gap_info_dict)
    """
    if weights is None:
        weights = {f"w{i}": 1 for i in range(1, 7)}
        weights["w_mech_fail"] = 0
        weights["w_odi"] = 0
    
    # Calculate postop values
    ll_postop = patient_preop["LL_preop"] + delta_predictions["delta_LL"]
    ss_postop = patient_preop["SS_preop"] + delta_predictions["delta_SS"]
    l4s1_postop = patient_preop["L4S1_preop"] + delta_predictions["delta_L4S1"]
    global_tilt_postop = patient_preop["GlobalTilt_preop"] + delta_predictions["delta_GlobalTilt"]
    t4pa_postop = patient_preop["T4PA_preop"] + delta_predictions["delta_T4PA"]
    l1pa_postop = patient_preop["L1PA_preop"] + delta_predictions["delta_L1PA"]
    
    pi = patient_preop["PI_preop"]
    age = patient_preop["age"]
    
    # Calculate GAP score and category
    gap_score, gap_category, rpv, rll, ldi, rsa = calculate_gap_from_postop_values(
        ss_postop, ll_postop, l4s1_postop, global_tilt_postop, pi, age
    )
    
    # Calculate ideal mismatches for L1PA and T4L1PA
    # L1PA ideal is typically 0 (aligned with vertical)
    l1pa_ideal_mismatch = l1pa_postop  # ideal L1PA = 0
    
    # T4L1PA ideal mismatch (difference between T4PA and L1PA should be small)
    t4l1pa_ideal_mismatch = t4pa_postop - l1pa_postop
    
    # Calculate composite score
    composite = composite_score_calc(
        gap_score_postop=gap_score,
        l1pa_ideal_mismatch_postop=l1pa_ideal_mismatch,
        l4_s1_postop=l4s1_postop,
        t4l1pa_ideal_mismatch_postop=t4l1pa_ideal_mismatch,
        ll_postop=ll_postop,
        pi_preop=pi,
        gap_category_postop=gap_category,
        gap_category_preop=patient_preop["gap_category"],
        mech_fail_prob=mech_fail_prob,
        odi_postop=odi_postop,
        **weights
    )
    
    postop_values = {
        "LL_postop": ll_postop,
        "SS_postop": ss_postop,
        "L4S1_postop": l4s1_postop,
        "GlobalTilt_postop": global_tilt_postop,
        "T4PA_postop": t4pa_postop,
        "L1PA_postop": l1pa_postop,
    }
    
    gap_info = {
        "gap_score": gap_score,
        "gap_category": gap_category,
        "RPV": rpv,
        "RLL": rll,
        "LDI": ldi,
        "RSA": rsa,
    }
    
    return composite, postop_values, gap_info
