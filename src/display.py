"""
Display utilities for optimization results.
"""

import pandas as pd
from src import config


def display_optimized_solution(result, patient_fixed):
    """
    Display the optimized solution summary and alignment table.
    
    Args:
        result: Dictionary from ou.evaluate_solution() containing:
            - composite_score, mech_fail_prob, plan, deltas, postop_values, gap_info
        patient_fixed: Dictionary of patient preop parameters
        
    Returns:
        pd.DataFrame: Comparison table of alignment parameters
    """
    print("=" * 60)
    print("BEST SOLUTION SUMMARY (OPTIMIZED)")
    print("=" * 60)
    print(f"\nComposite Score: {result['display_composite_score']:.4f} (lower is better)")
    print(f"Optimization Score: {result['composite_score']:.4f}")
    print(f"Mechanical Failure Probability: {result['mech_fail_prob'] * 100:.1f}%")
    odi_postop = result['postop_values'].get('ODI_postop')
    if odi_postop is not None:
        print(f"Predicted ODI Score: {odi_postop:.1f}")
    else:
        print("Predicted ODI Score: N/A (no preop ODI)")

    print("\nSurgical Plan:")
    for k, v in result['plan'].items():
        print(f"  {k}: {v}")

    # Build comparison table for alignment parameters
    alignment_params = ["LL", "SS", "L4S1", "GlobalTilt", "T4PA", "L1PA"]

    table_data = []
    for param in alignment_params:
        preop_key = f"{param}_preop"
        delta_key = f"delta_{param}"
        postop_key = param
        
        preop_val = patient_fixed.get(preop_key, None)
        delta_val = result["deltas"].get(delta_key, None)
        postop_val = result["postop_values"].get(f"{postop_key}_postop", None)
        
        if preop_val is not None:
            table_data.append({
                "Parameter": param,
                "Preop": round(preop_val, 1),
                "Delta (pred)": round(delta_val, 1) if delta_val else "-",
                "Postop (pred)": round(postop_val, 1) if postop_val else "-",
            })

    # Add SVA with delta if available
    sva_preop = patient_fixed.get("SVA_preop")
    sva_delta = result["deltas"].get("delta_SVA")
    sva_postop = result["postop_values"].get("SVA_postop")
    if sva_preop is not None:
        table_data.append({
            "Parameter": "SVA",
            "Preop": round(sva_preop, 1),
            "Delta (pred)": round(sva_delta, 1) if sva_delta is not None else "-",
            "Postop (pred)": round(sva_postop, 1) if sva_postop is not None else "-",
        })

    # Add PI (unchanged by surgery)
    pi_val = patient_fixed.get("PI_preop")
    if pi_val is not None:
        table_data.append({
            "Parameter": "PI",
            "Preop": round(pi_val, 1),
            "Delta (pred)": "-",
            "Postop (pred)": round(pi_val, 1),
        })

    # Add PT with calculated postop (PT = PI - SS)
    pt_preop = patient_fixed.get("PT_preop")
    ss_postop = result["postop_values"].get("SS_postop")
    if pt_preop is not None and pi_val is not None and ss_postop is not None:
        pt_postop = pi_val - ss_postop
        pt_delta = pt_postop - pt_preop
        table_data.append({
            "Parameter": "PT",
            "Preop": round(pt_preop, 1),
            "Delta (pred)": round(pt_delta, 1),
            "Postop (pred)": round(pt_postop, 1),
        })

    # Add Age (no delta)
    age_val = patient_fixed.get("age")
    if age_val is not None:
        table_data.append({
            "Parameter": "Age",
            "Preop": age_val,
            "Delta (pred)": "-",
            "Postop (pred)": "-",
        })

    # Add GAP Score and GAP Category
    table_data.append({
        "Parameter": "GAP Score",
        "Preop": patient_fixed.get("gap_score_preop", "-"),
        "Delta (pred)": "-",
        "Postop (pred)": result["gap_info"]["gap_score"],
    })
    table_data.append({
        "Parameter": "GAP Category",
        "Preop": patient_fixed.get("gap_category", "-"),
        "Delta (pred)": "→",
        "Postop (pred)": result["gap_info"]["gap_category"],
    })

    # ── CONSTRAINTS section ──
    table_data.append({
        "Parameter": "─── CONSTRAINTS ───",
        "Preop": "", "Delta (pred)": "", "Postop (pred)": "",
    })

    ll_preop = patient_fixed.get("LL_preop")
    ll_postop = result["postop_values"].get("LL_postop")

    # PI-LL (ideal 0-10)
    if pi_val is not None and ll_preop is not None and ll_postop is not None:
        pi_ll_preop = pi_val - ll_preop
        pi_ll_postop = pi_val - ll_postop
        pre_s = "✓" if 0 <= pi_ll_preop <= 10 else "⚠"
        post_s = "✓" if 0 <= pi_ll_postop <= 10 else "⚠"
        table_data.append({
            "Parameter": "PI-LL (0–10)",
            "Preop": f"{round(pi_ll_preop, 1)} {pre_s}",
            "Delta (pred)": round(pi_ll_postop - pi_ll_preop, 1),
            "Postop (pred)": f"{round(pi_ll_postop, 1)} {post_s}",
        })

    # L1PA (|val| ≤ 3)
    l1pa_preop = patient_fixed.get("L1PA_preop")
    l1pa_postop = result["postop_values"].get("L1PA_postop")
    if l1pa_preop is not None and l1pa_postop is not None:
        pre_s = "✓" if abs(l1pa_preop) <= 3 else "⚠"
        post_s = "✓" if abs(l1pa_postop) <= 3 else "⚠"
        table_data.append({
            "Parameter": "L1PA (|val|≤3)",
            "Preop": f"{round(l1pa_preop, 1)} {pre_s}",
            "Delta (pred)": round(l1pa_postop - l1pa_preop, 1),
            "Postop (pred)": f"{round(l1pa_postop, 1)} {post_s}",
        })

    # L4S1 (35-45)
    l4s1_preop = patient_fixed.get("L4S1_preop")
    l4s1_postop = result["postop_values"].get("L4S1_postop")
    if l4s1_preop is not None and l4s1_postop is not None:
        pre_s = "✓" if 35 <= l4s1_preop <= 45 else "⚠"
        post_s = "✓" if 35 <= l4s1_postop <= 45 else "⚠"
        table_data.append({
            "Parameter": "L4S1 (35–45)",
            "Preop": f"{round(l4s1_preop, 1)} {pre_s}",
            "Delta (pred)": round(l4s1_postop - l4s1_preop, 1),
            "Postop (pred)": f"{round(l4s1_postop, 1)} {post_s}",
        })

    # T4PA-L1PA (|diff| ≤ 3)
    t4pa_preop = patient_fixed.get("T4PA_preop")
    t4pa_postop = result["postop_values"].get("T4PA_postop")
    if t4pa_preop is not None and l1pa_preop is not None and t4pa_postop is not None and l1pa_postop is not None:
        diff_pre = t4pa_preop - l1pa_preop
        diff_post = t4pa_postop - l1pa_postop
        pre_s = "✓" if abs(diff_pre) <= 3 else "⚠"
        post_s = "✓" if abs(diff_post) <= 3 else "⚠"
        table_data.append({
            "Parameter": "T4PA−L1PA (|d|≤3)",
            "Preop": f"{round(diff_pre, 1)} {pre_s}",
            "Delta (pred)": round(diff_post - diff_pre, 1),
            "Postop (pred)": f"{round(diff_post, 1)} {post_s}",
        })

    # LL vs ideal (|LL - ideal| ≤ 3)
    from src.scoring import calculate_ideal_ll
    if pi_val is not None and ll_preop is not None and ll_postop is not None:
        ideal_ll = calculate_ideal_ll(pi_val)
        diff_pre = ll_preop - ideal_ll
        diff_post = ll_postop - ideal_ll
        pre_s = "✓" if abs(diff_pre) <= 3 else "⚠"
        post_s = "✓" if abs(diff_post) <= 3 else "⚠"
        table_data.append({
            "Parameter": "LL−ideal (|d|≤3)",
            "Preop": f"{round(diff_pre, 1)} {pre_s}",
            "Delta (pred)": round(diff_post - diff_pre, 1),
            "Postop (pred)": f"{round(diff_post, 1)} {post_s}",
        })

    # GAP Category (ideal: P)
    gap_cat_pre = patient_fixed.get("gap_category", "-")
    gap_cat_post = result["gap_info"]["gap_category"]
    pre_s = "✓" if gap_cat_pre == "P" else "⚠"
    post_s = "✓" if gap_cat_post == "P" else "⚠"
    table_data.append({
        "Parameter": "GAP Cat (P)",
        "Preop": f"{gap_cat_pre} {pre_s}",
        "Delta (pred)": "→",
        "Postop (pred)": f"{gap_cat_post} {post_s}",
    })

    print("\n" + "=" * 60)
    print("ALIGNMENT PARAMETERS: PREOP → POSTOP (PREDICTED)")
    print("=" * 60)
    
    return pd.DataFrame(table_data)


def display_actual_outcomes(patient_id, patient_fixed, data_path=None):
    """
    Display the actual surgical plan and outcomes for a patient.
    
    Args:
        patient_id: Patient ID to look up
        patient_fixed: Dictionary of patient preop parameters (for reference)
        data_path: Path to processed data CSV (defaults to config.DATA_PROCESSED)
        
    Returns:
        pd.DataFrame: Comparison table of actual alignment parameters
    """
    if data_path is None:
        data_path = config.DATA_PROCESSED
        
    df_full = pd.read_csv(data_path)
    patient_row = df_full[df_full["id"] == patient_id].iloc[0]

    # Actual surgical plan columns (map display name to data column name)
    actual_plan_cols = [
        ("UIV_implant", "UIV_implant"),
        ("num_levels_cat", "num_levels_cat"),
        ("num_interbody_fusion_levels", "num_interbody_fusion_levels"),
        ("ALIF", "ALIF"),
        ("XLIF", "XLIF"),
        ("TLIF", "TLIF"),
        ("num_rods", "num_rods"),
        ("num_pelvic_screws", "num_pelvic_screws"),
        ("osteotomy", "osteotomy"),
    ]

    print("=" * 60)
    print("ACTUAL SURGICAL PLAN (WHAT WAS PERFORMED)")
    print("=" * 60)
    for display_name, col in actual_plan_cols:
        if col in patient_row.index:
            print(f"  {display_name}: {patient_row[col]}")

    # Show key outcomes
    mech_fail = patient_row.get("mech_fail_last")
    if mech_fail is not None and pd.notna(mech_fail):
        print(f"\n  Mechanical Failure: {'Yes' if mech_fail else 'No'}")

    odi_pre = patient_row.get("ODI_preop")
    odi_post = patient_row.get("ODI_12mo")
    odi_pre_str = f"{round(odi_pre, 1)}" if odi_pre is not None and pd.notna(odi_pre) else "N/A"
    odi_post_str = f"{round(odi_post, 1)}" if odi_post is not None and pd.notna(odi_post) else "N/A"
    print(f"  ODI: {odi_pre_str} (preop) → {odi_post_str} (12mo)")

    # Build comparison table with actual postop values
    # Format: (display_name, preop_col_in_data, postop_col_in_data)
    alignment_params_actual = [
        ("LL", "LL_preop", "LL_postop"),
        ("SS", "SS_preop", "SS_postop"),
        ("L4S1", "L4S1_preop", "L4_S1_postop"),
        ("GlobalTilt", "global_tilt_preop", "global_tilt_postop"),
        ("T4PA", "T4PA_preop", "T4PA_postop"),
        ("L1PA", "L1PA_preop", "L1PA_postop"),
    ]

    table_actual = []
    for param, preop_col, postop_col in alignment_params_actual:
        preop_val = patient_row.get(preop_col, None)
        postop_val = patient_row.get(postop_col, None) if postop_col else None
        
        if pd.notna(preop_val) and pd.notna(postop_val):
            delta_val = postop_val - preop_val
        else:
            delta_val = None
        
        if preop_val is not None and pd.notna(preop_val):
            table_actual.append({
                "Parameter": param,
                "Preop": round(preop_val, 1),
                "Delta (actual)": round(delta_val, 1) if delta_val is not None else "-",
                "Postop (actual)": round(postop_val, 1) if postop_val is not None and pd.notna(postop_val) else "-",
            })

    # Add other parameters
    other_actual = [
        ("PI", "PI_preop", "PI_postop"),
        ("PT", "PT_preop", "PT_postop"),
        ("SVA", "SVA_preop", "SVA_postop"),
    ]
    for name, preop_col, postop_col in other_actual:
        preop_val = patient_row.get(preop_col, None)
        postop_val = patient_row.get(postop_col, None) if postop_col else None
        
        if pd.notna(preop_val) and pd.notna(postop_val):
            delta_val = postop_val - preop_val
        else:
            delta_val = None
        
        if preop_val is not None and pd.notna(preop_val):
            table_actual.append({
                "Parameter": name,
                "Preop": round(preop_val, 1) if isinstance(preop_val, float) else preop_val,
                "Delta (actual)": round(delta_val, 1) if delta_val is not None else "-",
                "Postop (actual)": round(postop_val, 1) if postop_val is not None and pd.notna(postop_val) else "-",
            })

    # Add Age
    age_val = patient_row.get("age")
    if age_val is not None and pd.notna(age_val):
        table_actual.append({
            "Parameter": "Age",
            "Preop": age_val if not isinstance(age_val, float) else round(age_val, 0),
            "Delta (actual)": "-",
            "Postop (actual)": "-",
        })

    # Add GAP Score and Category
    table_actual.append({
        "Parameter": "GAP Score",
        "Preop": patient_row.get("gap_score_preop", "-"),
        "Delta (actual)": "-",
        "Postop (actual)": patient_row.get("gap_score_postop", "-"),
    })
    table_actual.append({
        "Parameter": "GAP Category",
        "Preop": patient_row.get("gap_category", "-"),
        "Delta (actual)": "→",
        "Postop (actual)": patient_row.get("gap_category_postop", "-"),
    })

    # ─── CONSTRAINTS section ───
    table_actual.append({
        "Parameter": "─── CONSTRAINTS ───",
        "Preop": "", "Delta (actual)": "", "Postop (actual)": "",
    })

    pi_preop = patient_row.get("PI_preop")
    pi_postop = patient_row.get("PI_postop", pi_preop)
    ll_preop_actual = patient_row.get("LL_preop")
    ll_postop_actual = patient_row.get("LL_postop")

    # PI-LL (ideal 0-10)
    if pd.notna(pi_preop) and pd.notna(ll_preop_actual):
        pi_ll_pre = pi_preop - ll_preop_actual
        pre_s = "✓" if 0 <= pi_ll_pre <= 10 else "⚠"
        if pd.notna(ll_postop_actual):
            pi_ll_post = (pi_postop if pd.notna(pi_postop) else pi_preop) - ll_postop_actual
            post_s = "✓" if 0 <= pi_ll_post <= 10 else "⚠"
            table_actual.append({
                "Parameter": "PI-LL (0–10)",
                "Preop": f"{round(pi_ll_pre, 1)} {pre_s}",
                "Delta (actual)": round(pi_ll_post - pi_ll_pre, 1),
                "Postop (actual)": f"{round(pi_ll_post, 1)} {post_s}",
            })
        else:
            table_actual.append({
                "Parameter": "PI-LL (0–10)",
                "Preop": f"{round(pi_ll_pre, 1)} {pre_s}",
                "Delta (actual)": "-",
                "Postop (actual)": "-",
            })

    # L1PA (|val| ≤ 3)
    l1pa_pre = patient_row.get("L1PA_preop")
    l1pa_post = patient_row.get("L1PA_postop")
    if pd.notna(l1pa_pre):
        pre_s = "✓" if abs(l1pa_pre) <= 3 else "⚠"
        if pd.notna(l1pa_post):
            post_s = "✓" if abs(l1pa_post) <= 3 else "⚠"
            table_actual.append({
                "Parameter": "L1PA (|val|≤3)",
                "Preop": f"{round(l1pa_pre, 1)} {pre_s}",
                "Delta (actual)": round(l1pa_post - l1pa_pre, 1),
                "Postop (actual)": f"{round(l1pa_post, 1)} {post_s}",
            })

    # L4S1 (35-45)
    l4s1_pre = patient_row.get("L4S1_preop")
    l4s1_post = patient_row.get("L4_S1_postop")
    if pd.notna(l4s1_pre):
        pre_s = "✓" if 35 <= l4s1_pre <= 45 else "⚠"
        if pd.notna(l4s1_post):
            post_s = "✓" if 35 <= l4s1_post <= 45 else "⚠"
            table_actual.append({
                "Parameter": "L4S1 (35–45)",
                "Preop": f"{round(l4s1_pre, 1)} {pre_s}",
                "Delta (actual)": round(l4s1_post - l4s1_pre, 1),
                "Postop (actual)": f"{round(l4s1_post, 1)} {post_s}",
            })

    # T4PA−L1PA (|diff| ≤ 3)
    t4pa_pre = patient_row.get("T4PA_preop")
    t4pa_post = patient_row.get("T4PA_postop")
    if pd.notna(t4pa_pre) and pd.notna(l1pa_pre) and pd.notna(t4pa_post) and pd.notna(l1pa_post):
        diff_pre = t4pa_pre - l1pa_pre
        diff_post = t4pa_post - l1pa_post
        pre_s = "✓" if abs(diff_pre) <= 3 else "⚠"
        post_s = "✓" if abs(diff_post) <= 3 else "⚠"
        table_actual.append({
            "Parameter": "T4PA−L1PA (|d|≤3)",
            "Preop": f"{round(diff_pre, 1)} {pre_s}",
            "Delta (actual)": round(diff_post - diff_pre, 1),
            "Postop (actual)": f"{round(diff_post, 1)} {post_s}",
        })

    # LL vs ideal (|LL - ideal| ≤ 3)
    from src.scoring import calculate_ideal_ll
    if pd.notna(pi_preop) and pd.notna(ll_preop_actual):
        ideal_ll = calculate_ideal_ll(pi_preop)
        diff_pre = ll_preop_actual - ideal_ll
        pre_s = "✓" if abs(diff_pre) <= 3 else "⚠"
        if pd.notna(ll_postop_actual):
            diff_post = ll_postop_actual - ideal_ll
            post_s = "✓" if abs(diff_post) <= 3 else "⚠"
            table_actual.append({
                "Parameter": "LL−ideal (|d|≤3)",
                "Preop": f"{round(diff_pre, 1)} {pre_s}",
                "Delta (actual)": round(diff_post - diff_pre, 1),
                "Postop (actual)": f"{round(diff_post, 1)} {post_s}",
            })

    # GAP Category (ideal: P)
    gap_cat_pre = patient_row.get("gap_category", "-")
    gap_cat_post = patient_row.get("gap_category_postop", "-")
    if gap_cat_pre != "-":
        pre_s = "✓" if gap_cat_pre == "P" else "⚠"
        post_s = "✓" if gap_cat_post == "P" else "⚠"
        table_actual.append({
            "Parameter": "GAP Cat (P)",
            "Preop": f"{gap_cat_pre} {pre_s}",
            "Delta (actual)": "→",
            "Postop (actual)": f"{gap_cat_post} {post_s}",
        })

    print("\n" + "=" * 60)
    print("ALIGNMENT PARAMETERS: PREOP → POSTOP (ACTUAL)")
    print("=" * 60)
    
    return pd.DataFrame(table_actual)


def display_multiple_solutions(solutions_df, patient_fixed, side_by_side=True):
    """
    Display multiple solutions in detailed format (similar to best solution display).
    Uses precomputed values from get_diverse_solutions() - no re-evaluation needed.
    
    Args:
        solutions_df: DataFrame from get_diverse_solutions() with postop values already computed
        patient_fixed: Dictionary of patient preop parameters
        side_by_side: If True, display solutions side by side; if False, display vertically
        
    Returns:
        DataFrame with all solutions side by side (if side_by_side=True), else list of DataFrames
    """
    from src import config
    from IPython.display import HTML
    
    if side_by_side:
        return _display_solutions_side_by_side(solutions_df, patient_fixed)
    
    alignment_tables = []
    
    for idx, row in solutions_df.iterrows():
        print("\n" + "=" * 60)
        print(f"SOLUTION {idx + 1}")
        print("=" * 60)
        print(f"\nComposite Score: {row['composite_score']}")
        print(f"Optimization Score: {row.get('optimization_score', '-')}")
        print(f"Mechanical Failure Probability: {row['mech_fail_prob']}")
        odi_postop = row.get('ODI_postop')
        if odi_postop is not None:
            print(f"Predicted ODI Score: {odi_postop}")
        else:
            print("Predicted ODI Score: N/A (no preop ODI)")
        print(f"GAP Score: {row['gap_score']} ({row['gap_category']})")
        
        print("\nSurgical Plan:")
        for col in config.PLAN_COLS:
            if col in row.index:
                print(f"  {col}: {row[col]}")
        
        # Build alignment table from precomputed postop values
        alignment_params = ["LL", "SS", "L4S1", "GlobalTilt", "T4PA", "L1PA"]
        table_data = []
        
        for param in alignment_params:
            preop_key = f"{param}_preop"
            postop_key = f"{param}_postop"
            
            preop_val = patient_fixed.get(preop_key, None)
            postop_val = row.get(postop_key, None)
            
            if preop_val is not None and postop_val is not None:
                table_data.append({
                    "Parameter": param,
                    "Preop": round(preop_val, 1),
                    "Delta": round(postop_val - preop_val, 1),
                    "Postop": round(postop_val, 1),
                })
        
        # Add SVA
        sva_preop = patient_fixed.get("SVA_preop")
        sva_postop = row.get("SVA_postop")
        if sva_preop is not None and sva_postop is not None:
            table_data.append({
                "Parameter": "SVA",
                "Preop": round(sva_preop, 1),
                "Delta": round(sva_postop - sva_preop, 1),
                "Postop": round(sva_postop, 1),
            })
        
        # Add PI (unchanged)
        pi_val = patient_fixed.get("PI_preop")
        if pi_val is not None:
            table_data.append({
                "Parameter": "PI",
                "Preop": round(pi_val, 1),
                "Delta": "-",
                "Postop": round(pi_val, 1),
            })
        
        # Add PT (PT = PI - SS)
        pt_preop = patient_fixed.get("PT_preop")
        ss_postop = row.get("SS_postop")
        if pt_preop is not None and pi_val is not None and ss_postop is not None:
            pt_postop = pi_val - ss_postop
            table_data.append({
                "Parameter": "PT",
                "Preop": round(pt_preop, 1),
                "Delta": round(pt_postop - pt_preop, 1),
                "Postop": round(pt_postop, 1),
            })
        
        # Add GAP Score and GAP Category
        gap_score_preop = patient_fixed.get("gap_score_preop", "-")
        gap_category_preop = patient_fixed.get("gap_category", "-")
        table_data.append({
            "Parameter": "GAP Score",
            "Preop": gap_score_preop,
            "Delta": "-",
            "Postop": row.get("gap_score", "-"),
        })
        table_data.append({
            "Parameter": "GAP Category",
            "Preop": gap_category_preop,
            "Delta": "→",
            "Postop": row.get("gap_category", "-"),
        })
        
        # ─── CONSTRAINTS section ───
        table_data.append({
            "Parameter": "─── CONSTRAINTS ───",
            "Preop": "", "Delta": "", "Postop": "",
        })

        ll_preop = patient_fixed.get("LL_preop")
        ll_postop_v = row.get("LL_postop")

        # PI-LL (ideal 0-10)
        if pi_val is not None and ll_preop is not None and ll_postop_v is not None:
            pi_ll_pre = pi_val - ll_preop
            pi_ll_post = pi_val - ll_postop_v
            pre_s = "✓" if 0 <= pi_ll_pre <= 10 else "⚠"
            post_s = "✓" if 0 <= pi_ll_post <= 10 else "⚠"
            table_data.append({
                "Parameter": "PI-LL (0–10)",
                "Preop": f"{round(pi_ll_pre, 1)} {pre_s}",
                "Delta": round(pi_ll_post - pi_ll_pre, 1),
                "Postop": f"{round(pi_ll_post, 1)} {post_s}",
            })

        # L1PA (|val| ≤ 3)
        l1pa_pre = patient_fixed.get("L1PA_preop")
        l1pa_post = row.get("L1PA_postop")
        if l1pa_pre is not None and l1pa_post is not None:
            pre_s = "✓" if abs(l1pa_pre) <= 3 else "⚠"
            post_s = "✓" if abs(l1pa_post) <= 3 else "⚠"
            table_data.append({
                "Parameter": "L1PA (|val|≤3)",
                "Preop": f"{round(l1pa_pre, 1)} {pre_s}",
                "Delta": round(l1pa_post - l1pa_pre, 1),
                "Postop": f"{round(l1pa_post, 1)} {post_s}",
            })

        # L4S1 (35-45)
        l4s1_pre = patient_fixed.get("L4S1_preop")
        l4s1_post = row.get("L4S1_postop")
        if l4s1_pre is not None and l4s1_post is not None:
            pre_s = "✓" if 35 <= l4s1_pre <= 45 else "⚠"
            post_s = "✓" if 35 <= l4s1_post <= 45 else "⚠"
            table_data.append({
                "Parameter": "L4S1 (35–45)",
                "Preop": f"{round(l4s1_pre, 1)} {pre_s}",
                "Delta": round(l4s1_post - l4s1_pre, 1),
                "Postop": f"{round(l4s1_post, 1)} {post_s}",
            })

        # T4PA−L1PA (|diff| ≤ 3)
        t4pa_pre = patient_fixed.get("T4PA_preop")
        t4pa_post = row.get("T4PA_postop")
        if t4pa_pre is not None and l1pa_pre is not None and t4pa_post is not None and l1pa_post is not None:
            diff_pre = t4pa_pre - l1pa_pre
            diff_post = t4pa_post - l1pa_post
            pre_s = "✓" if abs(diff_pre) <= 3 else "⚠"
            post_s = "✓" if abs(diff_post) <= 3 else "⚠"
            table_data.append({
                "Parameter": "T4PA−L1PA (|d|≤3)",
                "Preop": f"{round(diff_pre, 1)} {pre_s}",
                "Delta": round(diff_post - diff_pre, 1),
                "Postop": f"{round(diff_post, 1)} {post_s}",
            })

        # LL vs ideal (|LL - ideal| ≤ 3)
        from src.scoring import calculate_ideal_ll
        if pi_val is not None and ll_preop is not None and ll_postop_v is not None:
            ideal_ll = calculate_ideal_ll(pi_val)
            diff_pre = ll_preop - ideal_ll
            diff_post = ll_postop_v - ideal_ll
            pre_s = "✓" if abs(diff_pre) <= 3 else "⚠"
            post_s = "✓" if abs(diff_post) <= 3 else "⚠"
            table_data.append({
                "Parameter": "LL−ideal (|d|≤3)",
                "Preop": f"{round(diff_pre, 1)} {pre_s}",
                "Delta": round(diff_post - diff_pre, 1),
                "Postop": f"{round(diff_post, 1)} {post_s}",
            })

        # GAP Category (ideal: P)
        gap_cat_post = row.get("gap_category", "-")
        pre_s = "✓" if gap_category_preop == "P" else "⚠"
        post_s = "✓" if gap_cat_post == "P" else "⚠"
        table_data.append({
            "Parameter": "GAP Cat (P)",
            "Preop": f"{gap_category_preop} {pre_s}",
            "Delta": "→",
            "Postop": f"{gap_cat_post} {post_s}",
        })
        
        df_table = pd.DataFrame(table_data)
        alignment_tables.append(df_table)
        
        print("\nAlignment Parameters:")
        display(df_table)
    
    return alignment_tables


def _display_solutions_side_by_side(solutions_df, patient_fixed):
    """
    Display multiple solutions side by side in a single table.
    
    Args:
        solutions_df: DataFrame from get_diverse_solutions()
        patient_fixed: Dictionary of patient preop parameters
        
    Returns:
        DataFrame with solutions displayed side by side
    """
    from src import config
    
    # Parameters to display
    alignment_params = ["LL", "SS", "L4S1", "GlobalTilt", "T4PA", "L1PA", "SVA"]
    pi_val = patient_fixed.get("PI_preop")
    
    # Build combined table
    rows = []
    
    # Header rows: summary info for each solution
    summary_row = {"Parameter": "Composite Score"}
    for idx, row in solutions_df.iterrows():
        summary_row[f"Sol {idx+1}"] = row["composite_score"]
    rows.append(summary_row)
    
    if "optimization_score" in solutions_df.columns:
        summary_row = {"Parameter": "Optimization Score"}
        for idx, row in solutions_df.iterrows():
            summary_row[f"Sol {idx+1}"] = row.get("optimization_score", "-")
        rows.append(summary_row)
    
    summary_row = {"Parameter": "Mech Fail Prob"}
    for idx, row in solutions_df.iterrows():
        summary_row[f"Sol {idx+1}"] = row["mech_fail_prob"]
    rows.append(summary_row)
    
    # ODI postop (always show)
    summary_row = {"Parameter": "Predicted ODI"}
    for idx, row in solutions_df.iterrows():
        odi_val = row.get("ODI_postop") if "ODI_postop" in solutions_df.columns else None
        summary_row[f"Sol {idx+1}"] = odi_val if odi_val is not None else "N/A (no preop ODI)"
    rows.append(summary_row)
    
    # GAP Score: show preop → postop (include categories)
    gap_score_pre = patient_fixed.get("gap_score_preop", "-")
    gap_cat_pre = patient_fixed.get("gap_category", "-")
    summary_row = {"Parameter": "GAP Score"}
    for idx, row in solutions_df.iterrows():
        post_score = row.get("gap_score", "-")
        post_cat = row.get("gap_category", "-")
        summary_row[f"Sol {idx+1}"] = f"{gap_score_pre} ({gap_cat_pre}) → {post_score} ({post_cat})"
    rows.append(summary_row)
    
    # Separator
    rows.append({"Parameter": "─" * 12, **{f"Sol {idx+1}": "─" * 10 for idx in range(len(solutions_df))}})
    
    # Surgical plan rows
    rows.append({"Parameter": "SURGICAL PLAN", **{f"Sol {idx+1}": "" for idx in range(len(solutions_df))}})
    for col in config.PLAN_COLS:
        plan_row = {"Parameter": col}
        for idx, row in solutions_df.iterrows():
            plan_row[f"Sol {idx+1}"] = row.get(col, "-")
        rows.append(plan_row)
    
    # Separator
    rows.append({"Parameter": "─" * 12, **{f"Sol {idx+1}": "─" * 10 for idx in range(len(solutions_df))}})
    
    # Postop values
    rows.append({"Parameter": "POSTOP VALUES", **{f"Sol {idx+1}": "" for idx in range(len(solutions_df))}})
    
    for param in alignment_params:
        postop_key = f"{param}_postop"
        param_row = {"Parameter": param}
        for idx, row in solutions_df.iterrows():
            val = row.get(postop_key)
            param_row[f"Sol {idx+1}"] = round(val, 1) if val is not None else "-"
        rows.append(param_row)
    
    # Add PI (unchanged)
    if pi_val is not None:
        pi_row = {"Parameter": "PI"}
        for idx in range(len(solutions_df)):
            pi_row[f"Sol {idx+1}"] = round(pi_val, 1)
        rows.append(pi_row)
    
    # Add PT (calculated)
    pt_preop = patient_fixed.get("PT_preop")
    if pt_preop is not None and pi_val is not None:
        pt_row = {"Parameter": "PT"}
        for idx, row in solutions_df.iterrows():
            ss_postop = row.get("SS_postop")
            if ss_postop is not None:
                pt_row[f"Sol {idx+1}"] = round(pi_val - ss_postop, 1)
            else:
                pt_row[f"Sol {idx+1}"] = "-"
        rows.append(pt_row)
    
    # Separator
    rows.append({"Parameter": "─" * 12, **{f"Sol {idx+1}": "─" * 10 for idx in range(len(solutions_df))}})
    
    # Constraints section
    rows.append({"Parameter": "CONSTRAINTS", **{f"Sol {idx+1}": "" for idx in range(len(solutions_df))}})
    
    # PI-LL (ideal: 0-10)
    pi_ll_row = {"Parameter": "PI-LL (0–10)"}
    for idx, row in solutions_df.iterrows():
        ll_post = row.get("LL_postop")
        if pi_val is not None and ll_post is not None:
            val = pi_val - ll_post
            status = "✓" if 0 <= val <= 10 else "⚠"
            pi_ll_row[f"Sol {idx+1}"] = f"{val:.1f} {status}"
        else:
            pi_ll_row[f"Sol {idx+1}"] = "-"
    rows.append(pi_ll_row)
    
    # L1PA (ideal: |L1PA| ≤ 3)
    l1pa_row = {"Parameter": "L1PA (|val|≤3)"}
    for idx, row in solutions_df.iterrows():
        val = row.get("L1PA_postop")
        if val is not None:
            status = "✓" if abs(val) <= 3 else "⚠"
            l1pa_row[f"Sol {idx+1}"] = f"{val:.1f} {status}"
        else:
            l1pa_row[f"Sol {idx+1}"] = "-"
    rows.append(l1pa_row)
    
    # L4S1 (ideal: 35-45)
    l4s1_row = {"Parameter": "L4S1 (35–45)"}
    for idx, row in solutions_df.iterrows():
        val = row.get("L4S1_postop")
        if val is not None:
            status = "✓" if 35 <= val <= 45 else "⚠"
            l4s1_row[f"Sol {idx+1}"] = f"{val:.1f} {status}"
        else:
            l4s1_row[f"Sol {idx+1}"] = "-"
    rows.append(l4s1_row)
    
    # T4PA-L1PA (ideal: |diff| ≤ 3)
    t4l1_row = {"Parameter": "T4PA−L1PA (|d|≤3)"}
    for idx, row in solutions_df.iterrows():
        t4pa = row.get("T4PA_postop")
        l1pa = row.get("L1PA_postop")
        if t4pa is not None and l1pa is not None:
            diff = t4pa - l1pa
            status = "✓" if abs(diff) <= 3 else "⚠"
            t4l1_row[f"Sol {idx+1}"] = f"{diff:.1f} {status}"
        else:
            t4l1_row[f"Sol {idx+1}"] = "-"
    rows.append(t4l1_row)
    
    # LL vs ideal (ideal LL = PI*0.54 + 27.6, tolerance ±3)
    from src.scoring import calculate_ideal_ll
    ll_row = {"Parameter": "LL−ideal (|d|≤3)"}
    for idx, row in solutions_df.iterrows():
        ll_post = row.get("LL_postop")
        if pi_val is not None and ll_post is not None:
            ideal_ll = calculate_ideal_ll(pi_val)
            diff = ll_post - ideal_ll
            status = "✓" if abs(diff) <= 3 else "⚠"
            ll_row[f"Sol {idx+1}"] = f"{diff:.1f} {status}"
        else:
            ll_row[f"Sol {idx+1}"] = "-"
    rows.append(ll_row)
    
    # GAP Category (ideal: P)
    gap_row = {"Parameter": "GAP Category (P)"}
    for idx, row in solutions_df.iterrows():
        cat = row.get("gap_category", "-")
        status = "✓" if cat == "P" else "⚠"
        gap_row[f"Sol {idx+1}"] = f"{cat} {status}"
    rows.append(gap_row)
    
    df_combined = pd.DataFrame(rows)
    
    print("=" * 60)
    print("SOLUTIONS COMPARISON")
    print("=" * 60)
    
    display(df_combined)
    return df_combined
