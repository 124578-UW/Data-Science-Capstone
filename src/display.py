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
    print(f"\nComposite Score: {result['composite_score']:.4f} (lower is better)")
    print(f"Mechanical Failure Probability: {result['mech_fail_prob'] * 100:.1f}%")

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

    # Add PI-LL (ideal range: 0 to 10)
    ll_preop = patient_fixed.get("LL_preop")
    ll_postop = result["postop_values"].get("LL_postop")
    if pi_val is not None and ll_preop is not None and ll_postop is not None:
        pi_ll_preop = pi_val - ll_preop
        pi_ll_postop = pi_val - ll_postop
        preop_status = "✓" if 0 <= pi_ll_preop <= 10 else "⚠"
        postop_status = "✓" if 0 <= pi_ll_postop <= 10 else "⚠"
        table_data.append({
            "Parameter": "PI-LL",
            "Preop": f"{round(pi_ll_preop, 1)} {preop_status}",
            "Delta (pred)": round(pi_ll_postop - pi_ll_preop, 1),
            "Postop (pred)": f"{round(pi_ll_postop, 1)} {postop_status}",
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
        ("num_levels_cat", "updated_num_levels"),  # Column is named differently in data
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

    # Build comparison table with actual postop values
    # Format: (display_name, preop_col_in_data, postop_col_in_data)
    alignment_params_actual = [
        ("LL", "LL_preop", "LL_postop"),
        ("SS", "SS_preop", "SS_postop"),
        ("L4S1", "L4S1_preop", "L4_S1_postop"),
        ("GlobalTilt", "global_tilt", "global_tilt.1"),
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

    # Add PI-LL (ideal range: 0 to 10)
    pi_preop = patient_row.get("PI_preop")
    pi_postop = patient_row.get("PI_postop", pi_preop)  # PI usually unchanged
    ll_preop_actual = patient_row.get("LL_preop")
    ll_postop_actual = patient_row.get("LL_postop")
    if pd.notna(pi_preop) and pd.notna(ll_preop_actual):
        pi_ll_preop = pi_preop - ll_preop_actual
        preop_status = "✓" if 0 <= pi_ll_preop <= 10 else "⚠"
        if pd.notna(ll_postop_actual):
            pi_ll_postop = (pi_postop if pd.notna(pi_postop) else pi_preop) - ll_postop_actual
            postop_status = "✓" if 0 <= pi_ll_postop <= 10 else "⚠"
            table_actual.append({
                "Parameter": "PI-LL",
                "Preop": f"{round(pi_ll_preop, 1)} {preop_status}",
                "Delta (actual)": round(pi_ll_postop - pi_ll_preop, 1),
                "Postop (actual)": f"{round(pi_ll_postop, 1)} {postop_status}",
            })
        else:
            table_actual.append({
                "Parameter": "PI-LL",
                "Preop": f"{round(pi_ll_preop, 1)} {preop_status}",
                "Delta (actual)": "-",
                "Postop (actual)": "-",
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
        print(f"Mechanical Failure Probability: {row['mech_fail_prob']}")
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
        
        # Add PI-LL (already has status marker from solutions.py)
        pi_ll_postop_str = row.get("PI-LL_postop")
        ll_preop = patient_fixed.get("LL_preop")
        if pi_val is not None and ll_preop is not None and pi_ll_postop_str is not None:
            pi_ll_preop = pi_val - ll_preop
            preop_status = "✓" if 0 <= pi_ll_preop <= 10 else "⚠"
            table_data.append({
                "Parameter": "PI-LL",
                "Preop": f"{round(pi_ll_preop, 1)} {preop_status}",
                "Delta": "-",
                "Postop": pi_ll_postop_str,
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
    
    summary_row = {"Parameter": "Mech Fail Prob"}
    for idx, row in solutions_df.iterrows():
        summary_row[f"Sol {idx+1}"] = row["mech_fail_prob"]
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
    
    # Add PI-LL with status
    pi_ll_row = {"Parameter": "PI-LL"}
    for idx, row in solutions_df.iterrows():
        pi_ll_row[f"Sol {idx+1}"] = row.get("PI-LL_postop", "-")
    rows.append(pi_ll_row)
    
    df_combined = pd.DataFrame(rows)
    
    print("=" * 60)
    print("SOLUTIONS COMPARISON (SIDE BY SIDE)")
    print("=" * 60)
    
    display(df_combined)
    return df_combined
