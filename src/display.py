"""
Display utilities for optimization results.
"""

import pandas as pd
from src import config

# Display-name overrides for plan columns
_PLAN_COL_DISPLAY = {
    "osteotomy": "osteotomy (PSO)",
}


def display_patient_preop(patient_fixed, patient_id=None, holdout_row=None):
    """
    Display a compact summary of the patient's preop measurements and attributes
    as a two-column styled DataFrame table matching the results table format.

    Args:
        patient_fixed: dict of patient preoperative parameters
        patient_id: optional patient ID for the header
        holdout_row: optional Series from holdout_df to show actual outcomes
    """
    from IPython.display import display as ipy_display, Markdown

    pi = patient_fixed.get("PI_preop")
    ll = patient_fixed.get("LL_preop")

    def _fmt(v, decimals=1):
        if v is None:
            return "–"
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)

    def _yes_no(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "–"
        return "Yes" if bool(v) else "No"

    rows = []

    # ── Demographics ──
    rows.append({"Parameter": "DEMOGRAPHICS", "Value": ""})
    rows.append({"Parameter": "Age", "Value": _fmt(patient_fixed.get("age"), 0)})
    rows.append({"Parameter": "Sex", "Value": _fmt(patient_fixed.get("sex"))})
    rows.append({"Parameter": "BMI", "Value": _fmt(patient_fixed.get("bmi"))})
    rows.append({"Parameter": "ASA Class", "Value": _fmt(patient_fixed.get("ASA_CLASS"))})
    rows.append({"Parameter": "CCI", "Value": _fmt(patient_fixed.get("CCI"))})
    rows.append({"Parameter": "Revision", "Value": "Yes" if patient_fixed.get("revision") else "No"})
    rows.append({"Parameter": "Smoking", "Value": _yes_no(patient_fixed.get("smoking"))})

    rows.append({"Parameter": "─" * 12, "Value": "─" * 10})

    # ── Alignment ──
    rows.append({"Parameter": "ALIGNMENT", "Value": ""})
    rows.append({"Parameter": "PI", "Value": _fmt(pi)})
    rows.append({"Parameter": "LL", "Value": _fmt(ll)})
    rows.append({"Parameter": "SS", "Value": _fmt(patient_fixed.get("SS_preop"))})
    rows.append({"Parameter": "PT", "Value": _fmt(patient_fixed.get("PT_preop"))})
    rows.append({"Parameter": "L4-S1", "Value": _fmt(patient_fixed.get("L4S1_preop"))})
    rows.append({"Parameter": "T4PA", "Value": _fmt(patient_fixed.get("T4PA_preop"))})
    rows.append({"Parameter": "L1PA", "Value": _fmt(patient_fixed.get("L1PA_preop"))})
    rows.append({"Parameter": "SVA", "Value": _fmt(patient_fixed.get("SVA_preop"))})
    rows.append({"Parameter": "Global Tilt", "Value": _fmt(patient_fixed.get("global_tilt_preop"))})
    rows.append({"Parameter": "C7-CSVL", "Value": _fmt(patient_fixed.get("C7CSVL_preop"))})
    rows.append({"Parameter": "Cobb (main)", "Value": _fmt(patient_fixed.get("cobb_main_curve_preop"))})

    rows.append({"Parameter": "─" * 12, "Value": "─" * 10})

    # ── Bone Quality ──
    rows.append({"Parameter": "BONE QUALITY", "Value": ""})
    rows.append({"Parameter": "T-score (fem neck)", "Value": _fmt(patient_fixed.get("tscore_femneck_preop"))})
    rows.append({"Parameter": "HU UIV", "Value": _fmt(patient_fixed.get("HU_UIV_preop"))})
    rows.append({"Parameter": "HU UIV+1", "Value": _fmt(patient_fixed.get("HU_UIVplus1_preop"))})
    rows.append({"Parameter": "HU UIV+2", "Value": _fmt(patient_fixed.get("HU_UIVplus2_preop"))})
    rows.append({"Parameter": "Frailty (FC)", "Value": _fmt(patient_fixed.get("FC_preop"))})

    rows.append({"Parameter": "─" * 12, "Value": "─" * 10})

    # ── Functional ──
    rows.append({"Parameter": "PRE-SURGERY SCORES", "Value": ""})
    rows.append({"Parameter": "ODI", "Value": _fmt(patient_fixed.get("ODI_preop"))})
    rows.append({"Parameter": "GAP Score", "Value": _fmt(patient_fixed.get("gap_score_preop"))})
    rows.append({"Parameter": "GAP Category", "Value": _fmt(patient_fixed.get("gap_category"))})

    # ── Actual Outcomes ──
    if holdout_row is not None:
        outcome_items = []
        mf = holdout_row.get("mech_fail_last")
        if mf is not None and pd.notna(mf):
            outcome_items.append(("Mech Failure (actual)", "Yes" if mf else "No"))
        odi_post = holdout_row.get("ODI_12mo")
        if odi_post is not None and pd.notna(odi_post):
            outcome_items.append(("ODI 12mo (actual)", _fmt(odi_post)))
        if outcome_items:
            rows.append({"Parameter": "─" * 12, "Value": "─" * 10})
            rows.append({"Parameter": "ACTUAL OUTCOMES", "Value": ""})
            for label, val in outcome_items:
                rows.append({"Parameter": label, "Value": val})

    df = pd.DataFrame(rows)

    if patient_id is not None:
        ipy_display(Markdown(f"**Patient {patient_id} — Preop Profile**"))

    styled = df.style.hide(axis="index")
    ipy_display(styled)


def display_best_per_scenario(all_results, patient_fixed, actual_eval, scenarios,
                              deduplicate=True, include_actual=True,
                              min_plan_diff=2, candidate_depth=12):
    """
    Display the single best solution from each scenario (and optionally the
    actual plan) with columns labeled by scenario name.

    When *deduplicate=True*, scenarios are selected to maximize cross-scenario
    plan variety while still preferring stronger objective scores.

    Args:
        all_results: dict mapping scenario key → run result dict (from runs.run_optimization)
        patient_fixed: dict of patient preop parameters
        actual_eval: dict from ou.evaluate_solution() for the actual surgical plan
            (required only when include_actual=True)
        scenarios: list of scenario keys to display (in order)
        deduplicate: if True, select scenario plans with diversity-aware logic
        include_actual: if True, include an "Actual" column; if False, show only scenarios
        min_plan_diff: minimum number of plan fields that should differ from
            already-selected scenarios when alternatives exist
        candidate_depth: max number of candidate rows to consider per scenario

    Returns:
        DataFrame with combined table (unstyled)
    """
    from src.scoring import calculate_ideal_l1pa, calculate_ideal_ll
    from IPython.display import display as ipy_display

    alignment_params = ["LL", "SS", "L4S1", "GlobalTilt", "T4PA", "L1PA", "SVA"]
    pi_val = patient_fixed.get("PI_preop")

    col_actual = "Actual"

    if include_actual and actual_eval is None:
        raise ValueError("actual_eval is required when include_actual=True")

    def _plan_key(result_dict):
        """Return a hashable tuple of plan values for dedup."""
        return tuple(result_dict["plan"].get(c) for c in config.PLAN_COLS)

    def _plan_distance(plan_a, plan_b):
        """Hamming distance between two plan tuples."""
        return sum(a != b for a, b in zip(plan_a, plan_b))

    def _diverse_row_to_result(row):
        """Convert a diverse_df row into a best_result-style dict."""
        plan = {c: row[c] for c in config.PLAN_COLS}
        postop = {k: row[k] for k in row.index if k.endswith("_postop") and k != "PI-LL_postop"}
        mfp_str = row.get("mech_fail_prob", "0%")
        mfp = float(str(mfp_str).replace("%", "")) / 100
        return {
            "plan": plan,
            "display_composite_score": row.get("composite_score", 0),
            "composite_score": row.get("optimization_score", 0),
            "mech_fail_prob": mfp,
            "postop_values": postop,
            "gap_info": {
                "gap_score": row.get("gap_score", "-"),
                "gap_category": row.get("gap_category", "-"),
            },
        }

    def _result_to_candidate(result_dict):
        return {
            "result": result_dict,
            "plan_key": _plan_key(result_dict),
            "score": float(result_dict.get("composite_score", 0.0)),
        }

    # Build best results, optionally deduplicated/diversified
    seen_plan_keys = []
    scenario_cols = []
    best_results = []
    for key in scenarios:
        if key not in all_results:
            continue
        r = all_results[key]
        label = r["label"]
        scenario_cols.append(label)
        br = r["best_result"]
        best_candidate = _result_to_candidate(br)

        if not deduplicate:
            best_results.append(br)
            continue

        candidates = [best_candidate]
        diverse_df = r.get("diverse_df")
        if diverse_df is not None and not diverse_df.empty:
            for _, drow in diverse_df.head(candidate_depth).iterrows():
                cand_result = _diverse_row_to_result(drow)
                candidates.append(_result_to_candidate(cand_result))

        # Keep one candidate per unique plan key (best score wins)
        unique_candidates = {}
        for candidate in candidates:
            pk = candidate["plan_key"]
            prev = unique_candidates.get(pk)
            if prev is None or candidate["score"] < prev["score"]:
                unique_candidates[pk] = candidate
        candidates = list(unique_candidates.values())

        if not seen_plan_keys:
            selected = min(candidates, key=lambda c: c["score"])
        else:
            scored_candidates = []
            for candidate in candidates:
                min_dist = min(_plan_distance(candidate["plan_key"], spk) for spk in seen_plan_keys)
                scored_candidates.append((candidate, min_dist))

            preferred = [item for item in scored_candidates if item[1] >= min_plan_diff]
            pool = preferred if preferred else scored_candidates

            selected, _ = min(
                pool,
                key=lambda item: (-item[1], item[0]["score"]),
            )

        best_results.append(selected["result"])
        seen_plan_keys.append(selected["plan_key"])

    all_cols = ([col_actual] if include_actual else []) + scenario_cols
    rows = []

    def _f1(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) else str(v)

    # ── Summary rows ──
    row = {"Parameter": "Alignment Score"}
    if include_actual:
        row[col_actual] = _f1(actual_eval["display_composite_score"])
    for label, br in zip(scenario_cols, best_results):
        row[label] = _f1(br["display_composite_score"])
    rows.append(row)

    row = {"Parameter": "Optimization Score"}
    if include_actual:
        row[col_actual] = _f1(actual_eval["composite_score"])
    for label, br in zip(scenario_cols, best_results):
        row[label] = _f1(br["composite_score"])
    rows.append(row)

    row = {"Parameter": "Mech Fail Prob"}
    if include_actual:
        row[col_actual] = f"{actual_eval['mech_fail_prob']*100:.1f}%"
    for label, br in zip(scenario_cols, best_results):
        row[label] = f"{br['mech_fail_prob']*100:.1f}%"
    rows.append(row)

    actual_odi = actual_eval.get("postop_values", {}).get("ODI_postop") if include_actual else None
    row = {"Parameter": "Predicted ODI"}
    if include_actual:
        row[col_actual] = _f1(actual_odi) if actual_odi is not None else "N/A"
    for label, br in zip(scenario_cols, best_results):
        odi = br.get("postop_values", {}).get("ODI_postop")
        row[label] = _f1(odi) if odi is not None else "N/A"
    rows.append(row)

    gap_score_pre = patient_fixed.get("gap_score_preop", "-")
    gap_cat_pre = patient_fixed.get("gap_category", "-")
    a_gap = actual_eval["gap_info"] if include_actual else None
    row = {"Parameter": "GAP Score"}
    if include_actual:
        row[col_actual] = f"{gap_score_pre} ({gap_cat_pre}) → {a_gap['gap_score']} ({a_gap['gap_category']})"
    for label, br in zip(scenario_cols, best_results):
        g = br["gap_info"]
        row[label] = f"{gap_score_pre} ({gap_cat_pre}) → {g['gap_score']} ({g['gap_category']})"
    rows.append(row)

    # ── Separator ──
    rows.append({"Parameter": "─" * 12, **{c: "─" * 10 for c in all_cols}})

    # ── Surgical Plan ──
    rows.append({"Parameter": "SURGICAL PLAN", **{c: "" for c in all_cols}})
    for col in config.PLAN_COLS:
        plan_row = {"Parameter": _PLAN_COL_DISPLAY.get(col, col)}
        if include_actual:
            plan_row[col_actual] = actual_eval["plan"].get(col, "-")
        for label, br in zip(scenario_cols, best_results):
            plan_row[label] = br["plan"].get(col, "-")
        rows.append(plan_row)

    rows.append({"Parameter": "─" * 12, **{c: "─" * 10 for c in all_cols}})

    # ── Postop Values ──
    rows.append({"Parameter": "POSTOP VALUES", **{c: "" for c in all_cols}})
    for param in alignment_params:
        postop_key = f"{param}_postop"
        param_row = {"Parameter": param}
        if include_actual:
            av = actual_eval["postop_values"].get(postop_key)
            param_row[col_actual] = _f1(av) if av is not None else "-"
        for label, br in zip(scenario_cols, best_results):
            val = br["postop_values"].get(postop_key)
            param_row[label] = _f1(val) if val is not None else "-"
        rows.append(param_row)

    if pi_val is not None:
        rows.append({"Parameter": "PI", **{c: _f1(pi_val) for c in all_cols}})

    pt_preop = patient_fixed.get("PT_preop")
    if pt_preop is not None and pi_val is not None:
        pt_row = {"Parameter": "PT"}
        if include_actual:
            a_ss = actual_eval["postop_values"].get("SS_postop")
            pt_row[col_actual] = _f1(pi_val - a_ss) if a_ss is not None else "-"
        for label, br in zip(scenario_cols, best_results):
            ss = br["postop_values"].get("SS_postop")
            pt_row[label] = _f1(pi_val - ss) if ss is not None else "-"
        rows.append(pt_row)

    rows.append({"Parameter": "─" * 12, **{c: "─" * 10 for c in all_cols}})

    # ── Constraints ──
    rows.append({"Parameter": "CONSTRAINTS", **{c: "" for c in all_cols}})

    def _cval(result, key):
        return result["postop_values"].get(key)

    pairs = ([(col_actual, actual_eval)] if include_actual else []) + list(zip(scenario_cols, best_results))

    # PI-LL (0–10)
    cr = {"Parameter": "PI-LL (0–10)"}
    for c, br in pairs:
        ll = _cval(br, "LL_postop")
        if pi_val is not None and ll is not None:
            v = pi_val - ll
            cr[c] = f"{v:.1f} {'✓' if 0 <= v <= 10 else '⚠'}"
        else:
            cr[c] = "-"
    rows.append(cr)

    # L1PA−ideal (|d|≤3)
    if pi_val is not None:
        ideal_l1pa = calculate_ideal_l1pa(pi_val)
        cr = {"Parameter": "L1PA−ideal (|d|≤3)"}
        for c, br in pairs:
            l1pa = _cval(br, "L1PA_postop")
            if l1pa is not None:
                d = l1pa - ideal_l1pa
                cr[c] = f"{d:.1f} {'✓' if abs(d) <= 3 else '⚠'}"
            else:
                cr[c] = "-"
        rows.append(cr)

    # L4S1 (35–45)
    cr = {"Parameter": "L4S1 (35–45)"}
    for c, br in pairs:
        v = _cval(br, "L4S1_postop")
        if v is not None:
            cr[c] = f"{v:.1f} {'✓' if 35 <= v <= 45 else '⚠'}"
        else:
            cr[c] = "-"
    rows.append(cr)

    # T4PA−L1PA (|d|≤3)
    cr = {"Parameter": "T4PA−L1PA (|d|≤3)"}
    for c, br in pairs:
        t4 = _cval(br, "T4PA_postop")
        l1 = _cval(br, "L1PA_postop")
        if t4 is not None and l1 is not None:
            d = t4 - l1
            cr[c] = f"{d:.1f} {'✓' if abs(d) <= 3 else '⚠'}"
        else:
            cr[c] = "-"
    rows.append(cr)

    # LL−ideal (|d|≤3)
    if pi_val is not None:
        ideal_ll = calculate_ideal_ll(pi_val)
        cr = {"Parameter": "LL−ideal (|d|≤3)"}
        for c, br in pairs:
            ll = _cval(br, "LL_postop")
            if ll is not None:
                d = ll - ideal_ll
                cr[c] = f"{d:.1f} {'✓' if abs(d) <= 3 else '⚠'}"
            else:
                cr[c] = "-"
        rows.append(cr)

    # GAP Category (P)
    cr = {"Parameter": "GAP Category (P)"}
    for c, br in pairs:
        cat = br["gap_info"]["gap_category"]
        cr[c] = f"{cat} {'✓' if cat == 'P' else '⚠'}"
    rows.append(cr)

    df_combined = pd.DataFrame(rows)

    def _highlight_actual(col):
        if include_actual and col.name == col_actual:
            return ["background-color: rgba(255,255,255,0.08)"] * len(col)
        return [""] * len(col)

    styled = df_combined.style.apply(_highlight_actual, axis=0).hide(axis="index")
    ipy_display(styled)
    return df_combined
