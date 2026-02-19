import numpy as np
import pandas as pd
import src.optimization_utils as ou
from src import config


def get_diverse_solutions(res,
                          top_n=10,
                          top_per_gen=50,
                          score_tolerance=0.01,
                          bucket_cols=("UIV_implant",),
                          n_per_bucket=1,
                          patient_fixed=None,
                          delta_bundles=None,
                          mech_fail_bundle=None,
                          odi_bundle=None,
                          weights=None):
    """
    Extract diverse solutions from optimization history.
    
    Args:
        res: pymoo result object with history
        top_n: number of top solutions to return
        top_per_gen: how many top solutions to consider per generation
        score_tolerance: max difference from best score to be considered "good"
        bucket_cols: columns to use for diversity bucketing
        n_per_bucket: solutions to keep per bucket
        patient_fixed: patient preop data (optional, for full evaluation)
        delta_bundles: delta model bundles (optional, for full evaluation)
        mech_fail_bundle: mech fail model bundle (optional, for full evaluation)
        odi_bundle: ODI model bundle (optional, for full evaluation)
        weights: composite score weights (optional, for full evaluation)
        
    Returns:
        DataFrame with diverse top solutions (with postop values if bundles provided)
    """
    rows = []

    for gen, algo in enumerate(res.history):
        pop = algo.pop
        Xg = np.asarray(pop.get("X")).astype(int)
        Fg = pop.get("F").flatten()

        order = np.argsort(Fg)
        take = min(top_per_gen, len(order))

        for idx in order[:take]:
            plan = ou.decode_plan(Xg[idx])
            rows.append({**plan, "fitness": float(Fg[idx]), "gen": gen, "x": Xg[idx]})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Use plan columns from config
    plan_cols = config.PLAN_COLS

    df = (
        df.sort_values(["fitness", "gen"])
          .drop_duplicates(subset=plan_cols, keep="first")
          .reset_index(drop=True)
    )

    best = float(df["fitness"].min())
    df_good = df[df["fitness"] <= best + score_tolerance]

    if df_good.empty:
        df_good = df.copy()

    df_good = df_good.sort_values(["fitness", "gen"]).reset_index(drop=True)

    if bucket_cols:
        bucketed = (
            df_good.groupby(list(bucket_cols), as_index=False, sort=False)
                   .head(n_per_bucket)
        )
    else:
        bucketed = df_good.copy()

    bucketed = bucketed.sort_values(["fitness", "gen"]).reset_index(drop=True)
    selected = bucketed.head(top_n).copy()

    if len(selected) < top_n:
        used = set(tuple(r) for r in selected[plan_cols].to_numpy())

        fill_rows = []
        for _, r in df_good.iterrows():
            key = tuple(r[plan_cols].to_numpy())
            if key in used:
                continue
            fill_rows.append(r)
            used.add(key)
            if len(selected) + len(fill_rows) >= top_n:
                break

        if fill_rows:
            selected = pd.concat([selected, pd.DataFrame(fill_rows)], ignore_index=True)

    selected = selected.sort_values(["fitness", "gen"]).reset_index(drop=True)
    
    # If evaluation bundles provided, add postop values and mech fail probability
    if patient_fixed is not None and delta_bundles is not None and mech_fail_bundle is not None:
        eval_rows = []
        for _, row in selected.iterrows():
            x = row["x"]
            result = ou.evaluate_solution(x, patient_fixed, delta_bundles, mech_fail_bundle, weights, odi_bundle=odi_bundle)
            
            eval_row = {col: row[col] for col in plan_cols}
            eval_row["composite_score"] = round(result["composite_score"], 2)
            eval_row["mech_fail_prob"] = f"{result['mech_fail_prob'] * 100:.1f}%"
            eval_row["gap_score"] = result["gap_info"]["gap_score"]
            eval_row["gap_category"] = result["gap_info"]["gap_category"]
            
            # Add postop values
            for k, v in result["postop_values"].items():
                eval_row[k] = round(v, 1)
            
            # Add PI-LL (ideal range: 0 to 10) with status marker
            pi_val = patient_fixed.get("PI_preop")
            ll_postop = result["postop_values"].get("LL_postop")
            if pi_val is not None and ll_postop is not None:
                pi_ll_postop = pi_val - ll_postop
                status = "✓" if 0 <= pi_ll_postop <= 10 else "⚠"
                eval_row["PI-LL_postop"] = f"{round(pi_ll_postop, 1)} {status}"
            
            eval_rows.append(eval_row)
        
        return pd.DataFrame(eval_rows)
    
    # Remove the x column before returning if not doing full evaluation
    return selected.drop(columns=["x"]).reset_index(drop=True)