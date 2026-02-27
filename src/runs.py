import numpy as np

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

import src.optimization_utils as ou
import src.solutions as s
from src.problem import SpineProblem


# =============================================================================
# Weight Presets
# =============================================================================

PRESETS = {
    "equal": {
        "label": "Equal Weights (Composite)",
        "description": "All 6 alignment components and mechanical failure weighted equally",
        "weights": {"w1": 1/7, "w2": 1/7, "w3": 1/7, "w4": 1/7, "w5": 1/7, "w6": 1/7, "w_mech_fail": 1/7},
    },
    "mech_fail": {
        "label": "Minimize Mechanical Failure",
        "description": "Primarily mechanical failure probability, with small alignment weights",
        "weights": {"w1": 0.05, "w2": 0.05, "w3": 0.05, "w4": 0.05, "w5": 0.05, "w6": 0.05, "w_mech_fail": 0.7},
    },
    "mech_fail_t4l1pa": {
        "label": "Minimize Mechanical Failure + T4L1PA",
        "description": "Mechanical failure probability and T4PA-L1PA mismatch weighted equally",
        "weights": {"w1": 0, "w2": 0, "w3": 0, "w4": 0.5, "w5": 0, "w6": 0, "w_mech_fail": 0.5},
    },
    "l4s1": {
        "label": "Minimize L4S1 Penalty",
        "description": "Primarily L4-S1 in ideal range (35-45°), with mech failure guard",
        "weights": {"w1": 0, "w2": 0, "w3": 0.8, "w4": 0, "w5": 0, "w6": 0, "w_mech_fail": 0.2},
    },
    "t4l1pa": {
        "label": "Minimize T4L1PA Penalty",
        "description": "Primarily T4PA-L1PA mismatch, with mech failure guard",
        "weights": {"w1": 0, "w2": 0, "w3": 0, "w4": 0.8, "w5": 0, "w6": 0, "w_mech_fail": 0.2},
    },
    "equal_plus_mech": {
        "label": "Equal Alignment + Mechanical Failure",
        "description": "All 6 alignment components weighted equally, blended 50/50 with mechanical failure",
        "weights": {"w1": 1/12, "w2": 1/12, "w3": 1/12, "w4": 1/12, "w5": 1/12, "w6": 1/12, "w_mech_fail": 0.5},
    },
    "odi": {
        "label": "Minimize Postop ODI",
        "description": "Primarily lowest predicted postoperative ODI, with mech failure guard",
        "weights": {"w1": 0, "w2": 0, "w3": 0, "w4": 0, "w5": 0, "w6": 0, "w_mech_fail": 0.2, "w_odi": 0.8},
    },
    "gap_score": {
        "label": "Minimize GAP Score",
        "description": "Primarily overall GAP alignment score and category improvement, with mech failure guard",
        "weights": {"w1": 0.5, "w2": 0, "w3": 0, "w4": 0, "w5": 0, "w6": 0.3, "w_mech_fail": 0.2},
    },
    "ll": {
        "label": "Minimize LL (PI-LL) Penalty",
        "description": "Primarily PI-LL mismatch, with mech failure guard",
        "weights": {"w1": 0, "w2": 0, "w3": 0, "w4": 0, "w5": 0.8, "w6": 0, "w_mech_fail": 0.2},
    },
}

WEIGHT_LABELS = {
    "w1": "GAP Score",
    "w2": "L1PA penalty",
    "w3": "L4S1 penalty",
    "w4": "T4L1PA penalty",
    "w5": "LL penalty",
    "w6": "GAP category improvement",
    "w_mech_fail": "Mechanical failure prob",
    "w_odi": "ODI postop",
}


def print_preset(preset):
    """Print a preset's weights in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"  {preset['label']}")
    print(f"  {preset['description']}")
    print(f"{'=' * 60}")
    for k, v in preset["weights"].items():
        marker = " ◀" if v > 0 else ""
        print(f"  {WEIGHT_LABELS.get(k, k):>30}: {v:.4f}{marker}")


# =============================================================================
# Run Optimization
# =============================================================================

def run_optimization(
    preset,
    patient_fixed,
    delta_bundles,
    mech_fail_bundle,
    xl,
    xu,
    odi_bundle=None,
    pop_size=100,
    n_gen=20,
    seed=42,
    verbose=False,
    top_n=12,
    score_tolerance=2,
):
    """
    Run a single GA optimization with the given preset.

    Args:
        preset: dict from PRESETS (must have 'weights' and 'label')
        patient_fixed: patient preop data dict
        delta_bundles: dict of delta model bundles
        mech_fail_bundle: mechanical failure model bundle
        xl, xu: decision variable bounds
        pop_size: GA population size
        n_gen: number of generations
        seed: random seed
        verbose: print GA progress
        top_n: number of diverse solutions to extract
        score_tolerance: score window around best for diverse solutions

    Returns:
        dict with keys:
            - label: preset label
            - weights: weight dict used
            - res: raw pymoo result
            - best_x: best decision vector
            - best_result: full evaluation dict of best solution
            - diverse_df: DataFrame of top diverse solutions
    """
    weights = preset["weights"]

    # Only pass bundles to optimizer if their weights are active
    mf_bundle = mech_fail_bundle if weights.get("w_mech_fail", 0) > 0 else None
    odi_bun_optim = odi_bundle if weights.get("w_odi", 0) > 0 else None

    problem = SpineProblem(
        patient_fixed=patient_fixed,
        delta_bundles=delta_bundles,
        xl=xl,
        xu=xu,
        weights=weights,
        mech_fail_bundle=mf_bundle,
        odi_bundle=odi_bun_optim,
    )

    algorithm = GA(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
        save_history=True,
    )

    best_x = np.asarray(res.X).astype(int)
    best_result = ou.evaluate_solution(
        best_x, patient_fixed, delta_bundles, mech_fail_bundle, weights=weights, odi_bundle=odi_bundle
    )

    diverse_df = s.get_diverse_solutions(
        res=res,
        top_n=top_n,
        top_per_gen=50,
        score_tolerance=score_tolerance,
        bucket_cols=("UIV_implant", "ALIF", "XLIF", "TLIF"),
        n_per_bucket=1,
        patient_fixed=patient_fixed,
        delta_bundles=delta_bundles,
        mech_fail_bundle=mech_fail_bundle,
        odi_bundle=odi_bundle,
        weights=weights,
    )

    return {
        "label": preset["label"],
        "weights": weights,
        "res": res,
        "best_x": best_x,
        "best_result": best_result,
        "diverse_df": diverse_df,
    }
