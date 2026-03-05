import numpy as np
from pymoo.core.problem import ElementwiseProblem

from src import optimization_utils as ou


class SpineProblem(ElementwiseProblem):
    """
    Optimization problem that minimizes composite score for spine surgery planning.
    
    The composite score is calculated from predicted postoperative values using 
    the GAP (Global Alignment and Proportion) framework.
    
    Constraints:
        - If num_interbody_fusion_levels > 0, at least one of ALIF, XLIF, or TLIF must be 1
        - If any fusion type is selected (ALIF/XLIF/TLIF), num_interbody_fusion_levels must be > 0
        - If ALIF=1 and XLIF=0 and TLIF=0, num_interbody_fusion_levels must be < 4
    
    Args:
        patient_fixed: dict with patient preoperative values
        delta_bundles: dict with delta model bundles (keys: L4S1, LL, T4PA, L1PA, SS, GlobalTilt)
        xl: array of lower bounds for decision variables
        xu: array of upper bounds for decision variables
        weights: dict with keys w1-w6, w_mech_fail, w_odi for composite score weights (optional)
        mech_fail_bundle: mechanical failure model bundle (optional, needed if w_mech_fail > 0)
        odi_bundle: ODI model bundle (optional, needed if w_odi > 0)
    """
    
    def __init__(self, patient_fixed, delta_bundles, xl, xu, weights=None, mech_fail_bundle=None, odi_bundle=None):
        super().__init__(
            n_var=len(xl),
            n_obj=1,
            n_ieq_constr=3,
            xl=xl,
            xu=xu,
            vtype=int,
        )
        self.patient_fixed = patient_fixed
        self.delta_bundles = delta_bundles
        self.weights = weights or {}
        self.mech_fail_bundle = mech_fail_bundle
        self.odi_bundle = odi_bundle

    def _evaluate(self, x, out, *args, **kwargs):
        # Round to integers first — GA operates on floats internally,
        # but constraints must be checked on the actual decoded plan.
        xi = np.rint(x).astype(int)

        # Decision variable indices (from config.DECISION_VAR_SPECS):
        # 2 = num_interbody_fusion_levels, 3 = ALIF, 4 = XLIF, 5 = TLIF
        num_interbody = xi[2]
        alif = xi[3]
        xlif = xi[4]
        tlif = xi[5]
        
        # Constraint 1: if num_interbody > 0, need ALIF + XLIF + TLIF >= 1
        # g <= 0 is feasible; g > 0 is infeasible
        g1 = num_interbody * (1 - (alif + xlif + tlif))
        
        # Constraint 2: if any fusion type selected, num_interbody must be >= 1
        # max(alif,xlif,tlif)=1 and num_interbody=0 → 1-0=1 > 0 → infeasible
        g2 = max(alif, xlif, tlif) - num_interbody
        
        # Constraint 3: if ALIF-only (no XLIF/TLIF), num_interbody must be < 4
        g3 = alif * (1 - xlif) * (1 - tlif) * (num_interbody - 3)
        
        # Fitness: composite score (lower = better)
        f = ou.fitness_composite_score(
            xi,
            self.patient_fixed,
            self.delta_bundles,
            mech_fail_bundle=self.mech_fail_bundle,
            odi_bundle=self.odi_bundle,
            weights=self.weights
        )
        
        out["F"] = [f]
        out["G"] = [g1, g2, g3]
