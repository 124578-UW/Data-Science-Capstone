import numpy as np
from pymoo.core.problem import ElementwiseProblem

from src import optimization_utils as ou


class SpineProblem(ElementwiseProblem):
    """
    Optimization problem that minimizes composite score for spine surgery planning.
    
    The composite score is calculated from predicted postoperative values using 
    the GAP (Global Alignment and Proportion) framework.
    
    Constraints:
        - If num_interbody_fusion_levels = 0, then ALIF = XLIF = TLIF = 0
        - num_interbody_fusion_levels >= ALIF + XLIF + TLIF
    
    Args:
        patient_fixed: dict with patient preoperative values
        delta_bundles: dict with delta model bundles (keys: L4S1, LL, T4PA, L1PA, SS, GlobalTilt)
        xl: array of lower bounds for decision variables
        xu: array of upper bounds for decision variables
        weights: dict with keys w1-w6, w_mech_fail, w_odi for composite score weights (optional)
        mech_fail_bundle: mechanical failure model bundle (optional, needed if w_mech_fail > 0)
        odi_bundle: ODI model bundle (optional, needed if w_odi > 0)
        pso_ll_override: if True, clamp predicted delta_LL to procedure-based correction range
    """
    
    def __init__(self, patient_fixed, delta_bundles, xl, xu, weights=None, mech_fail_bundle=None, odi_bundle=None, pso_ll_override=False):
        super().__init__(
            n_var=len(xl),
            n_obj=1,
            n_ieq_constr=2,
            xl=xl,
            xu=xu,
            vtype=int,
        )
        self.patient_fixed = patient_fixed
        self.delta_bundles = delta_bundles
        self.weights = weights or {}
        self.mech_fail_bundle = mech_fail_bundle
        self.odi_bundle = odi_bundle
        self.pso_ll_override = pso_ll_override

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
        
        # Constraint 1: if num_interbody == 0, then ALIF + XLIF + TLIF must == 0
        # g <= 0 is feasible; g > 0 is infeasible
        # When num_interbody=0: g1 = (alif+xlif+tlif) * 1 → infeasible if any selected
        # When num_interbody>0: g1 = (alif+xlif+tlif) * 0 = 0 → always feasible
        g1 = (alif + xlif + tlif) * (1 - min(num_interbody, 1))
        
        # Constraint 2: num_interbody must be >= sum of fusion types selected
        # e.g. ALIF=1,XLIF=1 → need num_interbody >= 2
        g2 = (alif + xlif + tlif) - num_interbody
        
        # Fitness: composite score (lower = better)
        f = ou.fitness_composite_score(
            xi,
            self.patient_fixed,
            self.delta_bundles,
            mech_fail_bundle=self.mech_fail_bundle,
            odi_bundle=self.odi_bundle,
            weights=self.weights,
            pso_ll_override=self.pso_ll_override,
        )
        
        out["F"] = [f]
        out["G"] = [g1, g2]
