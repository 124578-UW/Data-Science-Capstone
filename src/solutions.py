import numpy as np
import pandas as pd
import src.optimization_utils as ou


def get_diverse_solutions(res,
                          top_n=10,
                          top_per_gen=50,
                          eps=0.01,
                          bucket_cols=("UIV_implant"),
                          n_per_bucket=1):

    rows = []
    UIV_CHOICES, _, _ = ou.get_decision_config()

    for gen, algo in enumerate(res.history):
        pop = algo.pop
        Xg = np.asarray(pop.get("X")).astype(int)
        Fg = pop.get("F").flatten()

        order = np.argsort(Fg)
        take = min(top_per_gen, len(order))

        for idx in order[:take]:
            plan = ou.decode_plan(Xg[idx], UIV_CHOICES)
            rows.append({**plan, "fitness": float(Fg[idx]), "gen": gen})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    plan_cols = [
        "UIV_implant",
        "num_interbody_fusion_levels",
        "ALIF",
        "XLIF",
        "TLIF",
        "num_rods",
        "num_pelvic_screws",
        "osteotomy",
    ]

    df = (
        df.sort_values(["fitness", "gen"])
          .drop_duplicates(subset=plan_cols, keep="first")
          .reset_index(drop=True)
    )

    best = float(df["fitness"].min())
    df_good = df[df["fitness"] <= best + eps]

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

    return selected.sort_values(["fitness", "gen"]).reset_index(drop=True)