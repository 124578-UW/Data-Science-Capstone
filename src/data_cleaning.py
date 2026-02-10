### This module provides functions for reading and cleaning patient surgical data 

import pandas as pd

def read_data(data_path):
    "load patient surgical data"
    df = pd.read_excel(data_path)
    return df

def clean_data(df):
    "clean patient surgical data"

    #remove extra space or line breaks in column names
    df.columns = df.columns.astype(str).str.replace("\n"," ").str.replace(r"\s+"," ", regex=True).str.strip()

    #convert data type to integer for calculation of composite score
    df[["gap_score_preop", "gap_score_postop"]] = df[["gap_score_preop", "gap_score_postop"]]\
    .apply(pd.to_numeric, errors="coerce").astype("Int64")

    #unify spelling
    df["UIV_implant"] = df["UIV_implant"].str.capitalize().replace({"Fs": "FS", "Ps": "PS"})
    df.loc[df["UIV_implant"].str.contains("Fenestrated", na=False), "UIV_implant"] = "FS"
    df.loc[df["UIV_implant"].str.contains("ether", na=False), "UIV_implant"] = "PS"
    df["sex"]= df["sex"].replace({"F": "FEMALE", "M": "MALE"})

    #create column with updated num_levels
    df["num_levels"] = pd.to_numeric(df["num_levels"], errors="coerce").astype("Int64")
    df.loc[df["num_levels"] >= 10, "updated_num_levels"] = "higher"
    df.loc[df["num_levels"] < 10, "updated_num_levels"] = "lower"

    return df