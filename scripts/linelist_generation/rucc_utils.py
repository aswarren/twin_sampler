#!/usr/bin/env python3
"""
rucc_utils.py
Utilities for reading and reshaping USDA RUCC lookup data.
"""

from __future__ import annotations
import pandas as pd


def load_and_pivot_rucc(rucc_path: str, encoding: str = "latin1") -> pd.DataFrame:
    """
    Load a 'long' RUCC CSV and pivot to wide, returning columns:
        ['FIPS', 'State', 'County_Name', 'rucc_code', ...]
    Expects columns in the input: ['FIPS', 'State', 'County_Name', 'Attribute', 'Value'].
    """
    rucc_long = pd.read_csv(rucc_path, encoding=encoding)
    required_cols = {"FIPS", "State", "County_Name", "Attribute", "Value"}
    missing = required_cols - set(rucc_long.columns)
    if missing:
        raise ValueError(f"RUCC file missing columns: {missing}")

    rucc_wide = (
        rucc_long.pivot_table(
            index=["FIPS", "State", "County_Name"],
            columns="Attribute",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # Normalize types and names
    if "RUCC_2023" in rucc_wide.columns:
        rucc_wide = rucc_wide.rename(columns={"RUCC_2023": "rucc_code"})
    elif "RUCC" in rucc_wide.columns:
        rucc_wide = rucc_wide.rename(columns={"RUCC": "rucc_code"})
    else:
        # If neither exists, keep going but leave rucc_code missing
        rucc_wide["rucc_code"] = pd.NA

    rucc_wide["rucc_code"] = pd.to_numeric(rucc_wide["rucc_code"], errors="coerce")

    # Ensure FIPS is zero-padded 5-char string
    rucc_wide["FIPS"] = rucc_wide["FIPS"].astype(str).str.zfill(5)

    return rucc_wide
