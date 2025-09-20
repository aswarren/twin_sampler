#!/usr/bin/env python3
"""
testing_prob.py
Contains compute_testing_probability() with robust handling for missing data,
Unicode dashes in age groups, and occupation SOC grouping.
"""

from __future__ import annotations
import math
import pandas as pd


def _norm_age_group(val: object) -> str:
    """
    Normalize age group strings by converting Unicode dashes to ASCII hyphen
    and trimming spaces. Returns the normalized string or empty string if NA.
    """
    if pd.isna(val):
        return ""
    s = str(val)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-").strip()
    return s


def _safe_int(val: object, default: int = 0) -> int:
    """Coerce to int with a default for missing/non-numeric."""
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        try:
            # handle floats that are strings, etc.
            return int(float(val))
        except Exception:
            return default


def compute_testing_probability(row: pd.Series) -> float:
    """
    Row-wise probability model. Mirrors the user's original logic with
    defensive checks and consistent age-group normalization.
    """
    p = 0.5  # baseline

    # Income-based bias
    hh_income = row.get("hh_income")
    if pd.notnull(hh_income):
        try:
            income = float(hh_income)
            if income < 40000:
                p *= 0.6
            elif income < 70000:
                p *= 0.8
            elif income < 110000:
                p *= 1.0
            else:
                p *= 1.2
        except Exception:
            # If non-numeric, ignore income effect
            pass

    # Age group
    age_group = _norm_age_group(row.get("age_group"))
    if age_group == "Preschool (0-4)":
        p *= 0.5
    elif age_group == "Student (5-17)":
        p *= 0.7
    elif age_group == "Adult (18-49)":
        p *= 1.0
    elif age_group == "Older adult (50-64)":
        p *= 1.2
    elif age_group == "Senior (65+)":
        p *= 1.4
    else:
        # unknown → no age multiplier
        pass

    # Race group
    race = row.get("smh_race")
    if race == "White":
        p *= 1.0
    elif race == "Asian":
        p *= 0.85
    elif race == "Black":
        p *= 0.8
    elif race == "Latino":
        p *= 0.75
    elif race == "Other":
        p *= 0.7
    else:
        # unknown → no race multiplier
        pass

    # Occupation (SOC major group) — only for working age groups
    soc = row.get("occupation_socp")
    soc_code = "" if pd.isna(soc) else str(soc)
    major_group = soc_code[:2] if soc_code else ""

    # Note: we already normalized age_group to ASCII hyphen above.
    if age_group in {"Adult (18-49)", "Older adult (50-64)"}:
        if major_group == "29":        # Healthcare
            p *= 1.4
        elif major_group == "35":      # Food service
            p *= 1.1
        elif major_group == "25":      # Education
            p *= 1.2
        elif major_group in {"11", "13"}:  # Management, business
            p *= 1.0
        elif major_group in {"41", "53"}:  # Sales, transport
            p *= 0.9
        elif major_group == "51":      # Manufacturing / production
            p *= 0.85
        else:
            p *= 0.7  # by default assume lower propensity / not employed

    # Vehicles available (proxy for mobility/access)
    v = _safe_int(row.get("vehicles"), default=0)
    if v == 0:
        p *= 0.7
    elif v == 1:
        p *= 0.9
    else:  # 2+
        p *= 1.0

    # RUCC (metro vs suburban vs rural)
    rucc = row.get("rucc_code")
    try:
        if pd.isna(rucc):
            pass  # unknown → no multiplier
        else:
            rucc_val = float(rucc)
            if rucc_val <= 3:
                p *= 1.0   # metro
            elif rucc_val <= 6:
                p *= 0.85  # suburban/small city
            else:
                p *= 0.7   # rural
    except Exception:
        pass

    # Symptoms: asymptomatic people are less likely to test
    asymp = row.get("asymptomatic")
    if asymp is True:
        p *= 0.0
    elif asymp is False:
        p *= 1.0
    else:
        # unknown → assume no additional multiplier
        pass

    # clip to [0,1]
    return max(0.0, min(1.0, float(p)))
