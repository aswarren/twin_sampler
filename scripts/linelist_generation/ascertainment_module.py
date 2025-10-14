import pandas as pd
import yaml
from typing import Dict, Any

def load_ascertainment_parameters(param_filepath: str) -> Dict[str, Any]:
    """
    Loads ascertainment model parameters from a specified YAML file.
    (This function remains the same)
    """
    with open(param_filepath, 'r') as file:
        return yaml.safe_load(file)

def _map_abm_state_to_model_inputs(exit_state: str) -> tuple[str | None, str]:
    """
    Parses an agent's exit_state to determine symptom severity and age group.
    (This function remains the same)
    """
    if pd.isna(exit_state):
        return None, '18-49' # Default for missing data

    parts = str(exit_state).split('_')
    state_prefix = parts[0]
    age_code = parts[1] if len(parts) > 1 else 'a'

    symptom_severity = None
    if state_prefix.startswith(('A', 'P')): # Group A and P together
        symptom_severity = 'asymptomatic'
    elif state_prefix.startswith('I'):
        symptom_severity = 'mild'
    elif state_prefix.startswith(('H', 'hM', 'Vent', 'D', 'dM', 'dVent', 'dH')): # Added 'dm' for completeness
        symptom_severity = 'severe'

    age_map = {'p': '0-17', 'a': '18-49', 'o': '50-64', 's': '65+'}
    age_group = age_map.get(age_code, '18-49')

    return symptom_severity, age_group

def preprocess_for_ascertainment(df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW: Prepares the merged DataFrame by adding columns required for the ascertainment model.
    This is the data transformation step.
    """
    if 'exit_state' not in df.columns:
        raise ValueError("DataFrame must contain 'exit_state' column for preprocessing.")

    # 1. Map ABM state to symptom severity and model-compatible age group
    mapped_data = df['exit_state'].apply(_map_abm_state_to_model_inputs)
    df[['symptom_severity', 'age_group_model']] = pd.DataFrame(mapped_data.tolist(), index=df.index)

    if 'hh_income' in df.columns:
        def map_income_to_ses(income):
            if pd.isna(income):
                return 'medium' # Default for missing income
            try:
                income = float(income)
                # These thresholds can be tuned, but let's start with something reasonable.
                if income < 50000:
                    return 'low'
                elif income > 125000:
                    return 'high'
                else:
                    return 'medium'
            except (ValueError, TypeError):
                return 'medium' # Default for non-numeric income
        
        df['ses_category'] = df['hh_income'].apply(map_income_to_ses)
        print("INFO: Created 'ses_category' from 'hh_income' column.")
    else:
        # Fallback if hh_income is not even available
        df['ses_category'] = 'medium'
    if 'rucc_code' in df.columns:
        def map_rucc_to_location(rucc):
            if pd.isna(rucc):
                return 'urban' # Default for missing RUCC
            try:
                rucc = float(rucc)
                # Using the old model's 3-tier system
                if rucc <= 3:
                    return 'urban' # Metro counties
                # A more granular option could be to add a 'suburban' category
                elif rucc <= 6:
                     return 'suburban'
                else:
                    return 'rural' # Micropolitan and noncore counties
            except (ValueError, TypeError):
                return 'urban' # Default for non-numeric RUCC

        df['location_type'] = df['rucc_code'].apply(map_rucc_to_location)
        print("INFO: Created 'location_type' from 'rucc_code' column.")
    else:
        # Fallback if rucc_code is not available
        df['location_type'] = 'urban'
    if 'comorbidity_category' not in df.columns:
        if 'has_comorbidities' in df.columns:
            # If the simple boolean exists, map it to the YAML's default categories
            # Assuming 'medium_impact' for presence, and 'no_or_low_impact' for absence.
            # This could be refined if more detail is available.
            df['comorbidity_category'] = df['has_comorbidities'].apply(
                lambda x: 'medium_impact' if x else 'no_or_low_impact'
            )
        else:
            # If no comorbidity info exists at all, default everyone to the baseline.
            df['comorbidity_category'] = 'no_or_low_impact'

    return df.dropna(subset=['symptom_severity'])



def compute_ascertainment_probability(row: pd.Series, params: Dict[str, Any]) -> float:
    """
    NEW: Row-wise probability model. This is the direct replacement for
    testing_prob.compute_testing_probability().
    """
    try:
        # --- ALL LOOKUPS BELOW ARE NOW CORRECTED ---

        # 1. Base Probability (This was already correct)
        severity_key = row['symptom_severity']
        base_prob = params['base_probabilities'][severity_key]['probability']
        
        # 2. Age Multiplier
        age_group_key = row['age_group_model']
        # FIX: Look up the age_group_key directly and get the 'multiplier'
        age_multiplier = params['modifiers']['age'][age_group_key]['multiplier']

        # 3. SES Multiplier
        ses_key = row['ses_category']
        # FIX: Added ['multiplier'] to get the numeric value
        ses_multiplier = params['modifiers']['socioeconomic_status'][ses_key]['multiplier']
        
        # 4. Geography Multiplier
        geo_key = row['location_type']
        # FIX: Added ['multiplier'] to get the numeric value
        geo_multiplier = params['modifiers']['geography'][geo_key]['multiplier']
        
        # 5. Comorbidity Multiplier
        comorbidity_key = row['comorbidity_category'] # Using the new robust column
        # FIX: Use the new key and add ['multiplier']
        comorbidity_multiplier = params['modifiers']['comorbidity'][comorbidity_key]['multiplier']

        # Final calculation
        final_prob = (base_prob * age_multiplier * ses_multiplier *
                      geo_multiplier * comorbidity_multiplier)
                      
        return min(final_prob, 1.0)

    except KeyError as e:
        print(f"Warning: Missing parameter or mapping for key {e}. Defaulting to 0.0 probability for this row.")
        return 0.0