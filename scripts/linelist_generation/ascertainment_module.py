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
    if state_prefix.startswith('A'):
        symptom_severity = 'asymptomatic'
    elif state_prefix.startswith(('P', 'I')):
        symptom_severity = 'mild'
    elif state_prefix.startswith(('H', 'hM', 'Vent', 'd')):
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

    # 2. Add placeholder columns if they don't exist in the source data.
    #    This makes the model more robust.
    if 'ses_category' not in df.columns:
        df['ses_category'] = 'middle' # Assume middle SES if not specified
    if 'location_type' not in df.columns:
        df['location_type'] = 'urban' # Assume urban if not specified
    if 'has_comorbidities' not in df.columns:
        df['has_comorbidities'] = False # Assume no comorbidities if not specified

    return df.dropna(subset=['symptom_severity'])


def compute_ascertainment_probability(row: pd.Series, params: Dict[str, Any]) -> float:
    """
    NEW: Row-wise probability model. This is the direct replacement for
    testing_prob.compute_testing_probability().
    """
    try:
            # Get the severity string (e.g., 'mild', 'severe') from the row
            severity_key = row['symptom_severity']
            
            # Look up the severity_key directly in base_probabilities
            # AND get the final 'probability' value from the nested dictionary.
            base_prob = params['base_probabilities'][severity_key]['probability']
            #base_prob = params['base_probabilities']['symptom_severity'][row['symptom_severity']]
            age_multiplier = params['modifiers']['age']['age_group_model'][row['age_group_model']]
            ses_multiplier = params['modifiers']['socioeconomic_status'][row['ses_category']]
            geo_multiplier = params['modifiers']['geography'][row['location_type']]
            
            comorbidity_status = 'present' if row['has_comorbidities'] else 'absent'
            comorbidity_multiplier = params['modifiers']['comorbidity'][comorbidity_status]

            final_prob = (base_prob * age_multiplier * ses_multiplier *
                        geo_multiplier * comorbidity_multiplier)
            return min(final_prob, 1.0)

    except KeyError as e:
        # If a category is missing from the YAML, default to a neutral (1.0) or zero probability
        # to avoid crashing the whole simulation.
        print(f"Warning: Missing parameter or mapping for key {e}. Defaulting to 0.0 probability for this row.")
        return 0.0