import pandas as pd
import yaml
import random
import io
from typing import Dict, Any, List

def load_ascertainment_parameters(param_filepath: str) -> Dict[str, Any]:
    """
    Loads ascertainment model parameters from a specified YAML file.
    """
    with open(param_filepath, 'r') as file:
        return yaml.safe_load(file)

def _map_abm_state_to_model_inputs(exit_state: str) -> tuple[str | None, str]:
    """
    Parses an agent's exit_state from the ABM to determine symptom severity
    and age group for the ascertainment model.
    """
    parts = exit_state.split('_')
    state_prefix = parts
    age_code = parts[1] if len(parts) > 1 else 'a' # Default to 'a' if no suffix

    # Map state prefix to symptom severity
    symptom_severity = None
    if state_prefix.startswith('A'):
        symptom_severity = 'asymptomatic'
    elif state_prefix.startswith(('P', 'I')):
        symptom_severity = 'mild'
    elif state_prefix.startswith(('H', 'hM', 'Vent', 'd')):
        symptom_severity = 'severe'

    # Map age code to age group category from YAML
    age_map = {'p': '0-17', 'a': '18-49', 'o': '50-64', 's': '65+'}
    age_group = age_map.get(age_code, '18-49') # Default to reference group

    return symptom_severity, age_group

def calculate_ascertainment_prob(agent: pd.Series, params: Dict[str, Any]) -> float:
    """
    Calculates the final ascertainment probability for a single agent
    using the multiplicative model.
    """
    try:
        base_prob = params['base_probability']['symptom_severity'][agent['symptom_severity']]
        
        age_multiplier = params['multipliers']['age_group'][agent['age_group']]
        ses_multiplier = params['multipliers']['socioeconomic_status'][agent['ses_category']]
        geo_multiplier = params['multipliers']['geography'][agent['location_type']]
        
        comorbidity_status = 'present' if agent['has_comorbidities'] else 'absent'
        comorbidity_multiplier = params['multipliers']['comorbidities'][comorbidity_status]

        final_prob = (base_prob * age_multiplier * ses_multiplier *
                      geo_multiplier * comorbidity_multiplier)

        return min(final_prob, 1.0)

    except KeyError as e:
        print(f"Error: Missing parameter or agent attribute mapping for key {e}.")
        print(f"Agent data: {agent.to_dict()}")
        return 0.0

def generate_line_list(
    abm_events_path: str,
    population_details_path: str,
    param_filepath: str,
    current_tick: int
) -> pd.DataFrame:
    """
    Generates a line list by processing raw ABM output for a specific tick.
    """
    # 1. Load data and parameters
    parameters = load_ascertainment_parameters(param_filepath)
    abm_events = pd.read_csv(abm_events_path)
    population_details = pd.read_csv(population_details_path)

    # 2. Filter for the current simulation time step
    current_events = abm_events[abm_events['tick'] == current_tick].copy()
    if current_events.empty:
        return pd.DataFrame()

    # 3. Map ABM states to model inputs (symptom severity and age group)
    mapped_data = current_events['exit_state'].apply(_map_abm_state_to_model_inputs)
    current_events[['symptom_severity', 'age_group']] = pd.DataFrame(mapped_data.tolist(), index=current_events.index)

    # 4. Filter for only infectious agents who are eligible for ascertainment
    infectious_agents = current_events.dropna(subset=['symptom_severity']).copy()
    if infectious_agents.empty:
        return pd.DataFrame()

    # 5. Merge with population data to get full agent attributes
    # Use a left merge to keep only the infectious agents from the current tick
    full_agent_details = pd.merge(infectious_agents, population_details, on='pid', how='left')
    
    # Handle cases where an agent in events is not in the population file
    if full_agent_details.isnull().any().any():
        print("Warning: Some agents in the event log are missing from the population details file.")
        full_agent_details.dropna(inplace=True)


    # 6. Calculate ascertainment probability for each agent
    full_agent_details['ascertainment_prob'] = full_agent_details.apply(
        lambda agent: calculate_ascertainment_prob(agent, parameters), axis=1
    )

    # 7. Perform Bernoulli trial for each agent
    ascertained_mask = full_agent_details['ascertainment_prob'] > [random.random() for _ in range(len(full_agent_details))]
    line_list = full_agent_details[ascertained_mask]

    return line_list.drop(columns=['ascertainment_prob'])

# --- Example Usage ---
if __name__ == '__main__':
    # Simulate reading the ABM output file provided in the prompt
    abm_output_data = """tick,pid,exit_state,contact_pid,location_id
-1,361192643,S_a,-1,-1
699,111489894,E2_a,111489896,1035111803
699,111496044,I2_o,-1,-1
699,111448533,R2_o,-1,-1
699,111486264,P2_o,-1,-1
699,111443361,E2_g,111443362,1032876644
699,111373215,R2_s,-1,-1
699,111383652,P2_s,-1,-1
699,111359961,hM2_a,-1,-1
699,111412931,P2_g,-1,-1
699,111422701,I2_s,-1,-1
699,111462432,P2_s,-1,-1
699,111464089,A2_a,-1,-1
699,111440603,A2_o,-1,-1
"""
    
    # Simulate a population details file with other required attributes
    population_data = {
        'pid': [111489894, 111496044, 111448533, 111486264, 111443361, 111373215, 111383652, 111359961, 111412931],
        'ses_category': ['low', 'middle', 'high', 'middle', 'low', 'high', 'middle', 'low', 'middle'],
        'location_type': ['rural', 'urban', 'urban', 'urban', 'rural', 'urban', 'rural', 'rural', 'urban'],
        'has_comorbidities': [True, False, True, False, True, False, True, False, True]
    }
    
    # Create in-memory CSV files for the function to read
    abm_events_filepath = io.StringIO(abm_output_data)
    population_details_filepath = io.StringIO(pd.DataFrame(population_data).to_csv(index=False))
    
    # Define the path to the parameters file
    params_file = 'ascertainment_parameters.yaml'

    # Generate the line list for the specified tick
    final_line_list = generate_line_list(
        abm_events_path=abm_events_filepath,
        population_details_path=population_details_filepath,
        param_filepath=params_file,
        current_tick=699
    )

    print("--- Generated Line List of Ascertained Cases (Tick 699) ---")
    if final_line_list.empty:
        print("No cases were ascertained in this run.")
    else:
        print(final_line_list)
    
    total_infectious = len(population_data['pid'])
    print(f"\nTotal infectious agents at tick 699: {total_infectious}. Total ascertained: {len(final_line_list)}.")