import argparse
import pandas as pd
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import minmax_scale

# --- Graph and Component Functions (from original script) ---
def create_graph_classic(infection_events_df):
    # Ensure PIDs are converted to float IDs using the tick value
    if not infection_events_df.empty:
        #as int
        infection_events_df['pid']=infection_events_df['pid'].astype(str)
        infection_events_df['contact_pid']=infection_events_df['contact_pid'].astype(str)
        #add alias column for pid as f"{pid}.{tick}" cast the resulting string to float
        infection_events_df['alias_pid'] = (infection_events_df['pid'] + '.' + infection_events_df['tick'].astype(str))
    #get dictinoary from pid and alias columns
    #alias_dict = infection_events_df.set_index('pid')['alias'].to_dict()
    #alias_dict = dict(zip(infection_events_df['pid'], infection_events_df['alias']))
    # Initialize an empty dictionary to keep track of the current alias for each target string (pid)
    alias_map = {}

    # Prepare a list to collect the alias_contact values for each row.
    alias_contact = []

    # Iterate over the DataFrame rows in order.
    # Use itertuples for faster iteration.
    for row in infection_events_df.itertuples(index=False):
        # Look up the current alias for the contact_pid.
        # If the target hasn't been seen yet, you can return np.nan or some default value.
        current_alias = alias_map.get(row.contact_pid, "-1")
        alias_contact.append(current_alias)

        # Now update the mapping for this row's pid using the alias_pid value.
        # Because the first occurrence for any pid is always in the pid column, this sets or updates the alias.
        alias_map[row.pid] = row.alias_pid

    # Assign the list as a new column in the DataFrame.
    infection_events_df['alias_contact'] = alias_contact

    # --- 1. Analyze Full Graph ---
    G_full = nx.DiGraph()
    #G_full = nx.MultiDiGraph()
    if not infection_events_df.empty:
        #only rows with contact_pid != -1
        # Filter the DataFrame to include only rows where contact_pid is not -1
        no_imports=infection_events_df[infection_events_df['contact_pid'] != "-1"]
        # OPTIMIZATION: Use add_edges_from for much faster graph construction
        edges = zip(no_imports['alias_contact'], no_imports['alias_pid'], no_imports['tick'])
        G_full.add_edges_from((u, v, {'tick': tick}) for u, v, tick in edges)

        G_full.add_edges_from(edges)
    return G_full 


def create_component_table(infection_df):
    G_full = create_graph_classic(infection_df)
    full_cascade_components = list(nx.weakly_connected_components(G_full))
    #create table with pid, tick, component_id
    component_data = []
    for i, component in enumerate(full_cascade_components):
        for node in component:
            component_data.append({'alias_pid': node, 'component_id': i})
    component_df = pd.DataFrame(component_data)
    return component_df

# --- New Variant Labeling Logic ---

def mode1_temporal_match(sim_components, real_imports):
    """Assigns variants by finding the closest real importation in time."""
    print("  Applying Mode 1: Temporal & Proportional Matching...")
    if real_imports.empty or sim_components.empty:
        return {}

    # Sort both by tick
    sim_components = sim_components.sort_values('first_tick').reset_index(drop=True)
    real_imports = real_imports.sort_values('tick').reset_index(drop=True)
    
    assignments = {}
    real_imports_used = [False] * len(real_imports)
    
    # For each simulated component, find the best available real import
    for sim_idx, sim_row in sim_components.iterrows():
        best_real_idx = -1
        min_dist = float('inf')
        
        # Find the closest *unused* real import
        for real_idx, real_row in real_imports.iterrows():
            if not real_imports_used[real_idx]:
                dist = abs(sim_row['first_tick'] - real_row['tick'])
                if dist < min_dist:
                    min_dist = dist
                    best_real_idx = real_idx
        
        if best_real_idx != -1:
            assignments[sim_row['component_id']] = real_imports.loc[best_real_idx, 'variant']
            real_imports_used[best_real_idx] = True # Mark as used
            
    return assignments

def mode2_bipartite_match(sim_components, real_imports, time_weight=0.7, max_time_penalty_days=90):
    """
    Assigns variants using optimized bipartite matching on time and size,
    with a heavy penalty for matches outside a reasonable time window.
    """
    print(f"  Applying Mode 2: Bipartite Matching (time_weight={time_weight}, penalty_window={max_time_penalty_days} days)...")
    if real_imports.empty or sim_components.empty:
        return {}

    # Normalize size columns for fair comparison in cost function
    sim_components['norm_size'] = minmax_scale(sim_components['component_size'])
    real_imports['norm_size'] = minmax_scale(real_imports['sample_count'])
    
    # Create the cost matrix
    cost_matrix = np.zeros((len(sim_components), len(real_imports)))
    
    # Normalize time cost by the penalty window, not the max time diff
    # This makes the time cost scale more aggressively.
    time_normalizer = float(max_time_penalty_days)
    
    for i, sim_row in sim_components.iterrows():
        for j, real_row in real_imports.iterrows():
            time_diff = abs(sim_row['first_tick'] - real_row['tick'])
            size_cost = abs(sim_row['norm_size'] - real_row['norm_size'])
            
            # --- NEW PENALTY LOGIC ---
            # Normalized time cost (scales from 0 to 1 within the window)
            norm_time_cost = time_diff / time_normalizer

            # The cost is a weighted average of time and size difference
            base_cost = (time_weight * norm_time_cost) + ((1 - time_weight) * size_cost)

            # Add a massive penalty if the time difference is outside our acceptable window
            if time_diff > max_time_penalty_days:
                base_cost += 1000  # A large number to make this match highly undesirable
            
            cost_matrix[i, j] = base_cost
            
    # Solve the assignment problem
    sim_indices, real_indices = linear_sum_assignment(cost_matrix)
    
    # Create the assignment map {component_id: variant}
    assignments = {}
    for i, j in zip(sim_indices, real_indices):
        component_id = sim_components.iloc[i]['component_id']
        variant = real_imports.iloc[j]['variant']
        assignments[component_id] = variant
        
    return assignments

def create_variant_labels(epihiper_df, schedule_df, mode):
    """
    Processes epihiper simulation, identifies components, and assigns variant labels
    based on a real-world importation schedule.
    """
    print("Step 1: Identifying infection events and transmission components in simulation...")
    # Filter for infection events
    infection_df = epihiper_df[epihiper_df['exit_state'].str.startswith('E')].copy()
    
    # Create graph and get component IDs
    component_df = create_component_table(infection_df)
    infection_df['alias_pid'] = (infection_df['pid'].astype(str) + '.' + infection_df['tick'].astype(str))
    # Merge component IDs back into the infection data
    merged_df = pd.merge(infection_df, component_df, on='alias_pid', how='left')
    
    # Get component properties: first tick and size
    component_summary = merged_df.groupby('component_id').agg(
        first_tick=('tick', 'min'),
        component_size=('pid', 'nunique') # Size is the number of unique people in the component
    ).reset_index()
    print(f"Found {len(component_summary)} unique transmission components in the simulation.")

    # --- Prepare Real-World Importation Data ---
    print("Step 2: Preparing real-world importation schedule...")
    # "Unroll" the schedule from the clusters column
    real_imports_list = []
    for _, row in schedule_df.iterrows():
        num_clusters = row['clusters']
        avg_sample_count = row['sample_count'] / num_clusters if num_clusters > 0 else 0
        for _ in range(num_clusters):
            real_imports_list.append({
                'tick': row['tick'],
                'variant': row['variant'],
                'sample_count': avg_sample_count
            })
    real_imports_df = pd.DataFrame(real_imports_list)
    print(f"Unrolled schedule into {len(real_imports_df)} individual importation events.")

    # --- Assign Variants based on Mode ---
    print(f"Step 3: Assigning variants to components using Mode {mode}...")
    assignment_map = {}
    if mode == 1:
        assignment_map = mode1_temporal_match(component_summary, real_imports_df)
    elif mode == 2:
        assignment_map = mode2_bipartite_match(component_summary, real_imports_df, time_weight=0.7, max_time_penalty_days=90)
    
    if not assignment_map:
        print("Warning: No variant assignments were made. Components will not be labeled.")
        merged_df['variant_label'] = 'unassigned'
        return merged_df

    # --- Apply Labels ---
    print("Step 4: Applying variant labels to the full simulation dataframe...")
    # Map the assigned variants to the component summary
    component_summary['variant_label'] = component_summary['component_id'].map(assignment_map)
    # Fill any unassigned components (if sim has more components than real imports)
    component_summary['variant_label'].fillna('unassigned', inplace=True)
    
    # Merge the final labels into the full infection dataframe
    final_df = pd.merge(merged_df, component_summary[['component_id', 'variant_label']], on='component_id', how='left')
    
    # propagate component_id and variant_label.
    #create alias_pid for epihiper_df
    epihiper_df['alias_pid'] = (epihiper_df['pid'].astype(str) + '.' + epihiper_df['tick'].astype(str))
    # link variant_label to every event in the full epihiper_df via pid
    pid_to_label = final_df.set_index('alias_pid')['variant_label'].to_dict()
    pid_to_component = final_df.set_index('alias_pid')['component_id'].to_dict()

    epihiper_df['variant_label'] = epihiper_df['alias_pid'].map(pid_to_label)
    epihiper_df['component_id'] = epihiper_df['alias_pid'].map(pid_to_component)
    
    # Forward fill the labels for each person's timeline
    #check if epihiper_df is sorted by only tick, prefer to keep it in orginal order
    if not epihiper_df['tick'].is_monotonic_increasing:
        print("Warning: epihiper_df is not sorted by 'tick'. Sorting now.")
        epihiper_df.sort_values(['tick'], inplace=True)
    epihiper_df['variant_label'] = epihiper_df.groupby('pid')['variant_label'].ffill()
    epihiper_df['variant_label'].fillna('background', inplace=True) # Label non-infected people
    epihiper_df['component_id'] = epihiper_df.groupby('pid')['component_id'].ffill()
    epihiper_df['component_id'].fillna(-1, inplace=True) # Label non-infected people

    return epihiper_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label epihiper simulation components with variants from a real-world schedule.")
    parser.add_argument("--epihiper_input", required=True, type=str, help="Path to the epihiper simulation output CSV file.")
    parser.add_argument("--schedule_input", required=True, type=str, help="Path to the importation schedule CSV file (generated by seed_seq_prep.py).")
    parser.add_argument("--output", required=True, type=str, help="Path for the output CSV file with variant labels.")
    parser.add_argument("--mode", required=True, type=int, choices=[1, 2], help="Labeling mode: 1 (Temporal/Proportional) or 2 (Bipartite Time & Size).")
    
    args = parser.parse_args()

    print(f"Loading epihiper data from: {args.epihiper_input}")
    epi_df = pd.read_csv(args.epihiper_input)
    
    print(f"Loading schedule data from: {args.schedule_input}")
    sched_df = pd.read_csv(args.schedule_input)
    
    # Call the main function
    labeled_df = create_variant_labels(epi_df, sched_df, args.mode)
    
    print(f"Saving labeled data to: {args.output}")
    #if the output file name doesn't end  with .gz add it
    if not args.output.endswith('.gz'):
        args.output += '.gz'
    labeled_df.to_csv(args.output, index=False, compression='gzip')
    
    print("Script finished.")
    print("\nVariant Label Distribution in Output:")
    print(labeled_df['variant_label'].value_counts())