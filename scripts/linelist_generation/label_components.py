import argparse
import pandas as pd
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import minmax_scale

# --- Graph and Component Functions (from original script) ---
def create_graph_classic(epihiper_df):
    
    print("Step 1: Identifying infection events and transmission components in simulation...")
    # Filter for infection events
    infection_events_df = epihiper_df[epihiper_df['exit_state'].str.startswith('E')].copy()

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
        G_full.add_nodes_from(infection_events_df['alias_pid'])
        #only rows with contact_pid != -1
        # Filter the DataFrame to include only rows where contact_pid is not -1
        no_imports = infection_events_df[
            (infection_events_df['contact_pid'] != "-1") & 
            (infection_events_df['alias_contact'] != "-1")
        ]
        edges = zip(no_imports['alias_contact'], no_imports['alias_pid'], no_imports['tick'])
        # OPTIMIZATION: Use add_edges_from for much faster graph construction
        G_full.add_edges_from((u, v, {'tick': tick}) for u, v, tick in edges)

        G_full.add_edges_from(edges)
    return G_full, infection_events_df 

import networkx as nx
import pandas as pd

def calculate_Re(infection_graph: nx.DiGraph, infection_df: pd.DataFrame = None, time_col: str = 'tick'):
    """
    Calculates the effective reproduction number (R_e) from the transmission graph.
    
    Args:
        infection_graph: NetworkX DiGraph representing transmissions. Nodes should be 'alias_pid'.
        infection_df: A DataFrame of the initial infection events (the 'E' states). 
                      Required to calculate time-varying R_e(t).
        time_col: The column in the dataframe representing time (e.g., 'tick' or 'date').
        
    Returns:
        overall_Re (float): The average R_e for the entire simulation.
        Re_over_time (pd.Series): The average R_e grouped by the infector's time of infection.
                                  Returns None if infection_df is not provided.
    """
    # 1. Calculate overall R_e
    # This is the average out-degree across all infected individuals in the graph.
    out_degrees = dict(infection_graph.out_degree())
    
    if len(out_degrees) == 0:
        return 0.0, None
        
    overall_Re = sum(out_degrees.values()) / len(out_degrees)
    print(f"Overall Epidemic R_e: {overall_Re:.3f}")
    
    # 2. Calculate time-varying R_e(t)
    Re_over_time = None
    if infection_df is not None:
        print(f"Calculating time-varying R_e based on '{time_col}'...")
        
        # Convert the out-degree dictionary to a DataFrame
        degree_df = pd.DataFrame(list(out_degrees.items()), columns=['alias_pid', 'out_degree'])
        
        # Ensure the infection_df has the alias_pid to join on
        df_copy = infection_df.copy()
        if 'alias_pid' not in df_copy.columns:
            df_copy['alias_pid'] = df_copy['pid'].astype(str) + '.' + df_copy['tick'].astype(str)
            
        # Merge the out-degree (secondary infections) back onto the infection events
        merged = pd.merge(df_copy, degree_df, on='alias_pid', how='inner')
        
        # Group by the time the *infector* was exposed, and average their secondary infections
        Re_over_time = merged.groupby(time_col)['out_degree'].mean()
        
    return overall_Re, Re_over_time

def create_component_table(infection_graph):
    full_cascade_components = list(nx.weakly_connected_components(infection_graph))
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
    (Optimized with NumPy broadcasting for massive component lists).
    """
    print(f"  Applying Mode 2: Bipartite Matching (time_weight={time_weight}, penalty_window={max_time_penalty_days} days)...")
    if real_imports.empty or sim_components.empty:
        return {}

    # Normalize size columns for fair comparison in cost function
    sim_components['norm_size'] = minmax_scale(sim_components['component_size'])
    real_imports['norm_size'] = minmax_scale(real_imports['sample_count'])
    
    # --- VECTORIZED MATRIX CALCULATION ---
    print("  Calculating cost matrix...")
    # 1. Extract values as numpy arrays and reshape for broadcasting
    # sim arrays become shape (N, 1)
    sim_ticks = sim_components['first_tick'].values[:, np.newaxis]
    sim_sizes = sim_components['norm_size'].values[:, np.newaxis]
    
    # real arrays become shape (1, M)
    real_ticks = real_imports['tick'].values[np.newaxis, :]
    real_sizes = real_imports['norm_size'].values[np.newaxis, :]
    
    # 2. Compute differences (Broadcasting automatically creates N x M matrices)
    time_diffs = np.abs(sim_ticks - real_ticks)
    size_costs = np.abs(sim_sizes - real_sizes)
    
    # 3. Calculate Base Cost
    time_normalizer = float(max_time_penalty_days)
    norm_time_cost = time_diffs / time_normalizer
    cost_matrix = (time_weight * norm_time_cost) + ((1 - time_weight) * size_costs)
    
    # 4. Apply massive penalty if out of window
    cost_matrix = np.where(time_diffs > max_time_penalty_days, cost_matrix + 1000, cost_matrix)
            
    # --- SOLVE THE ASSIGNMENT ---
    print(f"  Solving linear sum assignment for {cost_matrix.shape[0]}x{cost_matrix.shape[1]} matrix...")
    sim_indices, real_indices = linear_sum_assignment(cost_matrix)
    
    # --- CREATE THE ASSIGNMENT MAP ---
    assignments = {}
    
    # Extract native arrays for fast lookup
    comp_ids = sim_components['component_id'].values
    variants = real_imports['variant'].values
    
    for i, j in zip(sim_indices, real_indices):
        assignments[comp_ids[i]] = variants[j]
        
    return assignments


def find_components(epihiper_df):
    #count the number of introduction events by counting number of contact_pid == -1 in the E states
    print(f"Identified {len(epihiper_df[(epihiper_df['exit_state'].str.startswith('E')) & (epihiper_df['contact_pid'] == "-1")])} introduction events (contact_pid == -1) in the simulation.")
    infection_graph, infection_df = create_graph_classic(epihiper_df)
    # Create graph and get component IDs
    component_df = create_component_table(infection_graph)
    #infection_df = epihiper_df[epihiper_df['exit_state'].str.startswith('E')].copy()
    #infection_df['alias_pid'] = (infection_df['pid'].astype(str) + '.' + infection_df['tick'].astype(str))
    # Merge component IDs back into the infection data
    merged_df = pd.merge(infection_df, component_df, on='alias_pid', how='left')
    
    # Get component properties: first tick and size
    component_summary = merged_df.groupby('component_id').agg(
        first_tick=('tick', 'min'),
        component_size=('pid', 'nunique') # Size is the number of unique people in the component
    ).reset_index()
    print(f"Found {len(component_summary)} unique transmission chains in the simulation.")
    return merged_df, component_summary

def create_labels(epihiper_df, schedule_df, mode):
    """
    Identifies components and propagates aliases. Optionally assigns variants 
    based on a real-world importation schedule.
    """
    # 1. Get components using your new function
    merged_df, component_summary = find_components(epihiper_df)

    # 2. Assign Variants OR Skip (just_components)
    if mode == 'just_components' or schedule_df is None:
        print("  Skipping schedule matching (mode: just_components).")
        component_summary['variant_label'] = 'unassigned'
    else:
        print("  Preparing real-world importation schedule...")
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
        
        # Apply Matching
        assignment_map = {}
        if mode == 'variant_temporal':
            assignment_map = mode1_temporal_match(component_summary, real_imports_df)
        elif mode == 'variant_bipartite':
            assignment_map = mode2_bipartite_match(component_summary, real_imports_df, time_weight=0.7, max_time_penalty_days=90)
            
        component_summary['variant_label'] = component_summary['component_id'].map(assignment_map)
        component_summary['variant_label'].fillna('unassigned', inplace=True)

    # 3. Propagate component_id, variant_label, alias_pid, and alias_contact
    print("  Propagating labels and aliases to the full simulation dataframe...")
    final_df = pd.merge(merged_df, component_summary[['component_id', 'variant_label']], on='component_id', how='left')

    # Prepare Source of Truth
    label_source = final_df[['tick', 'pid', 'component_id', 'variant_label', 'alias_pid', 'alias_contact']].copy()
    label_source = label_source.sort_values('tick')

    # Prepare Target
    if not epihiper_df['tick'].is_monotonic_increasing:
        print("  Sorting epihiper_df by tick for label propagation...")
        epihiper_df.sort_values('tick', inplace=True)

    # --- CRITICAL FIX PRESERVED: Align Data Types Before Merge ---
    epihiper_df['pid'] = epihiper_df['pid'].astype(str)
    label_source['pid'] = label_source['pid'].astype(str)
    
    epihiper_df['tick'] = epihiper_df['tick'].astype(int)
    label_source['tick'] = label_source['tick'].astype(int)
    # -------------------------------------------------------------

    # Clean up existing columns to avoid suffix conflicts (_x, _y)
    cols_to_drop = [c for c in ['component_id', 'variant_label', 'alias_pid', 'alias_contact'] if c in epihiper_df.columns]
    if cols_to_drop:
        epihiper_df.drop(columns=cols_to_drop, inplace=True)

    # Perform Time-Aware Broadcast
    epihiper_df = pd.merge_asof(
        epihiper_df,
        label_source,
        on='tick',
        by='pid',
        direction='backward'
    )

    # 4. Fill Background/Unlabeled for agents with no exposure event
    epihiper_df['variant_label'] = epihiper_df['variant_label'].fillna('background')
    epihiper_df['component_id'] = epihiper_df['component_id'].fillna(-1).astype(int)
    epihiper_df['alias_contact'] = epihiper_df['alias_contact'].fillna("-1")
    epihiper_df['alias_pid'] = epihiper_df['alias_pid'].fillna(epihiper_df['pid'].astype(str) + '.' + epihiper_df['tick'].astype(str))

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
